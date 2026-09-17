#!/usr/bin/env python3
"""Benchmark the sampling-support output path: upstream bitmap vs compact pack.

Measures the *whole* per-step path a model runner pays for the sampling mask:

  upstream (vLLM v0.29.0 ``SamplingMaskTensors``):
    GPU  _pack_sampling_mask_kernel  -> [B, V/8] uint8 bitmap + counts
    D2H  bitmap + counts
    CPU  np.unpackbits(full vocab) -> np.nonzero -> CSR (token_ids, offsets)

  upstream_main (vLLM main after #54901, 0.29.1+):
    GPU  _compact_sampling_mask_kernel -> [B, max_top_k] int32 ids + counts AND the bitmap
    D2H  ids + counts + bitmap (the bitmap is the overflow fallback)
    CPU  per-row Python loop slicing ids (unpackbits only for overflowed rows)

  compact (this repository ``support_pack_out``):
    GPU  _support_pack_kernel -> [B, K] int32 ids + counts (+ logz, sampled logprob)
    D2H  ids + counts
    CPU  ragged slice -> CSR (token_ids, offsets)

Both produce identical CSR lists; the benchmark asserts that on every trial.
GPU time uses CUDA events; the D2H+CPU stage is wall-clock after a stream
sync, because that stage runs on the host critical path in vLLM's async
output pipeline. Inputs are top-k-masked random logits (finite only inside
each row's top-k), which is the only regime the upstream feature supports.

The upstream kernels and ``tolists`` logic are copied verbatim from vLLM
v0.29.0 and from main after #54901 (vllm/v1/worker/gpu/sample/output.py) so
the comparison runs in one process without a vLLM install; the v0.29.0 source
hash is recorded in the output when ``--vllm-source`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from l20_stack.ops.triton_support_pack import support_pack_out

UPSTREAM_SOURCE = "vllm/v1/worker/gpu/sample/output.py@v0.29.0"


# ---- verbatim copy of upstream vLLM v0.29.0 _pack_sampling_mask_kernel ----
@triton.jit
def _pack_sampling_mask_kernel(
    logits_ptr,
    logits_row_stride,
    logits_col_stride,
    num_sampled_tokens_ptr,
    packed_mask_ptr,
    packed_mask_row_stride,
    counts_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_active = tl.load(num_sampled_tokens_ptr + req_idx) > 0
    count = tl.zeros((), dtype=tl.int32)

    for start_idx in range(0, vocab_size, BLOCK_SIZE):
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        valid = offsets < vocab_size
        logits = tl.load(
            logits_ptr + req_idx * logits_row_stride + offsets * logits_col_stride,
            mask=valid,
            other=-float("inf"),
        )
        keep = (logits > -float("inf")) & (logits < float("inf")) & is_active
        count += tl.sum(keep).to(tl.int32)

        keep = tl.reshape(keep.to(tl.int32), (BLOCK_SIZE // 8, 8))
        bit_shifts = tl.arange(0, 8)[None, :]
        packed = tl.sum(keep << bit_shifts, axis=1).to(tl.uint8)
        byte_offsets = start_idx // 8 + tl.arange(0, BLOCK_SIZE // 8)
        tl.store(
            packed_mask_ptr + req_idx * packed_mask_row_stride + byte_offsets,
            packed,
            mask=byte_offsets < tl.cdiv(vocab_size, 8),
        )

    tl.store(counts_ptr + req_idx, count)


# ---- verbatim copy of upstream vLLM main (after #54901) _compact_sampling_mask_kernel ----
@triton.jit
def _compact_sampling_mask_kernel(
    logits_ptr,
    logits_row_stride,
    logits_col_stride,
    num_sampled_tokens_ptr,
    token_ids_ptr,
    token_ids_row_stride,
    packed_mask_ptr,
    packed_mask_row_stride,
    counts_ptr,
    vocab_size,
    max_num_kept,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_active = tl.load(num_sampled_tokens_ptr + req_idx) > 0
    count = tl.zeros((), dtype=tl.int32)

    for start_idx in range(0, vocab_size, BLOCK_SIZE):
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(
            logits_ptr + req_idx * logits_row_stride + offsets * logits_col_stride,
            mask=offsets < vocab_size,
            other=-float("inf"),
        )
        keep = (logits > -float("inf")) & (logits < float("inf")) & is_active
        keep_i32 = keep.to(tl.int32)
        pos = count + tl.cumsum(keep_i32, axis=0) - keep_i32
        tl.store(
            token_ids_ptr + req_idx * token_ids_row_stride + pos,
            offsets.to(tl.int32),
            mask=keep & (pos < max_num_kept),
        )
        count += tl.sum(keep_i32)

        bits = tl.reshape(keep_i32, (BLOCK_SIZE // 8, 8)) << tl.arange(0, 8)[None, :]
        byte_offsets = start_idx // 8 + tl.arange(0, BLOCK_SIZE // 8)
        tl.store(
            packed_mask_ptr + req_idx * packed_mask_row_stride + byte_offsets,
            tl.sum(bits, axis=1).to(tl.uint8),
            mask=byte_offsets < tl.cdiv(vocab_size, 8),
        )

    tl.store(counts_ptr + req_idx, count)


def upstream_main_tolists(ids_cpu, packed_cpu, counts_cpu, vocab_size):
    # verbatim logic of SamplingMaskTensors.tolists on main (after #54901)
    counts = counts_cpu.numpy()
    token_ids = ids_cpu.numpy()
    packed_mask = packed_cpu.numpy()
    width = token_ids.shape[1]

    def support(row: int) -> np.ndarray:
        if counts[row] <= width:
            return token_ids[row, : counts[row]]
        bits = np.unpackbits(packed_mask[row], count=vocab_size, bitorder="little")
        return np.flatnonzero(bits).astype(np.int32, copy=False)

    supports = [support(row) for row in range(len(counts))]
    offsets = np.zeros(len(supports) + 1, dtype=np.int64)
    np.cumsum([len(s) for s in supports], out=offsets[1:])
    return np.concatenate(supports), offsets


def upstream_pack(logits, num_sampled):
    num_reqs, vocab_size = logits.shape
    packed_width = (vocab_size + 7) // 8
    packed = torch.empty((num_reqs, packed_width), dtype=torch.uint8, device=logits.device)
    counts = torch.empty(num_reqs, dtype=torch.int32, device=logits.device)
    _pack_sampling_mask_kernel[(num_reqs,)](
        logits, logits.stride(0), logits.stride(1), num_sampled, packed, packed.stride(0), counts,
        vocab_size, BLOCK_SIZE=8192,
    )
    return packed, counts


def upstream_tolists(packed_cpu, counts_cpu, num_sampled_np, vocab_size):
    # verbatim logic of SamplingMaskTensors.tolists (v0.29.0)
    sampled_rows = np.flatnonzero(num_sampled_np)
    counts = counts_cpu.numpy()[sampled_rows]
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, dtype=np.int64, out=offsets[1:])
    unpacked = np.unpackbits(packed_cpu.numpy()[sampled_rows], axis=1, count=vocab_size, bitorder="little")
    token_ids = np.nonzero(unpacked)[1].astype(np.int32, copy=False)
    return token_ids, offsets


def compact_tolists(ids_cpu, counts_cpu, num_sampled_np, max_support):
    sampled_rows = np.flatnonzero(num_sampled_np)
    counts = counts_cpu.numpy()[sampled_rows]
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, dtype=np.int64, out=offsets[1:])
    ids = ids_cpu.numpy()[sampled_rows]
    col = np.arange(max_support)[None, :]
    token_ids = ids[col < counts[:, None]]
    return token_ids, offsets


def make_inputs(batch, vocab, top_k, gen, device):
    logits = torch.randn((batch, vocab), device=device, generator=gen)
    thresh = torch.topk(logits, top_k, dim=1).values[:, -1:]
    logits = torch.where(logits >= thresh, logits, torch.full_like(logits, -float("inf")))
    num_sampled = torch.ones(batch, device=device, dtype=torch.int32)
    sampled = torch.argmax(logits, dim=1)
    return logits, num_sampled, sampled


def time_gpu(fn, rounds):
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(rounds)]
    for s, e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in ev]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=151_936)
    ap.add_argument("--batches", default="8,32,128,256")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-support", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=113)
    ap.add_argument("--vllm-source", type=Path, help="path to a vLLM checkout to hash output.py")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(args.seed)
    rows = []
    for batch in [int(b) for b in args.batches.split(",")]:
        logits, num_sampled, sampled = make_inputs(batch, args.vocab, args.top_k, gen, device)
        ns_np = num_sampled.cpu().numpy()
        K = args.max_support
        # outputs
        packed = torch.empty((batch, (args.vocab + 7) // 8), dtype=torch.uint8, device=device)
        counts_u = torch.empty(batch, dtype=torch.int32, device=device)
        ids = torch.empty((batch, K), dtype=torch.int32, device=device)
        counts_c = torch.empty(batch, dtype=torch.int32, device=device)
        ovf = torch.empty(batch, dtype=torch.int32, device=device)
        logz = torch.empty(batch, dtype=torch.float32, device=device)
        slp = torch.empty(batch, dtype=torch.float32, device=device)
        pin_packed = torch.empty_like(packed, device="cpu", pin_memory=True)
        pin_counts_u = torch.empty_like(counts_u, device="cpu", pin_memory=True)
        pin_ids = torch.empty_like(ids, device="cpu", pin_memory=True)
        pin_counts_c = torch.empty_like(counts_c, device="cpu", pin_memory=True)
        # main path: ids sized by max top_k (not rounded), plus the bitmap
        ids_m = torch.empty((batch, args.top_k), dtype=torch.int32, device=device)
        packed_m = torch.empty_like(packed)
        counts_m = torch.empty(batch, dtype=torch.int32, device=device)
        pin_ids_m = torch.empty_like(ids_m, device="cpu", pin_memory=True)
        pin_packed_m = torch.empty_like(packed_m, device="cpu", pin_memory=True)
        pin_counts_m = torch.empty_like(counts_m, device="cpu", pin_memory=True)

        def um_gpu():
            _compact_sampling_mask_kernel[(batch,)](
                logits, logits.stride(0), logits.stride(1), num_sampled, ids_m, ids_m.stride(0),
                packed_m, packed_m.stride(0), counts_m, args.vocab, args.top_k, BLOCK_SIZE=8192,
            )

        def um_host():
            pin_ids_m.copy_(ids_m, non_blocking=True); pin_packed_m.copy_(packed_m, non_blocking=True)
            pin_counts_m.copy_(counts_m, non_blocking=True)
            torch.cuda.synchronize()
            return upstream_main_tolists(pin_ids_m, pin_packed_m, pin_counts_m, args.vocab)

        def up_gpu():
            _pack_sampling_mask_kernel[(batch,)](
                logits, logits.stride(0), logits.stride(1), num_sampled, packed, packed.stride(0), counts_u,
                args.vocab, BLOCK_SIZE=8192,
            )

        def cp_gpu():
            support_pack_out(
                logits, num_sampled, sampled, out_token_ids=ids, out_counts=counts_c, out_overflow=ovf,
                out_logz=logz, out_sampled_logprob=slp, max_support=K,
            )

        def up_host():
            pin_packed.copy_(packed, non_blocking=True); pin_counts_u.copy_(counts_u, non_blocking=True)
            torch.cuda.synchronize()
            return upstream_tolists(pin_packed, pin_counts_u, ns_np, args.vocab)

        def cp_host():
            pin_ids.copy_(ids, non_blocking=True); pin_counts_c.copy_(counts_c, non_blocking=True)
            torch.cuda.synchronize()
            return compact_tolists(pin_ids, pin_counts_c, ns_np, K)

        # warmup + equivalence
        up_gpu(); cp_gpu(); um_gpu(); torch.cuda.synchronize()
        a = up_host(); b = cp_host(); c = um_host()
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), "CSR mismatch"
        assert np.array_equal(a[0], c[0]) and np.array_equal(a[1], c[1]), "CSR mismatch (main)"
        assert int(ovf.max()) == 0
        ref_logz = torch.logsumexp(logits, dim=1)
        assert torch.allclose(logz, ref_logz, atol=1e-4)
        assert torch.allclose(slp, logits.gather(1, sampled[:, None]).squeeze(1) - ref_logz, atol=1e-4)

        trial_rows = []
        for t in range(args.trials):
            order = ["upstream", "upstream_main", "compact"]
            if t % 2 == 1:
                order = order[::-1]
            fns = {"upstream": (up_gpu, up_host), "upstream_main": (um_gpu, um_host), "compact": (cp_gpu, cp_host)}
            r = {}
            for name in order:
                g, h = fns[name]
                gpu_ms = time_gpu(g, args.rounds)
                host_ms = []
                for _ in range(args.rounds):
                    g(); torch.cuda.synchronize()
                    t0 = time.perf_counter(); h(); host_ms.append(1e3 * (time.perf_counter() - t0))
                r[name] = {"gpu_ms_median": statistics.median(gpu_ms), "host_ms_median": statistics.median(host_ms)}
            trial_rows.append(r)
        agg = {}
        for name in ("upstream", "upstream_main", "compact"):
            agg[name] = {
                "gpu_ms": statistics.median(x[name]["gpu_ms_median"] for x in trial_rows),
                "host_ms": statistics.median(x[name]["host_ms_median"] for x in trial_rows),
            }
            agg[name]["total_ms"] = agg[name]["gpu_ms"] + agg[name]["host_ms"]
        agg["bytes_d2h"] = {
            "upstream": batch * ((args.vocab + 7) // 8 + 4),
            "upstream_main": batch * ((args.vocab + 7) // 8 + args.top_k * 4 + 4),
            "compact": batch * (K * 4 + 4),
        }
        agg["speedup_total"] = agg["upstream"]["total_ms"] / agg["compact"]["total_ms"]
        agg["speedup_host"] = agg["upstream"]["host_ms"] / agg["compact"]["host_ms"]
        agg["speedup_gpu"] = agg["upstream"]["gpu_ms"] / agg["compact"]["gpu_ms"]
        agg["speedup_total_vs_main"] = agg["upstream_main"]["total_ms"] / agg["compact"]["total_ms"]
        row = {"batch": batch, "vocab": args.vocab, "top_k": args.top_k, "max_support": K, "trials": trial_rows, **agg}
        rows.append(row)
        print(
            f"B={batch:4d} v0.29.0 gpu {agg['upstream']['gpu_ms']:.3f} host {agg['upstream']['host_ms']:.3f} | "
            f"main gpu {agg['upstream_main']['gpu_ms']:.3f} host {agg['upstream_main']['host_ms']:.3f} | "
            f"compact gpu {agg['compact']['gpu_ms']:.3f} host {agg['compact']['host_ms']:.3f} | "
            f"total x{agg['speedup_total']:.2f} vs main x{agg['speedup_total_vs_main']:.2f}",
            flush=True,
        )

    prov = {}
    try:
        prov["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        prov["dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
    except Exception:
        pass
    prov["kernel_source_sha256"] = hashlib.sha256((REPO_ROOT / "src/l20_stack/ops/triton_support_pack.py").read_bytes()).hexdigest()
    prov["upstream_copy_of"] = UPSTREAM_SOURCE
    prov["upstream_main_copy_of"] = "vllm/v1/worker/gpu/sample/output.py@main after #54901 (e30bf70c)"
    if args.vllm_source:
        prov["upstream_output_py_sha256"] = hashlib.sha256((args.vllm_source / "vllm/v1/worker/gpu/sample/output.py").read_bytes()).hexdigest()
    out = {
        "schema_version": 1,
        "result_type": "support_pack_path_microbenchmark",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__, "python": sys.version.split()[0]},
        "protocol": {"rounds": args.rounds, "trials": args.trials, "gpu_timing": "cuda events", "host_timing": "wall clock after sync: D2H copy from pinned buffer + CSR construction", "order": "alternating per trial", "equivalence": "CSR token_ids/offsets identical; logz and sampled logprob vs torch reference atol 1e-4"},
        "provenance": prov,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
