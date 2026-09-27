#!/usr/bin/env python3
"""Pairing swap and single-request grid on isolated attention kernels, called the way vLLM calls them.

One attention layer's forward over a mixed batch: [decode row] + (q1 @ k_a) + (q2 @ k_b), paged KV
(block size 16, bf16, causal, per-request depths), exactly one kernel call per step as in vLLM's
FLASH_ATTN backend (FA3: with the AOT scheduler metadata vLLM builds). States A/B are the six
configurations of measure_pairing_swap.py. The grid is single requests q x k with no decode row.

Kernels: vLLM FA2 (fa_version=2), vLLM FA3 (fa_version=3), FlashInfer BatchPrefillWithPagedKVCache.
Timing: CUDA events around the kernel call only (metadata is built before, like vLLM's metadata
builder); L2 is flushed with a 256 MB write before every call, because in a model step the other
layers evict it; states are interleaved in blocks of 10 calls.

  python scripts/measure_kernel_pairing.py --output OUT.json [--shape qwen3-4b|qwen25-7b] [--calls 200]
  python scripts/measure_kernel_pairing.py --analyze OUT.json [--model-pairswap pairswap.json] [--layers 36]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_pairing_swap import CONFIGS  # noqa: E402

SHAPES = {"qwen3-4b": (32, 8, 128), "qwen25-7b": (28, 4, 128)}
BLOCK = 16
DECODE_DEPTH = 1024
GRID_Q = [128, 256, 512, 1024, 2048]
GRID_K = [0, 2048, 4096, 8192, 16384, 32768]


def batch_for(state, cfg):
    _, ka, kb, qa, qb = cfg
    q1, q2 = (qa, qb) if state == "A" else (qb, qa)
    return [(1, DECODE_DEPTH), (q1, ka), (q2, kb)]


class Runner:
    """Holds one paged KV cache big enough for every batch; builds per-batch metadata and calls a kernel."""

    def __init__(self, hq, hkv, d, max_tokens=80_000):
        import torch
        self.t = torch
        self.hq, self.hkv, self.d = hq, hkv, d
        nblocks = max_tokens // BLOCK + 64
        g = torch.Generator(device="cuda").manual_seed(0)
        self.k = torch.randn(nblocks, BLOCK, hkv, d, device="cuda", dtype=torch.bfloat16, generator=g)
        self.v = torch.randn(nblocks, BLOCK, hkv, d, device="cuda", dtype=torch.bfloat16, generator=g)
        self.flush = torch.empty(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
        self.scale = d ** -0.5

    def prepare(self, kernel, batch):
        """batch: list of (q_len, depth). KV of each request = depth + q_len tokens, in disjoint blocks."""
        t = self.t
        qlens = [q for q, _ in batch]; kvlens = [k + q for q, k in batch]
        ntok = sum(qlens)
        q = t.randn(ntok, self.hq, self.d, device="cuda", dtype=t.bfloat16)
        out = t.empty_like(q)
        cu_q = t.tensor([0] + list(itertools.accumulate(qlens)), device="cuda", dtype=t.int32)
        nb = [math.ceil(L / BLOCK) for L in kvlens]
        maxb = max(nb)
        bt = t.zeros(len(batch), maxb, device="cuda", dtype=t.int32)
        start = 0
        for i, n in enumerate(nb):
            bt[i, :n] = t.arange(start, start + n, device="cuda", dtype=t.int32); start += n
        seqused = t.tensor(kvlens, device="cuda", dtype=t.int32)
        if kernel in ("fa2", "fa3"):
            from vllm.vllm_flash_attn import flash_attn_varlen_func
            meta = None
            if kernel == "fa3":
                from vllm.vllm_flash_attn import get_scheduler_metadata
                meta = get_scheduler_metadata(batch_size=len(batch), max_seqlen_q=max(qlens), max_seqlen_k=max(kvlens),
                                              num_heads_q=self.hq, num_heads_kv=self.hkv, headdim=self.d, cache_seqlens=seqused,
                                              qkv_dtype=t.bfloat16, cu_seqlens_q=cu_q, page_size=BLOCK, causal=True,
                                              window_size=(-1, -1), num_splits=0)
            ver = 2 if kernel == "fa2" else 3
            return lambda: flash_attn_varlen_func(q=q, k=self.k, v=self.v, out=out, cu_seqlens_q=cu_q, max_seqlen_q=max(qlens),
                                                  seqused_k=seqused, max_seqlen_k=max(kvlens), softmax_scale=self.scale, causal=True,
                                                  block_table=bt, scheduler_metadata=meta, fa_version=ver)
        if kernel == "flashinfer":
            import flashinfer
            ws = t.empty(256 * 1024 * 1024, dtype=t.uint8, device="cuda")
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
            kv_indptr = t.tensor([0] + list(itertools.accumulate(nb)), device="cuda", dtype=t.int32)
            kv_indices = t.cat([bt[i, :n] for i, n in enumerate(nb)])
            last = t.tensor([L - (n - 1) * BLOCK for L, n in zip(kvlens, nb)], device="cuda", dtype=t.int32)
            w.plan(cu_q, kv_indptr, kv_indices, last, self.hq, self.hkv, self.d, BLOCK, causal=True,
                   sm_scale=self.scale, q_data_type=t.bfloat16, kv_data_type=t.bfloat16)
            return lambda: w.run(q, (self.k, self.v), out=out)
        raise ValueError(kernel)

    def time(self, fns, calls, block=10):
        """fns: dict state -> callable; interleave states in blocks; returns dict state -> list of ms."""
        t = self.t
        res = {s: [] for s in fns}
        for f in fns.values():
            for _ in range(5):
                f()
        t.cuda.synchronize()
        order = list(fns)
        rounds = math.ceil(calls / block)
        for r in range(rounds):
            for s in (order if r % 2 == 0 else order[::-1]):
                for _ in range(block):
                    self.flush.zero_()
                    e0, e1 = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
                    e0.record(); fns[s](); e1.record()
                    e1.synchronize()
                    res[s].append(e0.elapsed_time(e1))
        return {s: v[:calls] for s, v in res.items()}


def run(args):
    import torch
    hq, hkv, d = SHAPES[args.shape]
    R = Runner(hq, hkv, d)
    kernels = []
    for k in args.kernels.split(","):
        try:
            R.prepare(k, [(128, 1024)])()
            torch.cuda.synchronize()
            kernels.append(k)
        except Exception as e:  # recorded, not fatal
            print(f"kernel {k} unavailable: {type(e).__name__}: {e}", flush=True)
    out = {"shape": args.shape, "heads": [hq, hkv, d], "gpu": torch.cuda.get_device_name(0),
           "sms": torch.cuda.get_device_properties(0).multi_processor_count, "torch": torch.__version__,
           "calls": args.calls, "kernels": kernels, "pair": {}, "grid": {}}
    try:
        import vllm, flashinfer
        out["vllm"], out["flashinfer"] = vllm.__version__, flashinfer.__version__
    except Exception:
        pass
    for k in kernels:
        out["pair"][k] = []
        for cfg in CONFIGS:
            fns = {s: R.prepare(k, batch_for(s, cfg)) for s in ("A", "B")}
            ts = R.time(fns, args.calls)
            row = {"label": cfg[0], "k": [cfg[1], cfg[2]], "q": [cfg[3], cfg[4]], "ms_A": ts["A"], "ms_B": ts["B"],
                   "median_A": statistics.median(ts["A"]), "median_B": statistics.median(ts["B"])}
            row["delta"] = row["median_A"] - row["median_B"]
            out["pair"][k].append(row)
            print(f"{k:10s} {cfg[0]:34s} A {row['median_A']:.4f}  B {row['median_B']:.4f}  delta {row['delta']*1000:+.1f} us", flush=True)
        out["grid"][k] = []
        for q, kv in itertools.product(GRID_Q, GRID_K):
            ts = R.time({"x": R.prepare(k, [(q, kv)])}, args.grid_calls)["x"]
            out["grid"][k].append({"q": q, "k": kv, "median": statistics.median(ts), "p10": sorted(ts)[len(ts) // 10]})
        print(f"{k:10s} grid done", flush=True)
    args.output.write_text(json.dumps(out) + "\n")


# ---------------------------------------------------------------- analysis

def ctas(batch, bq, bk, heads):
    """CTA works (in key tiles) for a causal varlen prefill: one CTA per (request, query tile, query head)."""
    w = []
    for q, k in batch:
        for j in range(math.ceil(q / bq)):
            end = min((j + 1) * bq, q)
            w += [math.ceil((k + end) / bk)] * heads
    return w


def makespan(works, slots):
    loads = [0] * slots
    import heapq
    heapq.heapify(loads)
    for x in sorted(works, reverse=True):
        heapq.heappush(loads, heapq.heappop(loads) + x)
    return max(loads)


def work_units(batch):
    return sum(q * k + q * (q + 1) / 2 for q, k in batch)


def fit(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def analyze(args):
    d = json.loads(args.analyze.read_text())
    hq = d["heads"][0]; sms = d["sms"]
    model_ps = None
    if args.model_pairswap:
        model_ps = {c["label"]: c.get("delta_A_minus_B_ms") for c in json.load(open(args.model_pairswap))["configs"]}
    report = {"gpu": d["gpu"], "shape": d["shape"], "sms": sms, "kernels": {}}
    for k in d["kernels"]:
        grid = d["grid"][k]; pair = d["pair"][k]
        gb = [[(g["q"], g["k"])] for g in grid]; gy = [g["median"] for g in grid]
        # W model
        aW, bW = fit([work_units(b) / 1e6 for b in gb], gy)
        # split model at kernel level (HK3): least squares on [1, qk, q(q+1)/2]
        import numpy as np
        A = np.array([[1.0, q * kk / 1e6, q * (q + 1) / 2e6] for [(q, kk)] in gb])
        coef, *_ = np.linalg.lstsq(A, np.array(gy), rcond=None)
        # TS model: choose tile sizes on the grid
        best = None
        tiles = [tuple(args.ts_tiles)] if args.ts_tiles else itertools.product((64, 128), (64, 128, 176))
        for bq, bk in tiles:
            xs = [makespan(ctas(b, bq, bk, hq), sms) for b in gb]
            a, c = fit(xs, gy)
            mae = sum(abs(a + c * x - y) for x, y in zip(xs, gy)) / len(gy)
            if best is None or mae < best[0]:
                best = (mae, bq, bk, a, c)
        _, bq, bk, aT, cT = best
        rows = []
        for p in pair:
            cfg = next(c for c in CONFIGS if c[0] == p["label"])
            bA, bB = batch_for("A", cfg), batch_for("B", cfg)
            predW = bW * (work_units(bA) - work_units(bB)) / 1e6
            predT = cT * (makespan(ctas(bA, bq, bk, hq), sms) - makespan(ctas(bB, bq, bk, hq), sms))
            row = {"label": p["label"], "measured_ms": p["delta"], "median_A": p["median_A"], "median_B": p["median_B"],
                   "pred_W_ms": predW, "pred_TS_ms": predT}
            if predW:
                row["ratio_W"] = p["delta"] / predW
            if predT:
                row["ratio_TS"] = p["delta"] / predT
            if model_ps and model_ps.get(p["label"]) is not None:
                row["model_step_delta_ms"] = model_ps[p["label"]]
                if model_ps[p["label"]]:
                    row["closure"] = args.layers * p["delta"] / model_ps[p["label"]]
            rows.append(row)
        non = [r for r in rows if "control" not in r["label"]]
        ctl = [r for r in rows if "control" in r["label"]][0]
        rep = {"W_fit": {"a_ms": aW, "ms_per_M": bW}, "split_fit": {"a_ms": float(coef[0]), "c_X": float(coef[1]), "c_S": float(coef[2]), "c_S_over_c_X": float(coef[2] / coef[1])},
               "TS_fit": {"bq": bq, "bk": bk, "a_ms": aT, "ms_per_tile": cT, "grid_mae_ms": best[0]}, "pair": rows,
               "HK1": all(r["measured_ms"] > 0 for r in non) and abs(ctl["measured_ms"]) <= 0.02 * ctl["median_A"],
               "HK3": bool(coef[2] / coef[1] > 1.2),
               "HK4_mae_W": statistics.mean(abs(r["measured_ms"] - r["pred_W_ms"]) for r in non),
               "HK4_mae_TS": statistics.mean(abs(r["measured_ms"] - r["pred_TS_ms"]) for r in non),
               "HK5_W_over": sum(r.get("ratio_W", 9) < 0.9 for r in non), "HK5_TS_within": sum(0.75 <= r.get("ratio_TS", 0) <= 1.25 for r in non)}
        if all("closure" in r for r in non):
            rep["HK2_closure"] = [r["closure"] for r in non]
        report["kernels"][k] = rep
        print(f"## {k} ({d['shape']}, {d['gpu']}): W {aW*1000:.1f}us + {bW:.4f} ms/M | split c_S/c_X {coef[2]/coef[1]:.2f} | "
              f"TS bq {bq} bk {bk} grid MAE {best[0]*1000:.1f} us")
        for r in rows:
            print(f"   {r['label']:34s} meas {r['measured_ms']*1000:+8.1f} us  W {r['pred_W_ms']*1000:+8.1f}  TS {r['pred_TS_ms']*1000:+8.1f}"
                  + (f"  closure {r['closure']:.2f}" if "closure" in r else ""))
        print(f"   HK1 {rep['HK1']}  HK3 {rep['HK3']}  HK4 MAE W {rep['HK4_mae_W']*1000:.1f} vs TS {rep['HK4_mae_TS']*1000:.1f} us  "
              f"HK5 W-over {rep['HK5_W_over']}/5 TS-within {rep['HK5_TS_within']}/5")
    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="qwen3-4b", choices=list(SHAPES))
    ap.add_argument("--kernels", default="fa2,fa3,flashinfer")
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--grid-calls", type=int, default=50)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--analyze", type=Path)
    ap.add_argument("--model-pairswap", type=Path)
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--ts-tiles", type=int, nargs=2, metavar=("BQ", "BK"),
                    help="fix the TS tile sizes instead of choosing them on the grid (exploratory, not the registered HK4 test)")
    args = ap.parse_args()
    analyze(args) if args.analyze else run(args)


if __name__ == "__main__":
    main()
