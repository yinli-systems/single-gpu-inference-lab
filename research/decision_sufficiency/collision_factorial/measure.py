"""Fresh real-GPU factorial assay for decision-sufficient execution state.

This runner measures only the frozen collision manifest.  It never edits a
shared FlashInfer install and never promotes an optimization.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [
    str(HERE),
    str(ROOT / "research" / "section6"),
    str(ROOT / "benchmarks" / "results" / "plan-order-mechanism"),
]
from manifest import build, cases_for, digest  # noqa: E402
from metadata_adapter import PlanSnapshot  # noqa: E402

OFFICIAL_PREFILL_SHA256 = (
    "e66bce2652c0a3c5b54f88510aff2b4b48ff6a52c4ee23fa97f91fa06a935b87"
)
ACTIONS = ("identity", "identity_repeat", "heavy_first")
COMPARISONS = ("heavy_first", "identity_repeat")
EXECUTION_MODES = ("eager_one", "graph16")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def command(args: list[str]) -> dict:
    p = subprocess.run(args, capture_output=True, text=True, timeout=20)
    return {"rc": p.returncode, "out": p.stdout, "err": p.stderr}


def tensor_hash(tensor) -> str:
    import torch

    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def selected_fp32_reference(q, k, v, out, lse, query_lengths, total_lengths, dtype_name):
    """Independent selected-row causal attention check."""
    import torch

    absmax = 0.0
    squared = 0.0
    elements = 0
    lse_max = 0.0
    vectors = 0
    q_offset = 0
    k_offset = 0
    scale = math.sqrt(q.shape[-1])

    for nq, total_kv in zip(query_lengths, total_lengths):
        ids = sorted({0, nq // 4, nq // 2, (3 * nq) // 4, nq - 1})
        ids = [i for i in ids if 0 <= i < nq]
        ix = torch.tensor(ids, device=q.device)
        kr = k[k_offset : k_offset + total_kv]
        vr = v[k_offset : k_offset + total_kv]

        qq = q[q_offset + ix].float().transpose(0, 1)
        kk = kr.float().repeat_interleave(4, dim=1).transpose(0, 1)
        vv = vr.float().repeat_interleave(4, dim=1).transpose(0, 1)
        scores = (qq @ kk.transpose(-1, -2)) / scale
        visible_end = ix + total_kv - nq
        mask = (
            torch.arange(total_kv, device=q.device)[None, :]
            > visible_end[:, None]
        )
        scores.masked_fill_(mask[None], float("-inf"))
        expected = (scores.softmax(-1) @ vv).transpose(0, 1)
        expected_lse = scores.logsumexp(-1).transpose(0, 1) / math.log(2)

        actual = out[q_offset + ix].float()
        diff = actual - expected
        torch.testing.assert_close(
            actual,
            expected,
            atol=0.005 if dtype_name == "float16" else 0.02,
            rtol=0.02 if dtype_name == "float16" else 0.04,
        )
        torch.testing.assert_close(
            lse[q_offset + ix],
            expected_lse,
            atol=0.01,
            rtol=0.01,
        )

        absmax = max(absmax, float(diff.abs().max()))
        squared += float(diff.square().sum())
        elements += diff.numel()
        lse_max = max(
            lse_max,
            float((lse[q_offset + ix] - expected_lse).abs().max()),
        )
        vectors += len(ids) * 32
        q_offset += nq
        k_offset += total_kv

    return {
        "max_abs": absmax,
        "rmse": math.sqrt(squared / elements),
        "lse_max_abs": lse_max,
        "vectors": vectors,
        "elements": elements,
    }


def validate_resource_source(resource_mode: str, flashinfer) -> dict:
    pkg = Path(flashinfer.__file__).resolve().parent
    header = pkg / "data" / "include" / "flashinfer" / "attention" / "prefill.cuh"
    current = sha256(header)

    if resource_mode == "pristine":
        if current != OFFICIAL_PREFILL_SHA256:
            raise RuntimeError("pristine FlashInfer header hash mismatch")
        return {
            "mode": "pristine",
            "header": str(header),
            "header_sha256": current,
            "official_sha256": OFFICIAL_PREFILL_SHA256,
        }

    if resource_mode != "cap":
        raise ValueError("unsupported resource mode")

    binding_path = pkg.parent / "RESOURCE_BINDING.json"
    if not binding_path.exists():
        raise RuntimeError("cap overlay binding missing")
    binding = json.loads(binding_path.read_text())
    required = {
        "mode": "cap",
        "base_version": "0.7.0",
        "base_sha256": OFFICIAL_PREFILL_SHA256,
        "device_source_unchanged": True,
        "device_specialization_changes": False,
        "process_immutable": True,
        "ordered_descriptors_unchanged": True,
        "production_promoted": False,
    }
    for key, expected in required.items():
        if binding.get(key) != expected:
            raise RuntimeError("cap binding mismatch for " + key)
    if current != binding["modified_sha256"]:
        raise RuntimeError("cap overlay header hash mismatch")
    return {
        "mode": "cap",
        "header": str(header),
        "header_sha256": current,
        "binding_path": str(binding_path),
        "binding": binding,
    }


def source_receipt() -> dict:
    files = [
        HERE / "manifest.py",
        HERE / "measure.py",
        HERE / "PROTOCOL.md",
        ROOT / "research" / "section6" / "metadata_adapter.py",
        ROOT
        / "benchmarks"
        / "results"
        / "plan-order-mechanism"
        / "plan_contract.py",
        ROOT / "src" / "l20_stack" / "decision_sufficiency.py",
    ]
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in files
    }


def run(args) -> None:
    import torch
    import flashinfer

    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA GPU required")
    if flashinfer.__version__ != "0.7.0":
        raise RuntimeError("pinned FlashInfer 0.7.0 required")
    if args.out.exists():
        raise FileExistsError("preserve existing evidence")
    if args.stage not in ("canary", "formal"):
        raise ValueError("invalid stage")
    if args.stage == "canary" and args.rep != 0:
        raise ValueError("canary uses rep 0 only")
    if args.stage == "formal" and not 0 <= args.rep < 4:
        raise ValueError("formal rep must be in [0,3]")

    manifest = build()
    cases = cases_for(args.stage)
    blocks = 2 if args.stage == "canary" else manifest["blocks"]
    resource = validate_resource_source(args.resource_mode, flashinfer)

    args.out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    affinity_before = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(affinity_before[: min(2, len(affinity_before))]))

    environment = {
        "stage": args.stage,
        "rep": args.rep,
        "resource_mode": args.resource_mode,
        "case_hash": manifest["case_hash"],
        "cases": cases,
        "blocks": blocks,
        "actions": list(ACTIONS),
        "comparisons": list(COMPARISONS),
        "execution_modes": list(EXECUTION_MODES),
        "flashinfer": flashinfer.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "resource": resource,
        "source": source_receipt(),
        "hardware": command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,pci.bus_id,power.limit",
                "--format=csv",
            ]
        ),
        "processes_before": command(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv",
            ]
        ),
        "affinity_before": affinity_before,
        "affinity": sorted(os.sched_getaffinity(0)),
        "job": os.getenv("SLURM_JOB_ID"),
        "partition": os.getenv("SLURM_JOB_PARTITION"),
        "full_model": False,
        "serving_promotion": False,
        "default_promotion": False,
        "clock_lock": False,
        "whole_node_exclusive": False,
    }
    if environment["hardware"]["rc"] != 0:
        raise RuntimeError("hardware binding unavailable")
    save(args.out / "environment.json", environment)

    rows = []
    qualifications = []
    plans = []
    diagnostics = []
    started = time.monotonic()

    try:
        for case, dtype_name, split in itertools.product(
            cases,
            manifest["dtypes"],
            manifest["split_modes"],
        ):
            dtype = getattr(torch, dtype_name)
            query_lengths = case["q"]
            total_lengths = [
                q + cached for q, cached in zip(case["q"], case["cached"])
            ]
            cell_key = {
                "case": case["id"],
                "pair_id": case["pair_id"],
                "arm": case["arm"],
                "dtype": dtype_name,
                "split": split,
            }
            cell_digest = digest(cell_key)
            torch.manual_seed(int(cell_digest[:8], 16))

            q = torch.randn(
                (sum(query_lengths), 32, 128),
                device="cuda",
                dtype=dtype,
            )
            k = torch.randn(
                (sum(total_lengths), 8, 128),
                device="cuda",
                dtype=dtype,
            )
            v = torch.randn_like(k)
            query_indptr = torch.tensor(
                [0] + list(itertools.accumulate(query_lengths)),
                device="cuda",
                dtype=torch.int32,
            )
            kv_indptr = torch.tensor(
                [0] + list(itertools.accumulate(total_lengths)),
                device="cuda",
                dtype=torch.int32,
            )
            workspace = torch.empty(
                512 * 1024**2,
                device="cuda",
                dtype=torch.uint8,
            )
            wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace,
                backend="fa2",
            )

            plan_start = time.perf_counter()
            wrapper.plan(
                query_indptr,
                kv_indptr,
                32,
                8,
                128,
                causal=True,
                q_data_type=dtype,
                kv_data_type=dtype,
                disable_split_kv=(split == "unsplit"),
            )
            torch.cuda.synchronize()
            plan_us = (time.perf_counter() - plan_start) * 1e6

            snapshot = PlanSnapshot(
                wrapper,
                query_lengths,
                total_lengths,
            )
            out = torch.empty_like(q)
            lse = torch.empty(
                (sum(query_lengths), 32),
                device="cuda",
                dtype=torch.float32,
            )

            def call():
                return wrapper.run(
                    q,
                    k,
                    v,
                    out=out,
                    lse=lse,
                    return_lse=True,
                )

            snapshot.apply("identity")
            for _ in range(6):
                call()
            torch.cuda.synchronize()
            reference_out = out.clone()
            reference_lse = lse.clone()
            fp32 = selected_fp32_reference(
                q,
                k,
                v,
                out,
                lse,
                query_lengths,
                total_lengths,
                dtype_name,
            )

            graph1 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph1):
                call()
            graph16 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph16):
                for _ in range(16):
                    call()

            pointers = [
                tensor.data_ptr()
                for tensor in (
                    q,
                    k,
                    v,
                    out,
                    lse,
                    workspace,
                    wrapper._int_workspace_buffer,
                )
            ]
            action_receipts = {}
            for action in ACTIONS:
                signature = snapshot.apply(action)
                changed = signature != snapshot.report["descriptor_hash"]

                out.fill_(float("nan"))
                lse.fill_(float("nan"))
                call()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    out, reference_out, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    lse, reference_lse, atol=0, rtol=0
                )

                out.fill_(float("nan"))
                lse.fill_(float("nan"))
                graph1.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    out, reference_out, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    lse, reference_lse, atol=0, rtol=0
                )

                out.fill_(float("nan"))
                lse.fill_(float("nan"))
                graph16.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    out, reference_out, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    lse, reference_lse, atol=0, rtol=0
                )

                action_receipts[action] = {
                    "descriptor_hash": signature,
                    "changed_from_identity": changed,
                    "exact_output": True,
                    "exact_lse": True,
                }

            if pointers != [
                tensor.data_ptr()
                for tensor in (
                    q,
                    k,
                    v,
                    out,
                    lse,
                    workspace,
                    wrapper._int_workspace_buffer,
                )
            ]:
                raise RuntimeError("tensor/workspace ownership changed")

            qualifications.append(
                {
                    **cell_key,
                    "input_hashes": {
                        "q": tensor_hash(q),
                        "k": tensor_hash(k),
                        "v": tensor_hash(v),
                    },
                    "output_hash": tensor_hash(reference_out),
                    "lse_hash": tensor_hash(reference_lse),
                    "FP32": fp32,
                    "actions": action_receipts,
                    "eager_graph_exact": True,
                }
            )
            plans.append(
                {
                    **cell_key,
                    "plan_us": plan_us,
                    "plan": snapshot.report,
                    "resource_mode": args.resource_mode,
                }
            )

            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            event_start.record()
            event_end.record()
            event_end.synchronize()
            event_start.elapsed_time(event_end)

            rng = random.Random(int(cell_digest[:8], 16) + 1009 * args.rep)
            for block in range(blocks):
                combos = list(itertools.product(EXECUTION_MODES, COMPARISONS))
                rng.shuffle(combos)
                for execution_mode, comparison in combos:
                    sequence = (
                        ("identity", comparison, comparison, "identity")
                        if block % 2 == 0
                        else (
                            comparison,
                            "identity",
                            "identity",
                            comparison,
                        )
                    )
                    for position, arm in enumerate(sequence):
                        setup_start = time.perf_counter()
                        descriptor_hash = snapshot.apply(arm)
                        setup_us = (time.perf_counter() - setup_start) * 1e6

                        for _ in range(4):
                            graph1.replay()
                        torch.cuda.synchronize()

                        wall_start = time.perf_counter()
                        event_start.record()
                        if execution_mode == "eager_one":
                            call()
                            calls = 1
                        else:
                            graph16.replay()
                            calls = 16
                        event_end.record()
                        event_end.synchronize()
                        wall_us = (time.perf_counter() - wall_start) * 1e6 / calls
                        device_us = (
                            event_start.elapsed_time(event_end) * 1000 / calls
                        )
                        if not (
                            math.isfinite(device_us)
                            and device_us > 0
                            and math.isfinite(wall_us)
                            and wall_us > 0
                        ):
                            raise RuntimeError("invalid duration")

                        rows.append(
                            {
                                **cell_key,
                                "resource_mode": args.resource_mode,
                                "rep": args.rep,
                                "block": block,
                                "execution_mode": execution_mode,
                                "comparison": comparison,
                                "position": position,
                                "arm": arm,
                                "device_us": device_us,
                                "wall_us": wall_us,
                                "setup_us": setup_us,
                                "calls": calls,
                                "descriptor_hash": descriptor_hash,
                            }
                        )

                torch.testing.assert_close(
                    out, reference_out, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    lse, reference_lse, atol=0, rtol=0
                )
                if pointers != [
                    tensor.data_ptr()
                    for tensor in (
                        q,
                        k,
                        v,
                        out,
                        lse,
                        workspace,
                        wrapper._int_workspace_buffer,
                    )
                ]:
                    raise RuntimeError("buffer changed during timing")

            if (
                args.rep == 0
                and case["pair_id"]
                == ("canary" if args.stage == "canary" else "formal-01")
                and case["arm"] == "a"
                and dtype_name == "float16"
                and split == "unsplit"
            ):
                try:
                    snapshot.apply("identity")
                    graph1.replay()
                    torch.cuda.synchronize()
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as profiler:
                        graph1.replay()
                        torch.cuda.synchronize()
                    trace_path = args.out / "launch-diagnostic.json"
                    profiler.export_chrome_trace(str(trace_path))
                    trace = json.loads(trace_path.read_text())
                    kernels = [
                        {
                            "name": event.get("name"),
                            "dur": event.get("dur"),
                            "args": event.get("args", {}),
                        }
                        for event in trace.get("traceEvents", [])
                        if event.get("cat") == "kernel"
                    ]
                    diagnostics.append(
                        {
                            **cell_key,
                            "resource_mode": args.resource_mode,
                            "kernel_count": len(kernels),
                            "kernels": kernels,
                            "not_performance_evidence": True,
                        }
                    )
                except Exception as exc:
                    diagnostics.append(
                        {
                            **cell_key,
                            "resource_mode": args.resource_mode,
                            "error": str(exc),
                            "not_performance_evidence": True,
                        }
                    )

            save(
                args.out / "progress.json",
                {
                    "last_cell": cell_key,
                    "rows": len(rows),
                    "qualifications": len(qualifications),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )

            snapshot.apply("identity")
            del (
                graph1,
                graph16,
                snapshot,
                wrapper,
                workspace,
                q,
                k,
                v,
                query_indptr,
                kv_indptr,
                out,
                lse,
                reference_out,
                reference_lse,
            )
            torch.cuda.empty_cache()

        expected_rows = (
            len(cases)
            * len(manifest["dtypes"])
            * len(manifest["split_modes"])
            * len(EXECUTION_MODES)
            * len(COMPARISONS)
            * blocks
            * 4
        )
        expected_qualifications = (
            len(cases)
            * len(manifest["dtypes"])
            * len(manifest["split_modes"])
        )
        if len(rows) != expected_rows:
            raise RuntimeError("incomplete timing matrix")
        if len(qualifications) != expected_qualifications:
            raise RuntimeError("incomplete qualification matrix")

        save(args.out / "measurements.json", rows)
        save(args.out / "qualification.json", qualifications)
        save(args.out / "plans.json", plans)
        save(args.out / "diagnostics.json", diagnostics)
        file_hashes = {
            path.name: sha256(path)
            for path in args.out.glob("*.json")
            if path.name != "complete.json"
        }
        save(
            args.out / "complete.json",
            {
                "complete": True,
                "stage": args.stage,
                "rep": args.rep,
                "resource_mode": args.resource_mode,
                "rows": len(rows),
                "expected_rows": expected_rows,
                "qualifications": len(qualifications),
                "expected_qualifications": expected_qualifications,
                "files": file_hashes,
                "performance_promotion": False,
                "serving_promotion": False,
            },
        )
        print(
            "COMPLETE",
            args.stage,
            args.resource_mode,
            args.rep,
            len(rows),
            flush=True,
        )
    except BaseException as exc:
        save(args.out / "partial_measurements.json", rows)
        save(args.out / "qualification.json", qualifications)
        save(args.out / "plans.json", plans)
        save(args.out / "diagnostics.json", diagnostics)
        save(
            args.out / "failure.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["canary", "formal"], required=True)
    parser.add_argument("--resource-mode", choices=["pristine", "cap"], required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
