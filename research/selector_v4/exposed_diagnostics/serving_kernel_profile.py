"""Uncertified resource mechanism on exposed serving layouts, not selection.

Real ragged suffix projection views and page1 noncausal cached prefixes are
profiled separately from all qualification/HTTP measurements. Keep every raw
CUDA-event observation and CUPTI-backed Torch trace. No deployment certificate
or fresh shape is used. Missing profiler evidence is a retained diagnostic HOLD.
"""

import argparse
import functools
import hashlib
import itertools
import json
import os
import warnings
from pathlib import Path


def save(path, value):
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(out, expected_commit):
    import numpy as np
    import torch
    from flashinfer._build_meta import __git_commit__, __version__
    from flashinfer.prefill import (
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
        make_prefill_resource_runner,
    )
    from flashinfer.testing import bench_gpu_time

    assert (__git_commit__, __version__) == (expected_commit, "0.7.1")
    out.mkdir()
    torch.set_num_threads(1)
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(affinity[:2]))
    torch.manual_seed(20261001)
    qs = [3, 39, 107, 175, 243, 311, 411]
    cached = [22528, 16512, 11840, 8256, 2624, 672, 80]
    props = torch.cuda.get_device_properties(0)
    save(
        out / "environment.json",
        {
            "source_commit": expected_commit,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "gpu_uuid": str(props.uuid),
            "gpu_name": props.name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "explicit_uncertified_diagnostic": True,
            "qualification_authority": False,
            "fresh_cases_consumed": 0,
            "full_http_qualified": False,
        },
    )
    records = {}
    for layout in ("ragged_projection_suffix", "paged1_cached_prefix"):
        cell = out / layout
        cell.mkdir()
        paged = layout.startswith("paged")
        lengths = cached if paged else qs
        ip = lambda values: torch.tensor(
            [0, *itertools.accumulate(values)], dtype=torch.int32, device="cuda"
        )
        q = torch.randn(sum(qs), 32, 128, dtype=torch.bfloat16, device="cuda")
        workspace = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
        if paged:
            k = torch.randn(sum(lengths), 1, 8, 128, dtype=q.dtype, device="cuda")
            v = torch.randn_like(k)
            owner = BatchPrefillWithPagedKVCacheWrapper(workspace, backend="fa2")
            owner.plan(
                ip(qs),
                ip(lengths),
                torch.arange(sum(lengths), dtype=torch.int32, device="cuda"),
                torch.ones(len(qs), dtype=torch.int32, device="cuda"),
                32,
                8,
                128,
                1,
                causal=False,
                q_data_type=q.dtype,
                disable_split_kv=True,
            )
        else:
            projection = torch.randn(sum(lengths), 48, 128, dtype=q.dtype, device="cuda")
            k, v = projection[:, 32:40], projection[:, 40:48]
            owner = BatchPrefillWithRaggedKVCacheWrapper(workspace, backend="fa2")
            owner.plan(
                ip(qs),
                ip(lengths),
                32,
                8,
                128,
                causal=True,
                q_data_type=q.dtype,
                disable_split_kv=True,
            )
        inputs = [q, k, v]
        kwargs = {
            "return_lse": True,
            "out": torch.empty_like(q),
            "lse": torch.empty(sum(qs), 32, dtype=torch.float32, device="cuda"),
        }
        native = (
            functools.partial(owner.run, q, (k, v), **kwargs)
            if paged
            else functools.partial(owner.run, *inputs, **kwargs)
        )
        reference = tuple(t.clone() for t in native())
        runner = make_prefill_resource_runner(
            owner, inputs, qo_lengths=qs, kv_lengths=lengths, run_kwargs=kwargs
        )
        assert runner.receipt is None, "This diagnostic must not borrow a deployment certificate"
        assert runner._eligible and runner._prepare(inputs)
        resource = functools.partial(runner._resource, inputs, **kwargs)
        functions = {"native": native, "resource": resource}
        original_plan, original_module = owner._plan_info, owner._cached_module

        def exact(values, reference=reference):
            return all(torch.equal(a, b) for a, b in zip(values, reference, strict=True))

        assert exact(resource()) and exact(native())
        rows = []
        for cold in (False, True):
            for block in range(8):
                order = (
                    ["native", "resource", "resource", "native"]
                    if not block % 2
                    else ["resource", "native", "native", "resource"]
                )
                for position, arm in enumerate(order):
                    with warnings.catch_warnings(record=True) as caught:
                        values = bench_gpu_time(
                            fn=functions[arm],
                            enable_cupti=True,
                            dry_run_iters=8,
                            repeat_iters=64,
                            use_cuda_graph=False,
                            cold_l2_cache=cold,
                        )
                    row = {
                        "cold_l2": cold,
                        "block": block,
                        "position": position,
                        "arm": arm,
                        "gpu_event_ms": [float(x) for x in values],
                        "backend_warnings": [str(x.message) for x in caught],
                        "exact": exact(functions[arm]()),
                    }
                    rows.append(row)
                    save(cell / "event-observations.json", rows)
                    assert row["exact"] and len(values) == 64
                    assert all(np.isfinite(x) and x > 0 for x in values)
                assert exact(native())
        profiles = {}
        for arm, function in functions.items():
            for _ in range(8):
                function()
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as profile:
                for _ in range(16):
                    function()
                torch.cuda.synchronize()
            path = cell / (arm + "-profile.json")
            profile.export_chrome_trace(str(path))
            trace = json.loads(path.read_text())
            kernels = [
                e
                for e in trace["traceEvents"]
                if e.get("cat") == "kernel" and "BatchPrefillWith" in e.get("name", "")
            ]
            profiles[arm] = {
                "kernels": kernels,
                "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            save(cell / (arm + "-launches.json"), profiles[arm])
            expected_memory = 65536 if arm == "resource" else 49152
            assert len(kernels) == 16, "Missing actual CUPTI-backed kernel launch evidence"
            assert all(e["args"].get("shared memory") == expected_memory for e in kernels)
            assert all(("ResourceKernel" in e["name"]) == (arm == "resource") for e in kernels)
            assert exact(function())
        assert owner._plan_info is original_plan and owner._cached_module is original_module
        assert exact(native())
        result = {
            "layout": layout,
            "q_lengths": qs,
            "kv_lengths": lengths,
            "inputs": [{"shape": list(t.shape), "stride": list(t.stride())} for t in inputs],
            "native_plan": list(owner._plan_info),
            "exact": True,
            "event_observations": len(rows),
            "launches": profiles,
            "profile_timing_excluded_from_qualification": True,
            "raw_resource_is_uncertified_diagnostic_only": True,
            "event_fallback_includes_host_launch_gaps": True,
            "hardware_residency_counters_collected": False,
        }
        save(cell / "complete.json", result)
        records[layout] = result
    save(
        out / "complete.json",
        {
            "complete": True,
            "records": records,
            "qualification_authority": False,
            "default_promotion": False,
            "serving_promotion": False,
            "fresh_cases_consumed": 0,
            "historical_token_divergence_resolved": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("expected_commit")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Preserve every started diagnostic; output already exists")
    try:
        run(args.out, args.expected_commit)
    except BaseException as error:
        if args.out.exists():
            save(
                args.out / "failure.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "qualification_authority": False,
                },
            )
        raise
