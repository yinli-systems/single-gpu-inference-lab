"""Fixed exposed counterfactual controls, independent of qualification timings."""

import argparse
import copy
import functools
import hashlib
import itertools
import json
import os
import subprocess
import traceback
from pathlib import Path

from attribute_launch_control import build_modules, normalize_symbols
from audit_main_sass import parse
from serving_kernel_profile import save


def run(out, expected_commit):
    import numpy as np
    import torch
    from flashinfer._build_meta import __git_commit__, __version__
    from flashinfer.jit.attention.modules import _gen_batch_prefill_primary_module
    from flashinfer.prefill import (
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
        make_prefill_resource_runner,
    )
    from flashinfer.testing import bench_gpu_time

    assert (__git_commit__, __version__) == (expected_commit, "0.7.1")
    out.mkdir()
    torch.set_num_threads(1)
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    torch.manual_seed(20261001)
    props = torch.cuda.get_device_properties(0)
    save(
        out / "environment.json",
        {
            "source_commit": expected_commit,
            "gpu_name": props.name,
            "gpu_uuid": str(props.uuid),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "qualification_authority": False,
            "fresh_cases_consumed": 0,
            "full_http_qualified": False,
        },
    )
    qs = [3, 39, 107, 175, 243, 311, 411]
    cached = [22528, 16512, 11840, 8256, 2624, 672, 80]
    module_args = (
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.int32,
        128,
        128,
        0,
        False,
        False,
        False,
    )
    modules = None
    records = {}
    for paged in (False, True):
        layout = "paged1_cached_prefix" if paged else "ragged_projection_suffix"
        cell = out / layout
        cell.mkdir()
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
        call = lambda wrapper, q=q, k=k, v=v, kwargs=kwargs, inputs=inputs, paged=paged: (
            functools.partial(wrapper.run, q, (k, v), **kwargs)
            if paged
            else functools.partial(wrapper.run, *inputs, **kwargs)
        )
        functions = {"native": call(owner)}
        reference = tuple(t.clone() for t in functions["native"]())
        runner = make_prefill_resource_runner(
            owner, inputs, qo_lengths=qs, kv_lengths=lengths, run_kwargs=kwargs
        )
        assert runner._eligible and runner.receipt is None
        if modules is None:
            module_evidence = out / "module-evidence"
            module_evidence.mkdir()
            modules = build_modules(module_args, runner.source_id, module_evidence)
        for label, (module, proof) in modules.items():
            proxy = copy.copy(owner)
            proxy._cached_module = module
            functions[label] = call(proxy)
        original_plan, original_module = owner._plan_info, owner._cached_module
        exact = lambda result, reference=reference: all(
            torch.equal(a, b) for a, b in zip(result, reference, strict=True)
        )
        assert all(exact(fn()) for fn in functions.values())
        rows = []
        labels = list(functions)
        for cold in (False, True):
            for block in range(8):
                first = labels[block % 4 :] + labels[: block % 4]
                for position, label in enumerate(first + list(reversed(first))):
                    values = bench_gpu_time(
                        fn=functions[label],
                        enable_cupti=False,
                        dry_run_iters=8,
                        repeat_iters=64,
                        use_cuda_graph=False,
                        cold_l2_cache=cold,
                    )
                    row = {
                        "cold_l2": cold,
                        "block": block,
                        "position": position,
                        "arm": label,
                        "gpu_event_ms": [float(v) for v in values],
                        "exact": exact(functions[label]()),
                    }
                    rows.append(row)
                    save(cell / "event-observations.json", rows)
                    assert (
                        row["exact"]
                        and len(values) == 64
                        and all(np.isfinite(v) and v > 0 for v in values)
                    )
                assert exact(functions["native"]())
        profiles = {}
        for label, fn in functions.items():
            for _ in range(8):
                fn()
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                for _ in range(16):
                    fn()
                torch.cuda.synchronize()
            path = cell / (label + "-profile.json")
            profile.export_chrome_trace(str(path))
            trace = json.loads(path.read_text())
            kernels = [
                e
                for e in trace["traceEvents"]
                if e.get("cat") == "kernel" and "BatchPrefillWith" in e.get("name", "")
            ]
            expected = 49152 if label == "native" else modules[label][1]["launch_bytes"]
            assert len(kernels) == 16 and all(
                e["args"]["shared memory"] == expected for e in kernels
            )
            profiles[label] = {
                "kernels": kernels,
                "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            save(cell / (label + "-launches.json"), profiles[label])
        assert owner._plan_info is original_plan and owner._cached_module is original_module
        records[layout] = {
            "exact": True,
            "profiles": profiles,
            "source_plan": list(original_plan),
            "native_owner_unchanged": True,
            "raw_observations": 8192,
            "event_backend": "CUDA events include host gaps; CUPTI kernel traces separate",
        }
        save(cell / "complete.json", records[layout])
    # Disassemble only after all timings. Full instruction hashes must match.
    tool = Path(os.environ["CUDA_HOME"]) / "bin/cuobjdump"
    native = _gen_batch_prefill_primary_module("fa2", *module_args)
    raw = subprocess.check_output(
        [str(tool), "--dump-sass", str(native.get_library_path())], text=True
    )
    (out / "native.sass").write_text(raw)
    expected = parse(raw)
    assert expected
    sass = {}
    for label, (_, proof) in modules.items():
        raw = subprocess.check_output([str(tool), "--dump-sass", proof["binary"]], text=True)
        (out / (label + ".sass")).write_text(raw)
        actual = parse(normalize_symbols(raw, proof["attribute_bytes"], proof["launch_bytes"]))
        mismatch = sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        sass[label] = {
            "mismatches": mismatch,
            "missing": missing,
            "extra": extra,
            "paired_kernels": len(expected),
        }
        save(out / "sass-comparison.json", sass)
        assert not (mismatch or missing or extra)
    save(
        out / "complete.json",
        {
            "pass_": True,
            "layouts": records,
            "sass": sass,
            "source_unchanged": True,
            "actual_hardware_residency_measured": False,
            "uncertified_diagnostic_only": True,
            "qualification_authority": False,
            "fresh_cases_consumed": 0,
            "full_http_qualified": False,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected-commit", required=True)
    a = p.parse_args()
    try:
        run(a.out, a.expected_commit)
    except Exception:
        a.out.mkdir(parents=True, exist_ok=True)
        save(
            a.out / "failure.json",
            {"traceback": traceback.format_exc(), "qualification_authority": False},
        )
        raise
