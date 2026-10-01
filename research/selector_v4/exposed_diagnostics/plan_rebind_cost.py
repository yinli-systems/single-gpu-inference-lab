"""Exposed full-call cost of explicit eager same-geometry plan reuse.

Eight balanced ABBA/BAAB blocks preserve every individual native plan+run and
native plan+rebind+managed-run call. An accepted prepared confidence receipt and
actual managed winner are reused under an immutable source envelope. A native
winner bypasses rebind on the direct native path. Per-plan costs are included;
these short diagnostic samples do not satisfy release timing/control gates.
"""

import argparse
import hashlib
import itertools
import json
import os
import time
from pathlib import Path


def save(path, value):
    with path.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def run(out, expected_source_commit):
    import torch
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper, MeasurementPolicy, autotune_v2
    from flashinfer._build_meta import __git_commit__, __version__
    from flashinfer.autotuner import AutoTuner
    from flashinfer.prefill import make_prefill_resource_runner

    assert (__git_commit__, __version__) == (expected_source_commit, "0.7.1")
    if out.exists():
        raise FileExistsError("Preserve every started setup diagnosis")
    out.mkdir(parents=True)
    affinity = sorted(os.sched_getaffinity(0))
    assert len(affinity) <= 12
    os.sched_setaffinity(0, set(affinity[:2]))
    torch.set_num_threads(1)
    torch.manual_seed(42)
    qs = [3, 39, 107, 175, 243, 311, 411]
    ks = [q + c for q, c in zip(qs, [22528, 16512, 11840, 8256, 2624, 672, 80])]
    ip = lambda x: torch.tensor([0, *itertools.accumulate(x)], dtype=torch.int32, device="cuda")
    qi, ki = ip(qs), ip(ks)
    q = torch.randn(sum(qs), 32, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(sum(ks), 8, 128, dtype=q.dtype, device=q.device)
    v = torch.randn_like(k)
    inputs = [q, k, v]
    w = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 << 20, dtype=torch.uint8, device="cuda"), backend="fa2"
    )
    kwargs = {"causal": True, "q_data_type": q.dtype, "disable_split_kv": True}
    plan = lambda: w.plan(qi, ki, 32, 8, 128, **kwargs)
    construct = lambda: make_prefill_resource_runner(w, inputs, qo_lengths=qs, kv_lengths=ks)
    plan()
    reference = w.run(*inputs).clone()
    runner = construct()
    runner.tuning_config.profiling_repeat = 256
    cert = runner.calibrate(inputs)
    save(out / "calibration.json", cert)
    published = runner.save_receipt()
    policy = MeasurementPolicy(execution_mode="eager")
    store = out / "managed-cache"
    with autotune_v2(mode="tune", measurement_policy=policy, cache_root=store):
        initial = runner.run(inputs)
        selected, tactic = AutoTuner.get().choose_one(
            "experimental_prefill_resource", [runner], runner.tuning_config, inputs
        )
    assert selected is runner and tactic in (-1, 0, 1)
    assert not AutoTuner.get().stats.failed_tactics.get(
        "experimental_prefill_resource::ResourceCapRunner"
    )
    assert (
        AutoTuner.get().stats.tuned_op_successful_configs.get("experimental_prefill_resource", 0)
        > 0
    )
    torch.testing.assert_close(initial, reference, rtol=0, atol=0)
    props = torch.cuda.get_device_properties(0)
    save(
        out / "environment.json",
        {
            "source_commit": __git_commit__,
            "package_version": __version__,
            "gpu_uuid": str(props.uuid),
            "gpu_name": props.name,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "confidence_identity": runner.identity,
            "confidence_checksum": cert["checksum"],
            "confidence_accepted": runner._receipt_valid,
            "confidence_published": published,
            "initial_managed_tactic": tactic,
            "fixed_managed_profiling_repeat": 256,
            "initial_confidence_checksum": runner._receipt["checksum"],
            "native_winner_uses_direct_native_path": True,
            "source_compiler_files_immutable_for_prepared_lifetime": True,
            "geometry_metadata_validation_in_scored_calls": True,
            "previously_exposed_geometry": True,
            "fresh_cases_consumed": 0,
            "qualification_authority": False,
        },
    )
    rows = []
    with autotune_v2(mode="replay", measurement_policy=policy, cache_root=store):
        for block in range(8):
            order = (
                ["native", "rebind", "rebind", "native"]
                if block % 2 == 0
                else ["rebind", "native", "native", "rebind"]
            )
            for position, arm in enumerate(order):
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                plan()
                planned = time.perf_counter_ns()
                bound = (
                    runner.rebind_same_geometry(inputs)
                    if arm == "rebind" and tactic == 1
                    else False
                )
                rebound = time.perf_counter_ns()
                if bound:
                    result = runner.run(inputs)
                else:
                    result = w.run(*inputs)
                torch.cuda.synchronize()
                finished = time.perf_counter_ns()
                rows.append(
                    {
                        "block": block,
                        "position": position,
                        "arm": arm,
                        "wall_ms": (finished - start) / 1e6,
                        "plan_call_wall_ms": (planned - start) / 1e6,
                        "rebind_call_wall_ms": (rebound - planned) / 1e6,
                        "run_and_sync_ms": (finished - rebound) / 1e6,
                        "resource_rebound": bound,
                        "effective_tactic": tactic if bound else -1,
                        "exact": torch.equal(result, reference),
                        "prepared_confidence_checksum": runner._receipt["checksum"],
                        "resource_eligible_after_call": runner._eligible,
                    }
                )
                save(out / "measurements.json", rows)
                assert runner._receipt["checksum"] == cert["checksum"]
                if bound:
                    assert runner._receipt_valid and runner._current(inputs)
                if not rows[-1]["exact"]:
                    torch.save(
                        {"actual": result, "native_reference": reference},
                        out / f"mismatch-{block}-{position}.pt",
                    )
                    raise RuntimeError("Setup diagnosis output mismatch; raw tensors retained")
    save(
        out / "complete.json",
        {
            "complete": True,
            "observations": len(rows),
            "all32_observations_retained": True,
            "strict_window_gates_satisfied": False,
            "full_serving_qualified": False,
            "fresh_cases_consumed": 0,
            "default_promotion": False,
            "files": {
                str(f.relative_to(out)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in out.rglob("*")
                if f.is_file()
            },
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected-source-commit", required=True)
    args = p.parse_args()
    run(args.out, args.expected_source_commit)
