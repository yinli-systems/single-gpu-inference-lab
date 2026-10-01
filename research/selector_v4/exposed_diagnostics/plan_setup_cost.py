"""Exposed setup-cost diagnosis of the public prepared runner, not qualification.

Eight balanced ABBA/BAAB blocks preserve every individual native plan+run and
native plan+new-runner+managed-run call. The same confidence receipt and managed
winner are loaded after each new native plan. Preparation costs are included;
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


def run(out):
    import torch
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper, MeasurementPolicy, autotune_v2
    from flashinfer._build_meta import __git_commit__, __version__
    from flashinfer.autotuner import AutoTuner
    from flashinfer.prefill import make_prefill_resource_runner

    assert (__git_commit__, __version__) == ("bab48695bbe72ba23c49bad69818b2975f3de6f7", "0.7.1")
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
    kwargs = dict(causal=True, q_data_type=q.dtype, disable_split_kv=True)
    plan = lambda: w.plan(qi, ki, 32, 8, 128, **kwargs)
    construct = lambda: make_prefill_resource_runner(w, inputs, qo_lengths=qs, kv_lengths=ks)
    plan()
    reference = w.run(*inputs).clone()
    runner = construct()
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
    torch.testing.assert_close(initial, reference, rtol=0, atol=0)
    props = torch.cuda.get_device_properties(0)
    save(
        out / "environment.json",
        dict(
            source_commit=__git_commit__,
            package_version=__version__,
            gpu_uuid=str(props.uuid),
            gpu_name=props.name,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            confidence_identity=runner.identity,
            confidence_checksum=cert["checksum"],
            confidence_accepted=runner._receipt_valid,
            confidence_published=published,
            initial_managed_tactic=tactic,
            previously_exposed_geometry=True,
            fresh_cases_consumed=0,
            qualification_authority=False,
        ),
    )
    rows = []
    with autotune_v2(mode="replay", measurement_policy=policy, cache_root=store):
        for block in range(8):
            order = (
                ["native", "new_runner", "new_runner", "native"]
                if block % 2 == 0
                else ["new_runner", "native", "native", "new_runner"]
            )
            for position, arm in enumerate(order):
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                plan()
                planned = time.perf_counter_ns()
                current = construct() if arm == "new_runner" else None
                constructed = time.perf_counter_ns()
                if current:
                    result = current.run(inputs)
                else:
                    result = w.run(*inputs)
                torch.cuda.synchronize()
                finished = time.perf_counter_ns()
                loaded = current.receipt if current else None
                rows.append(
                    dict(
                        block=block,
                        position=position,
                        arm=arm,
                        wall_ms=(finished - start) / 1e6,
                        plan_call_wall_ms=(planned - start) / 1e6,
                        constructor_call_wall_ms=(constructed - planned) / 1e6,
                        run_and_sync_ms=(finished - constructed) / 1e6,
                        exact=torch.equal(result, reference),
                        confidence_identity_stable=current.identity == runner.identity
                        if current
                        else None,
                        loaded_confidence_checksum=loaded["checksum"] if loaded else None,
                    )
                )
                save(out / "measurements.json", rows)
                if current:
                    assert current.identity == runner.identity
                    if published:
                        assert loaded["checksum"] == cert["checksum"]
                    else:
                        assert loaded is None and not current._receipt_valid
                if not rows[-1]["exact"]:
                    torch.save(
                        {"actual": result, "native_reference": reference},
                        out / f"mismatch-{block}-{position}.pt",
                    )
                    raise RuntimeError("Setup diagnosis output mismatch; raw tensors retained")
    save(
        out / "complete.json",
        dict(
            complete=True,
            observations=len(rows),
            all32_observations_retained=True,
            strict_window_gates_satisfied=False,
            full_serving_qualified=False,
            fresh_cases_consumed=0,
            default_promotion=False,
            files={
                str(f.relative_to(out)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in out.rglob("*")
                if f.is_file()
            },
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args().out)
