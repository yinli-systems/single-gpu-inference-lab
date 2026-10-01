"""Exposed eager managed profiling precision; all fits retained, no promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(out, rep):
    import torch
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper, MeasurementPolicy, autotune_v2
    from flashinfer._build_meta import __git_commit__, __version__
    from flashinfer.autotuner import AutoTuner
    from flashinfer.jit.core import logger
    from flashinfer.prefill import make_prefill_resource_runner

    assert (__git_commit__, __version__) == ("eb4e3e60a6de71f00c0fb558da6f21e4b45d8b55", "0.7.1")
    if out.exists():
        raise FileExistsError("Preserve every started diagnostic")
    out.mkdir(parents=True)
    affinity = sorted(os.sched_getaffinity(0))
    assert len(affinity) <= 12
    os.sched_setaffinity(0, set(affinity[:2]))
    torch.set_num_threads(1)
    torch.manual_seed(42017 + rep)
    qs = [3, 39, 107, 175, 243, 311, 411]
    ks = [q + c for q, c in zip(qs, [22528, 16512, 11840, 8256, 2624, 672, 80], strict=True)]

    def ip(lengths):
        return torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
        )

    q = torch.randn(sum(qs), 32, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(sum(ks), 8, 128, dtype=q.dtype, device=q.device)
    v = torch.randn_like(k)
    inputs = [q, k, v]
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 << 20, dtype=torch.uint8, device="cuda"), backend="fa2"
    )
    wrapper.plan(
        ip(qs), ip(ks), 32, 8, 128, causal=True, q_data_type=q.dtype, disable_split_kv=True
    )
    reference = wrapper.run(*inputs).clone()
    runner = make_prefill_resource_runner(wrapper, inputs, qo_lengths=qs, kv_lengths=ks)
    certificate = runner.calibrate(inputs)
    save(out / "calibration.json", certificate)
    props = torch.cuda.get_device_properties(0)
    save(
        out / "environment.json",
        dict(
            source_commit=__git_commit__,
            package_version=__version__,
            gpu_name=props.name,
            gpu_uuid=str(props.uuid),
            affinity=sorted(os.sched_getaffinity(0)),
            rep=rep,
            script_sha256=digest(Path(__file__)),
            runner_identity=runner.identity,
            certificate_checksum=certificate["checksum"],
            certificate_accepted=runner._receipt_valid,
            previously_exposed_geometry=True,
            fresh_cases_consumed=0,
            qualified_release=False,
        ),
    )
    # Two balanced order blocks, preregistered before these diagnostic fits.
    counts = [10, 256, 256, 10, 256, 10, 10, 256]
    rows = []
    logger.setLevel("DEBUG")
    tuner = AutoTuner.get()
    for fit, count in enumerate(counts):
        store = out / f"fit-{fit:02d}-repeat{count}"
        store.mkdir()
        tuner.clear_cache()
        tuner.reset_statistics()
        runner.tuning_config.profiling_repeat = count
        # Both repeat settings receive the same synthesized-value RNG seed.
        torch.manual_seed(202610010 + rep)
        handler = logging.FileHandler(store / "profiling.log")
        logger.addHandler(handler)
        try:
            with autotune_v2(
                mode="tune",
                measurement_policy=MeasurementPolicy(execution_mode="eager"),
                cache_root=store / "managed-cache",
            ):
                result = runner.run(inputs)
                selected, tactic = tuner.choose_one(
                    "experimental_prefill_resource", [runner], runner.tuning_config, inputs
                )
            assert selected is runner and tactic in (-1, 0, 1)
            assert (
                tuner.stats.tuned_op_successful_configs.get("experimental_prefill_resource", 0) > 0
            )
            assert not tuner.stats.failed_tactics.get(
                "experimental_prefill_resource::ResourceCapRunner"
            )
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
            assert runner.receipt["checksum"] == certificate["checksum"]
            entries = list((store / "managed-cache").glob("v2/*/entries/*.json"))
            assert len(entries) == 1
            entry = json.loads(entries[0].read_text())
            assert (
                entry["tactic"] == tactic
                and runner.identity in entry["key"]
                and certificate["checksum"] in entry["key"]
            )
            row = dict(
                fit=fit,
                profiling_repeat=count,
                chosen_tactic=tactic,
                exact=True,
                certificate_accepted=runner._receipt_valid,
                persisted_entry_sha256=digest(entries[0]),
                original_confidence_geomean=certificate.get("geomean"),
                same_certificate_for_both_precision_settings=True,
                default_promotion=False,
            )
            save(store / "result.json", row)
            rows.append(row)
            save(out / "results.json", rows)
        finally:
            logger.removeHandler(handler)
            handler.close()
    save(
        out / "complete.json",
        dict(
            complete=True,
            rep=rep,
            fits=len(rows),
            files={str(f.relative_to(out)): digest(f) for f in out.rglob("*") if f.is_file()},
            fresh_cases_consumed=0,
            qualified_release=False,
            qualified_http=False,
            historical_token_divergence_resolved=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rep", type=int, required=True)
    args = parser.parse_args()
    run(args.out, args.rep)
