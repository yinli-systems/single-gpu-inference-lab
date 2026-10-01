"""Actual prepared certificates and v2 winners transferred across processes.

One producer and two fresh consumers per physical GPU. Only an already exposed
BF16 ragged geometry is used. Consumers never calibrate or tune. These are
functional lifetime checks, not held-out performance observations.
"""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(out, producer, expected_commit):
    import torch
    from flashinfer import (
        BatchPrefillWithRaggedKVCacheWrapper,
        MeasurementPolicy,
        autotune_v2,
    )
    from flashinfer._build_meta import __git_commit__, __version__
    from flashinfer.autotuner import AutoTuner
    from flashinfer.prefill import make_prefill_resource_runner

    assert (__git_commit__, __version__) == (expected_commit, "0.7.1")
    out.mkdir()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(affinity[:2]))
    qs = [3, 39, 107, 175, 243, 311, 411]
    cached = [22528, 16512, 11840, 8256, 2624, 672, 80]
    lengths = [q + k for q, k in zip(qs, cached, strict=True)]

    def ip(values):
        return torch.tensor([0, *itertools.accumulate(values)], dtype=torch.int32, device="cuda")

    qi, ki = ip(qs), ip(lengths)
    q = torch.randn(sum(qs), 32, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(sum(lengths), 8, 128, dtype=q.dtype, device=q.device)
    v = torch.randn_like(k)
    inputs = [q, k, v]
    output = torch.empty_like(q)
    lse = torch.empty(sum(qs), 32, device="cuda", dtype=torch.float32)
    kwargs = {"out": output, "lse": lse, "return_lse": True}
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 << 20, device="cuda", dtype=torch.uint8), backend="fa2"
    )

    def plan():
        wrapper.plan(qi, ki, 32, 8, 128, causal=True, q_data_type=q.dtype, disable_split_kv=True)

    def native():
        return wrapper.run(*inputs, **kwargs)

    def clone_pair(values):
        return [t.detach().clone() for t in values]

    def exact(actual, reference):
        return all(torch.equal(a, b) for a, b in zip(actual, reference, strict=True))

    plan()
    reference = clone_pair(native())
    props = torch.cuda.get_device_properties(0)
    environment = {
        "source_commit": __git_commit__,
        "gpu_uuid": str(props.uuid),
        "gpu_name": props.name,
        "pid": os.getpid(),
        "source_sha256": sha(Path(__file__)),
        "affinity": sorted(os.sched_getaffinity(0)),
        "inputs_sha256": [
            hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            for t in inputs
        ],
        "producer": str(producer) if producer else None,
        "native_output_lse_sha256": [
            hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            for t in reference
        ],
        "calibrates_and_tunes": producer is None,
        "fresh_cases_consumed": 0,
        "performance_qualification": False,
        "full_http_qualified": False,
    }
    save(out / "environment.json", environment)
    if producer:
        prior_env = json.loads((producer / "environment.json").read_text())
        for key in (
            "source_commit",
            "gpu_uuid",
            "gpu_name",
            "inputs_sha256",
            "source_sha256",
            "native_output_lse_sha256",
        ):
            assert prior_env[key] == environment[key], key
        assert prior_env["pid"] != environment["pid"]
        assert json.loads((producer / "complete.json").read_text())["pass"]

    results = {}
    for mode, count in [("eager_run", 1), ("graph_replay", 1), ("graph_replay", 16)]:
        name = f"{mode}-{count}"
        plan()
        runner = make_prefill_resource_runner(
            wrapper,
            inputs,
            qo_lengths=qs,
            kv_lengths=lengths,
            execution_mode=mode,
            graph_replays=count,
            run_kwargs=kwargs,
        )
        runner.tuning_config.profiling_repeat = 256
        policy = MeasurementPolicy(
            execution_mode="eager" if count == 1 and mode == "eager_run" else "cuda_graph"
        )
        root = producer if producer else out
        certificate_root = root / name / "certificates"
        cache = root / name / "managed-cache"
        (out / name).mkdir()
        if producer is None:
            certificate = runner.calibrate(inputs)
            # Preserve rejected real receipts too; load_receipt independently
            # recomputes acceptance and cannot promote these raw observations.
            certificate_root.mkdir(parents=True)
            save(certificate_root / (runner.identity + ".json"), certificate)
            with autotune_v2(mode="tune", measurement_policy=policy, cache_root=cache):
                initial = clone_pair(runner.run(inputs))
                chosen, tactic = AutoTuner.get().choose_one(
                    "experimental_prefill_resource", [runner], runner.tuning_config, inputs
                )
            assert (
                AutoTuner.get().stats.tuned_op_successful_configs.get(
                    "experimental_prefill_resource", 0
                )
                > 0
            )
            assert not AutoTuner.get().stats.failed_tactics.get(
                "experimental_prefill_resource::ResourceCapRunner"
            )
        else:
            previous = json.loads((producer / name / "result.json").read_text())
            assert runner.identity == previous["identity"]
            assert runner.load_receipt(certificate_root) == previous["certificate_accepted"]
            certificate = runner.receipt
            assert certificate["checksum"] == previous["certificate_checksum"]
            if previous["certificate_accepted"]:
                assert runner._prepare(inputs), (
                    "Fresh consumer must load its own real resource module"
                )
            with autotune_v2(mode="replay", measurement_policy=policy, cache_root=cache):
                tuner = AutoTuner.get()
                config = tuner._apply_measure_policy(runner.tuning_config, policy)
                hit, runner_id, disk_tactic, _ = tuner.search_cache(
                    "experimental_prefill_resource",
                    [runner],
                    tuple(tuner._get_input_sizes(inputs)),
                    config,
                    inputs=inputs,
                )
                assert hit and runner_id == 0 and disk_tactic == previous["managed_tactic"], (
                    "A native -1 cache miss must not imitate a reloaded winner"
                )
                initial = clone_pair(runner.run(inputs))
                chosen, tactic = AutoTuner.get().choose_one(
                    "experimental_prefill_resource", [runner], runner.tuning_config, inputs
                )
            assert tactic == previous["managed_tactic"]
        entries = [json.loads(f.read_text()) for f in cache.glob("v2/*/entries/*.json")]
        matches = [
            e
            for e in entries
            if e.get("runner") == "ResourceCapRunner"
            and e.get("tactic") == tactic
            and runner.identity in e.get("key", "")
            and certificate["checksum"] in e.get("key", "")
        ]
        assert chosen is runner and tactic in (-1, 0, 1) and len(matches) == 1
        result = {
            "identity": runner.identity,
            "certificate_checksum": certificate["checksum"],
            "certificate_accepted": runner._receipt_valid,
            "managed_tactic": tactic,
            "initial_output_lse_exact": exact(initial, reference),
            "stored_winner": matches[0],
            "producer_pid": os.getpid() if producer is None else prior_env["pid"],
            "consumer_pid": None if producer is None else os.getpid(),
        }
        save(out / name / "result.json", result)
        assert result["initial_output_lse_exact"]
        with autotune_v2(mode="replay", measurement_policy=policy, cache_root=cache):
            if mode == "eager_run":
                plan()
                rebound = runner.rebind_same_geometry(inputs) if tactic == 1 else False
                updated = runner.run(inputs) if rebound else native()
                result["replanned_output_lse_exact"] = exact(updated, reference)
                result["resource_rebound"] = rebound
                assert tactic != 1 or rebound
                save(out / name / "result.json", result)
                assert result["replanned_output_lse_exact"]
            else:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(count):
                        captured = runner.run(inputs)
                graph.replay()
                torch.cuda.synchronize()
                result["captured_output_lse_exact"] = exact(captured, reference)
                save(out / name / "result.json", result)
                assert result["captured_output_lse_exact"]
                original = q.clone()
                q.mul_(0.75)
                updated_reference = clone_pair(native())
                output.fill_(float("nan"))
                lse.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                result["same_pointer_input_update_output_lse_exact"] = exact(
                    captured, updated_reference
                )
                save(out / name / "result.json", result)
                assert result["same_pointer_input_update_output_lse_exact"]
                q.copy_(original)
        results[name] = result
    save(
        out / "complete.json",
        dict(
            pass_=True,
            **{"pass": True},
            modes=results,
            fresh_cases_consumed=0,
            performance_qualification=False,
            default_promotion=False,
            serving_promotion=False,
            files={str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--producer", type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    args = parser.parse_args()
    run(args.out, args.producer, args.expected_source_commit)
