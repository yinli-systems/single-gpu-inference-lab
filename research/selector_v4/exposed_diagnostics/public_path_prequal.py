"""Exposed public-path measurements with independent training and scoring.

This development experiment never dispatches fresh cases or authorizes release.
Native plan+run and native plan+public rebind+real v2 run are timed in eager mode.
Graph1/16 measure replay only. Every completed window is durable before checks.
"""

import argparse
import gc
import itertools
import json
import math
import os
import time
from pathlib import Path

from public_path_contract import CONTRACT, digest, full_call_choice


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def tensor_hash(tensor):
    import hashlib

    import torch

    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def data(case, dtype, layout, torch):
    qs = case["q"]
    lengths = [q + k for q, k in zip(qs, case["cached"], strict=True)]
    ip = lambda values: torch.tensor(
        [0, *itertools.accumulate(values)], device="cuda", dtype=torch.int32
    )
    q = torch.randn(sum(qs), 32, 128, dtype=dtype, device="cuda")
    k = torch.randn(sum(lengths), 8, 128, dtype=dtype, device="cuda")
    v = torch.randn_like(k)
    metadata = [ip(qs), ip(lengths)]
    if layout == "paged":
        counts = [(length + 15) // 16 for length in lengths]
        total = sum(counts)
        order = torch.randperm(total, device="cuda")
        keys = torch.zeros(total, 16, 8, 128, dtype=dtype, device="cuda")
        values = torch.zeros_like(keys)
        offset = 0
        for length, count, key, value in zip(
            lengths, counts, k.split(lengths), v.split(lengths), strict=True
        ):
            flat_k = torch.zeros(count * 16, 8, 128, dtype=dtype, device="cuda")
            flat_v = torch.zeros_like(flat_k)
            flat_k[:length].copy_(key)
            flat_v[:length].copy_(value)
            index = order[offset : offset + count]
            keys[index] = flat_k.view(count, 16, 8, 128)
            values[index] = flat_v.view(count, 16, 8, 128)
            offset += count
        metadata = [
            metadata[0],
            ip(counts),
            order.to(torch.int32),
            torch.tensor(
                [(length - 1) % 16 + 1 for length in lengths], dtype=torch.int32, device="cuda"
            ),
        ]
        k, v = keys, values
    return [q, k, v], metadata, lengths


def run(args):
    import flashinfer
    import torch
    from flashinfer._build_meta import __git_commit__, __version__

    binding = json.loads((args.campaign / "binding.json").read_text())
    expected = (
        binding["pristine_commit"] if args.role == "pristine" else binding["candidate_commit"]
    )
    assert (__git_commit__, __version__) == (expected, "0.7.1")
    assert binding["contract"] == CONTRACT and args.rep in (0, 1, 2)
    case = binding["cases"][args.case]
    assert case["family"] == "dev" and case["exposed_development"]
    args.out.mkdir()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(affinity[:2]))
    props = torch.cuda.get_device_properties(0)
    job_root = args.out.parent
    environment = {
        "source_commit": expected,
        "role": args.role,
        "rep": args.rep,
        "pid": os.getpid(),
        "gpu_uuid": str(props.uuid),
        "gpu_name": props.name,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "profiled_scoring": False,
        "case": case,
        "fresh_cases_consumed": 0,
    }
    save(args.out / "environment.json", environment)
    results = {}

    def process_cell(dtype_name, layout, requested):
        cell = digest([case["id"], dtype_name, layout, requested])
        torch.manual_seed(int(cell[:8], 16))
        inputs, metadata, lengths = data(case, getattr(torch, dtype_name), layout, torch)
        q, k, v = inputs
        workspace = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
        if layout == "ragged":
            wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, backend="fa2")
            plan = lambda: wrapper.plan(
                *metadata,
                32,
                8,
                128,
                causal=True,
                q_data_type=q.dtype,
                kv_data_type=q.dtype,
                disable_split_kv=requested == "unsplit",
            )
        else:
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace, "NHD", backend="fa2"
            )
            plan = lambda: wrapper.plan(
                *metadata,
                32,
                8,
                128,
                16,
                causal=True,
                q_data_type=q.dtype,
                kv_data_type=q.dtype,
                disable_split_kv=requested == "unsplit",
            )
        output = torch.empty_like(q)
        lse = torch.empty(sum(case["q"]), 32, dtype=torch.float32, device="cuda")
        kwargs = {"out": output, "lse": lse, "return_lse": True}
        native = lambda: (
            wrapper.run(q, (k, v), **kwargs)
            if layout == "paged"
            else wrapper.run(*inputs, **kwargs)
        )
        plan()
        assert len(wrapper._plan_info) == 15
        reference_path = job_root / "references" / (cell + ".pt")
        original = native()
        input_hashes = [tensor_hash(t) for t in inputs]
        if args.role == "pristine" and args.rep == 0:
            baseline = {
                "out": original[0].cpu(),
                "lse": original[1].cpu(),
                "inputs_sha256": input_hashes,
                "plan": list(wrapper._plan_info),
                "environment": environment,
                "windows": {},
            }
        else:
            baseline = torch.load(reference_path, map_location="cpu", weights_only=True)
            assert input_hashes == baseline["inputs_sha256"]
            assert baseline["environment"]["gpu_uuid"] == environment["gpu_uuid"]
            assert baseline["environment"]["cpu_affinity"] == environment["cpu_affinity"]
            assert list(wrapper._plan_info) == baseline["plan"]
        reference = [baseline["out"].to("cuda"), baseline["lse"].to("cuda")]

        def exact(values):
            return all(torch.equal(a, b) for a, b in zip(values, reference, strict=True))

        assert exact(original)

        def process_execution(execution):
            mode = "eager_run" if execution == "eager_full_call" else "graph_replay"
            count = 16 if execution == "graph16_replay" else 1
            key = cell + "-" + execution
            root = args.out / key
            root.mkdir()
            plan()
            runner = None
            choice = "native"
            managed_tactic = -1
            certificate = None
            source_artifact = None
            if args.role != "pristine":
                from flashinfer import MeasurementPolicy, autotune_v2
                from flashinfer.autotuner import AutoTuner
                from flashinfer.experimental.prefill_resource._policy import accepts
                from flashinfer.prefill import make_prefill_resource_runner

                runner = make_prefill_resource_runner(
                    wrapper,
                    inputs,
                    qo_lengths=case["q"],
                    kv_lengths=lengths,
                    execution_mode=mode,
                    graph_replays=count,
                    run_kwargs=kwargs,
                )
                runner.tuning_config.profiling_repeat = 256
                policy = MeasurementPolicy(
                    execution_mode="eager" if mode == "eager_run" else "cuda_graph"
                )
                if args.role == "train":
                    certificate = runner.calibrate(inputs)
                    save(root / "certificates" / (runner.identity + ".json"), certificate)
                    cache = root / "managed-cache"
                    with autotune_v2(mode="tune", measurement_policy=policy, cache_root=cache):
                        initial = runner.run(inputs)
                        chosen, managed_tactic = AutoTuner.get().choose_one(
                            "experimental_prefill_resource", [runner], runner.tuning_config, inputs
                        )
                    assert exact(initial) and chosen is runner
                    assert (
                        AutoTuner.get().stats.tuned_op_successful_configs.get(
                            "experimental_prefill_resource", 0
                        )
                        > 0
                    )
                    assert not AutoTuner.get().stats.failed_tactics.get(
                        "experimental_prefill_resource::ResourceCapRunner"
                    )
                    choice = "resource" if managed_tactic == 1 else "native"
                else:
                    training = {
                        i: json.loads((job_root / f"train-{i}" / key / "complete.json").read_text())
                        for i in range(3)
                    }
                    for item in training.values():
                        assert item["environment"]["gpu_uuid"] == environment["gpu_uuid"]
                        assert item["environment"]["cpu_affinity"] == environment["cpu_affinity"]
                        assert item["identity"] == runner.identity
                        assert item["certificate_accepted"] == accepts(
                            item["certificate"], runner.identity
                        )
                    decision = full_call_choice(training, args.rep, key + ":" + runner.identity)
                    save(root / "frozen-choice-before-scoring.json", decision)
                    choice = decision["choice"]
                    source_artifact = job_root / f"train-{decision['primary_artifact']}" / key
                    primary = training[decision["primary_artifact"]]
                    assert (
                        runner.load_receipt(source_artifact / "certificates")
                        == primary["certificate_accepted"]
                    )
                    certificate = runner.receipt
                    assert certificate["checksum"] == primary["certificate"]["checksum"]
                    cache = source_artifact / "managed-cache"
                    if runner._receipt_valid:
                        assert runner._prepare(inputs)
                    with autotune_v2(mode="replay", measurement_policy=policy, cache_root=cache):
                        tuner = AutoTuner.get()
                        config = tuner._apply_measure_policy(runner.tuning_config, policy)
                        hit, index, managed_tactic, _ = tuner.search_cache(
                            "experimental_prefill_resource",
                            [runner],
                            tuple(tuner._get_input_sizes(inputs)),
                            config,
                            inputs=inputs,
                        )
                        assert hit and index == 0 and managed_tactic == primary["managed_tactic"]
                entries = [json.loads(p.read_text()) for p in cache.glob("v2/*/entries/*.json")]
                matches = [
                    e
                    for e in entries
                    if e.get("runner") == "ResourceCapRunner"
                    and e.get("tactic") == managed_tactic
                    and runner.identity in e.get("key", "")
                    and certificate["checksum"] in e.get("key", "")
                ]
                assert len(matches) == 1
                save(root / "managed-winner.json", matches[0])
                # The serving context attaches once before any scored window.
                context = autotune_v2(mode="replay", measurement_policy=policy, cache_root=cache)
                context.__enter__()

            resource_requested = managed_tactic == 1
            binding_failures = 0

            def eager(use_resource):
                nonlocal binding_failures
                plan()
                if use_resource:
                    if runner.rebind_same_geometry(inputs):
                        return runner.run(inputs)
                    binding_failures += 1
                return native()

            graphs = {}
            if mode == "graph_replay":
                for arm in (
                    ["native", "oracle"] if args.role == "policy" else ["native", "candidate"]
                ):
                    use_resource = arm != "native" and resource_requested
                    function = (lambda: runner.run(inputs)) if use_resource else native
                    for _ in range(3):
                        function()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(count):
                            captured = function()
                    graph.replay()
                    torch.cuda.synchronize()
                    assert exact(captured)
                    graphs[arm] = graph
                if args.role == "policy":
                    graphs["policy"] = (
                        graphs["oracle"] if choice == "resource" else graphs["native"]
                    )

            def call(arm):
                if mode == "graph_replay":
                    graphs[arm].replay()
                    return output, lse
                use_resource = (
                    arm == "oracle"
                    and resource_requested
                    or arm in ("candidate", "policy")
                    and choice == "resource"
                )
                return eager(use_resource)

            if args.role == "pristine" and args.rep == 0:
                pilots = []
                for _ in range(2):
                    for _ in range(3):
                        call("native")
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    for _ in range(16):
                        call("native")
                    torch.cuda.synchronize()
                    pilots.append((time.perf_counter_ns() - start) / 1000 / 16)
                iterations = min(
                    4096,
                    1
                    << max(
                        0, math.ceil(CONTRACT["target_window_us"] / min(pilots)) - 1
                    ).bit_length(),
                )
                baseline["windows"][execution] = {"iterations": iterations, "pilots_us": pilots}
                reference_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(baseline, reference_path)
            iterations = baseline["windows"][execution]["iterations"]
            rows = []
            paired = []
            for block in range(24):
                if args.role == "policy":
                    arms = ["native", "oracle", "policy"]
                    first = arms[block % 3 :] + arms[: block % 3]
                    order = first + list(reversed(first))
                elif args.role == "pristine":
                    order = ["native", "native", "native", "native"]
                else:
                    order = ["native", "candidate", "candidate", "native"]
                if args.role == "train" and block % 2:
                    order = ["candidate", "native", "native", "candidate"]
                part = []
                for position, arm in enumerate(order):
                    for _ in range(3):
                        call(arm)
                    torch.cuda.synchronize()
                    restore_gc = gc.isenabled()
                    gc.disable()
                    try:
                        start = time.perf_counter_ns()
                        for _ in range(iterations):
                            actual = call(arm)
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter_ns() - start) / 1000
                    finally:
                        if restore_gc:
                            gc.enable()
                    row = {
                        "block": block,
                        "position": position,
                        "arm": arm,
                        "elapsed_us": elapsed,
                        "iterations": iterations,
                        "kernel_calls": count * iterations,
                        "wall_us_per_call": elapsed / (count * iterations),
                        "exact": exact(actual),
                        "binding_failures_total": binding_failures,
                    }
                    rows.append(row)
                    save(root / "windows.json", rows)
                    assert row["exact"], "Full output/LSE mismatch retained"
                    assert elapsed >= 120000, "Short scored window retained; no adaptive resampling"
                    part.append(row)
                if args.role == "train":
                    native_values = [r["wall_us_per_call"] for r in part if r["arm"] == "native"]
                    candidate_values = [
                        r["wall_us_per_call"] for r in part if r["arm"] == "candidate"
                    ]
                    paired.append(
                        [
                            native_values[0],
                            candidate_values[0],
                            candidate_values[1],
                            native_values[1],
                        ]
                    )
                assert exact(native()), "Native after resource must remain exact"
            if args.role != "pristine":
                context.__exit__(None, None, None)
            result = {
                "complete": True,
                "environment": environment,
                "execution": execution,
                "dtype": dtype_name,
                "layout": layout,
                "requested_split": requested,
                "actual_plan": list(wrapper._plan_info),
                "identity": runner.identity if runner else None,
                "certificate": certificate,
                "certificate_accepted": runner._receipt_valid if runner else False,
                "managed_tactic": managed_tactic,
                "choice": choice,
                "every_resource_call_bound": binding_failures == 0,
                "binding_failures": binding_failures,
                "paired_blocks": paired,
                "exact": all(r["exact"] for r in rows),
                "window_count": len(rows),
                "input_hashes": input_hashes,
                "native_output_lse_sha256": [tensor_hash(t) for t in native()],
                "source_artifact": str(source_artifact) if source_artifact else None,
                "contract": CONTRACT,
                "fresh_cases_consumed": 0,
                "qualification_authority": False,
            }
            save(root / "complete.json", result)
            return result

        for execution in ["eager_full_call", "graph1_replay", "graph16_replay"]:
            results[cell + "-" + execution] = process_execution(execution)

    for dtype_name, layout, requested in itertools.product(
        ["float16", "bfloat16"], ["ragged", "paged"], ["auto", "unsplit"]
    ):
        process_cell(dtype_name, layout, requested)
        gc.collect()
        torch.cuda.empty_cache()
    save(
        args.out / "complete.json",
        {
            "complete": True,
            "cells": results,
            "fresh_cases_consumed": 0,
            "qualification_authority": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--role", choices=["pristine", "train", "policy"], required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--case", type=int, choices=[0, 1], required=True)
    run(parser.parse_args())
