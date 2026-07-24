#!/usr/bin/env python3
"""Benchmark fused top-logprobs selection.

The target boundary is token logprob reporting: select top-N token IDs and
normalized logprobs without materializing a full ``[batch, vocab]`` log-softmax
tensor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - benchmark requires CUDA PyTorch
    torch = None

try:
    import triton
except ImportError:  # pragma: no cover - benchmark requires Triton
    triton = None

from l20_stack.benchmark_protocol import (
    paired_trial_ratios,
    permuted_provider_order,
    summarize_paired_speedups,
    summarize_trials,
)
from l20_stack.ops.triton_sampling import (
    logprob_topk_launch_config,
    top_logprobs,
    top_logprobs_out,
    top_logprobs_reference,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--block-vocab", type=int)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument(
        "--clock-policy",
        choices=("unconditioned", "steady-state-gemm"),
        default="unconditioned",
        help=(
            "Use 'steady-state-gemm' to enqueue the same GEMM before every timed "
            "provider call. The GEMM is outside the CUDA event interval."
        ),
    )
    parser.add_argument("--precondition-size", type=int, default=8192)
    parser.add_argument("--precondition-repeats", type=int, default=1)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in ("batch", "vocab", "top_n", "rounds", "trials"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and positive")
    if args.top_n > args.vocab:
        parser.error("--top-n cannot exceed --vocab")
    if args.precondition_size <= 0:
        parser.error("--precondition-size must be positive")
    if args.precondition_repeats <= 0:
        parser.error("--precondition-repeats must be positive")
    return args


def benchmark_interleaved(
    providers,
    *,
    precondition,
    warmup: int,
    rounds: int,
    trials: int,
    order_seed: int,
) -> dict[str, list[list[float]]]:
    """Benchmark providers in a balanced order with paired per-trial samples.

    Events are allocated before measurement. Each provider receives the same
    optional same-stream precondition immediately before its start event. All
    events in a trial are synchronized together so host gaps cannot selectively
    downclock one provider.
    """

    names = tuple(providers)
    samples = {name: [[] for _ in range(trials)] for name in names}
    event_pairs = {
        name: [
            [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(rounds)
            ]
            for _ in range(trials)
        ]
        for name in names
    }

    for trial in range(trials):
        for warmup_index in range(warmup):
            order = permuted_provider_order(
                names,
                trial=trial,
                round_index=warmup_index,
                seed=order_seed,
            )
            for name in order:
                if precondition is not None:
                    precondition()
                providers[name]()
        torch.cuda.synchronize()

        recorded = []
        for round_index in range(rounds):
            order = permuted_provider_order(
                names,
                trial=trial,
                round_index=round_index,
                seed=order_seed,
            )
            for name in order:
                start, end = event_pairs[name][trial][round_index]
                if precondition is not None:
                    precondition()
                start.record()
                providers[name]()
                end.record()
                recorded.append((name, start, end))
        torch.cuda.synchronize()
        for name, start, end in recorded:
            samples[name][trial].append(start.elapsed_time(end))
    return samples


def make_clock_precondition(args):
    if args.clock_policy == "unconditioned":
        return None, {
            "policy": "unconditioned",
            "workload": None,
            "same_stream": True,
            "included_in_timing": False,
        }

    matrix_dtype = torch.float16
    lhs = torch.randn(
        (args.precondition_size, args.precondition_size),
        device="cuda",
        dtype=matrix_dtype,
    )
    rhs = torch.randn_like(lhs)
    output = torch.empty_like(lhs)

    def precondition():
        for _ in range(args.precondition_repeats):
            torch.mm(lhs, rhs, out=output)

    return precondition, {
        "policy": "steady-state-gemm",
        "workload": {
            "operation": "torch.mm",
            "shape": [args.precondition_size, args.precondition_size],
            "dtype": str(matrix_dtype).removeprefix("torch."),
            "repeats_per_provider_sample": args.precondition_repeats,
            "allocated_bytes": sum(
                tensor.numel() * tensor.element_size()
                for tensor in (lhs, rhs, output)
            ),
        },
        "same_stream": True,
        "included_in_timing": False,
    }


NVIDIA_SMI_FIELDS = (
    "timestamp",
    "uuid",
    "name",
    "driver_version",
    "pstate",
    "clocks.current.sm",
    "clocks.max.sm",
    "clocks.current.memory",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "utilization.gpu",
    "memory.used",
    "memory.total",
)


def nvidia_smi_snapshot() -> dict[str, object]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(NVIDIA_SMI_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    rows = [
        dict(zip(NVIDIA_SMI_FIELDS, row))
        for row in csv.reader(completed.stdout.splitlines(), skipinitialspace=True)
        if row
    ]
    return {"available": True, "rows": rows}


def sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def git_provenance(repo_root: Path) -> dict[str, object]:
    def git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def tie_aware_correctness(logits, values, tokens, *, temperature: float):
    reference_values, reference_tokens = top_logprobs_reference(
        logits,
        top_n=values.shape[1],
        temperature=temperature,
    )
    normalized = torch.log_softmax(logits.float() / temperature, dim=-1)
    gathered_values = normalized.gather(-1, tokens)
    raw_logits = logits.float()
    gathered_raw_logits = raw_logits.gather(-1, tokens)
    reference_raw_logits = raw_logits.gather(-1, reference_tokens)
    cutoff = reference_raw_logits[:, -1:]
    selected_at_or_above_cutoff = bool(
        torch.all(gathered_raw_logits >= cutoff).item()
    )
    output_is_nonincreasing = bool(
        torch.all(gathered_raw_logits[:, :-1] >= gathered_raw_logits[:, 1:]).item()
    )
    strict_reference_mask = reference_raw_logits > cutoff
    reference_present = (
        tokens.unsqueeze(-1) == reference_tokens.unsqueeze(-2)
    ).any(dim=1)
    all_strict_top_tokens_present = bool(
        torch.all(reference_present | ~strict_reference_mask).item()
    )
    sorted_values = torch.sort(values, dim=-1, descending=True).values
    sorted_reference = torch.sort(reference_values, dim=-1, descending=True).values
    max_abs_error = float(torch.max(torch.abs(sorted_values - sorted_reference)).item())
    gathered_value_error = float(torch.max(torch.abs(values - gathered_values)).item())
    unique_tokens = all(
        len(set(row)) == len(row)
        for row in tokens.detach().cpu().tolist()
    )
    tie_aware_match = (
        unique_tokens
        and output_is_nonincreasing
        and selected_at_or_above_cutoff
        and all_strict_top_tokens_present
        and max_abs_error <= 5e-3
        and gathered_value_error <= 5e-3
    )
    return {
        "tie_aware_match": tie_aware_match,
        "output_is_nonincreasing": output_is_nonincreasing,
        "selected_at_or_above_exact_cutoff": selected_at_or_above_cutoff,
        "all_strict_top_tokens_present": all_strict_top_tokens_present,
        "tokens_match_exact_order": torch.equal(tokens, reference_tokens),
        "token_sets_match": torch.equal(
            torch.sort(tokens, dim=-1).values,
            torch.sort(reference_tokens, dim=-1).values,
        ),
        "tokens_unique_per_row": unique_tokens,
        "max_abs_logprob_error": max_abs_error,
        "max_reported_vs_gathered_error": gathered_value_error,
    }


def main() -> int:
    args = parse_args()
    if torch is None or triton is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    repo_root = Path(__file__).resolve().parents[1]
    kernel_source = repo_root / "src/l20_stack/ops/triton_sampling.py"
    clock_before = nvidia_smi_snapshot()
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    logits = torch.randn((args.batch, args.vocab), device="cuda", dtype=dtype)
    config = logprob_topk_launch_config(
        args.vocab,
        args.top_n,
        batch=args.batch,
        block_vocab_override=args.block_vocab,
    )
    output_values = torch.empty((args.batch, args.top_n), device="cuda", dtype=torch.float32)
    output_tokens = torch.empty((args.batch, args.top_n), device="cuda", dtype=torch.int64)
    partial_shape = (args.batch, config.blocks_per_row, args.top_n)
    partial_values = torch.empty(partial_shape, device="cuda", dtype=torch.float32)
    partial_tokens = torch.empty(partial_shape, device="cuda", dtype=torch.int64)
    partial_max = torch.empty((args.batch, config.blocks_per_row), device="cuda", dtype=torch.float32)
    partial_sum_exp = torch.empty(
        (args.batch, config.blocks_per_row), device="cuda", dtype=torch.float32
    )

    expected_values, expected_tokens = top_logprobs_reference(
        logits,
        top_n=args.top_n,
        temperature=args.temperature,
    )
    actual_values, actual_tokens = top_logprobs(
        logits,
        top_n=args.top_n,
        temperature=args.temperature,
        block_vocab_override=args.block_vocab,
    )
    torch.cuda.synchronize()
    allocating_correctness = tie_aware_correctness(
        logits,
        actual_values,
        actual_tokens,
        temperature=args.temperature,
    )
    if not allocating_correctness["tie_aware_match"]:
        raise AssertionError(f"allocating top-logprobs mismatch: {allocating_correctness}")

    top_logprobs_out(
        logits,
        output_values,
        output_tokens,
        partial_values=partial_values,
        partial_tokens=partial_tokens,
        partial_max=partial_max,
        partial_sum_exp=partial_sum_exp,
        top_n=args.top_n,
        temperature=args.temperature,
        block_vocab_override=args.block_vocab,
    )
    torch.cuda.synchronize()
    preallocated_correctness = tie_aware_correctness(
        logits,
        output_values,
        output_tokens,
        temperature=args.temperature,
    )
    if not preallocated_correctness["tie_aware_match"]:
        raise AssertionError(f"preallocated top-logprobs mismatch: {preallocated_correctness}")
    preallocated_matches_allocating = bool(
        torch.equal(output_tokens, actual_tokens)
        and torch.allclose(output_values, actual_values, atol=0.0, rtol=0.0)
    )
    if not preallocated_matches_allocating:
        raise AssertionError("preallocated and allocating top-logprobs paths differ")

    def triton_preallocated():
        top_logprobs_out(
            logits,
            output_values,
            output_tokens,
            partial_values=partial_values,
            partial_tokens=partial_tokens,
            partial_max=partial_max,
            partial_sum_exp=partial_sum_exp,
            top_n=args.top_n,
            temperature=args.temperature,
            block_vocab_override=args.block_vocab,
        )

    def torch_logsoftmax_topk():
        return torch.topk(
            torch.log_softmax(logits.float() / args.temperature, dim=-1),
            args.top_n,
            dim=-1,
        )

    def torch_logsumexp_topk():
        scaled = logits.float() / args.temperature
        values, tokens = torch.topk(scaled, args.top_n, dim=-1)
        return values - torch.logsumexp(scaled, dim=-1, keepdim=True), tokens

    providers = {
        "triton_top_logprobs_preallocated": triton_preallocated,
        "torch_logsoftmax_then_topk": torch_logsoftmax_topk,
        "torch_logsumexp_then_topk": torch_logsumexp_topk,
    }
    precondition, clock_policy = make_clock_precondition(args)
    provider_trials = benchmark_interleaved(
        providers,
        precondition=precondition,
        warmup=args.warmup,
        rounds=args.rounds,
        trials=args.trials,
        order_seed=args.seed,
    )
    provider_results = {
        name: summarize_trials(trials)
        for name, trials in provider_trials.items()
    }
    fused_trial_medians = provider_results[
        "triton_top_logprobs_preallocated"
    ]["trial_medians_ms"]
    logsoftmax_trial_medians = provider_results[
        "torch_logsoftmax_then_topk"
    ]["trial_medians_ms"]
    logsumexp_trial_medians = provider_results[
        "torch_logsumexp_then_topk"
    ]["trial_medians_ms"]
    paired_logsoftmax = paired_trial_ratios(
        logsoftmax_trial_medians,
        fused_trial_medians,
    )
    paired_logsumexp = paired_trial_ratios(
        logsumexp_trial_medians,
        fused_trial_medians,
    )
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    clock_after = nvidia_smi_snapshot()
    result = {
        "schema_version": 2,
        "hardware": torch.cuda.get_device_name(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            **git_provenance(repo_root),
            "benchmark_script_sha256": sha256_file(Path(__file__).resolve()),
            "kernel_source_sha256": sha256_file(kernel_source),
            "command": [sys.executable, *sys.argv],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "triton": triton.__version__,
            "gpu": {
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
                "uuid": str(getattr(properties, "uuid", "unknown")),
            },
            "nvidia_smi_snapshots": {
                "scope": (
                    "host snapshots before setup and after measurement; "
                    "not sampled inside CUDA event intervals"
                ),
                "before_setup": clock_before,
                "after_measurement": clock_after,
            },
        },
        "shape": {
            "batch": args.batch,
            "vocab": args.vocab,
            "top_n": args.top_n,
            "temperature": args.temperature,
            "dtype": args.dtype,
        },
        "launch": config.to_dict(),
        "timing_policy": {
            "clock": clock_policy,
            "provider_order": {
                "strategy": (
                    "deterministic_all_permutations_for_within_round_position_balance"
                ),
                "seed": args.seed,
                "providers": list(providers),
                "cross_round_carryover": (
                    "controlled by the identical per-provider GEMM"
                    if precondition is not None
                    else "uncontrolled"
                ),
            },
            "synchronization": "once_after_all_recorded_events_in_each_trial",
            "events_allocated_before_measurement": True,
            "warmup_rounds_per_provider_per_trial": args.warmup,
            "measured_rounds_per_provider_per_trial": args.rounds,
            "trials": args.trials,
            "speedup_aggregation": (
                "paired baseline/fused trial medians, then median/min/max"
            ),
        },
        "comparator_contract": {
            "triton": "caller-owned output and workspace tensors",
            "pytorch": (
                "eager composed baseline including GPU kernels and "
                "full-vocabulary FP32 temporary materialization"
            ),
            "timing_excludes": "host launch and allocator latency",
            "claim_boundary": (
                "CUDA-event operator microbenchmark; not host latency or "
                "serving throughput"
            ),
        },
        "correctness": {
            **preallocated_correctness,
            "preallocated_matches_allocating": preallocated_matches_allocating,
            "reference_tokens_match_exact_order": torch.equal(
                expected_tokens,
                output_tokens,
            ),
            "reference_values_max_abs_error": float(
                torch.max(torch.abs(expected_values - output_values)).item()
            ),
        },
        "providers": provider_results,
        "paired_speedups": {
            "vs_torch_logsoftmax_then_topk": summarize_paired_speedups(
                paired_logsoftmax
            ),
            "vs_torch_logsumexp_then_topk": summarize_paired_speedups(
                paired_logsumexp
            ),
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
