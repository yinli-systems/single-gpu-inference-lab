"""Shared helpers for reproducible paired GPU microbenchmarks."""

from __future__ import annotations

import itertools
import statistics
from collections.abc import Sequence


def summarize_samples(samples: Sequence[float]) -> dict[str, object]:
    if not samples:
        raise ValueError("at least one sample is required")
    values = list(samples)
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(values),
        "p10_ms": ordered[round(0.10 * (len(ordered) - 1))],
        "p90_ms": ordered[round(0.90 * (len(ordered) - 1))],
        "samples_ms": values,
    }


def summarize_trials(trials: Sequence[Sequence[float]]) -> dict[str, object]:
    if not trials:
        raise ValueError("at least one trial is required")
    summaries = [
        {"trial": trial_index, **summarize_samples(samples)}
        for trial_index, samples in enumerate(trials)
    ]
    medians = [float(summary["median_ms"]) for summary in summaries]
    return {
        "median_ms": statistics.median(medians),
        "min_trial_median_ms": min(medians),
        "max_trial_median_ms": max(medians),
        "trial_medians_ms": medians,
        "trials": summaries,
    }


def paired_trial_ratios(
    baseline_trial_medians: Sequence[float],
    candidate_trial_medians: Sequence[float],
) -> list[float]:
    if len(baseline_trial_medians) != len(candidate_trial_medians):
        raise ValueError("paired trial counts must match")
    if not baseline_trial_medians:
        raise ValueError("at least one paired trial is required")
    if any(value <= 0 for value in candidate_trial_medians):
        raise ValueError("candidate trial medians must be positive")
    return [
        baseline / candidate
        for baseline, candidate in zip(
            baseline_trial_medians,
            candidate_trial_medians,
        )
    ]


def summarize_paired_speedups(samples: Sequence[float]) -> dict[str, object]:
    if not samples:
        raise ValueError("at least one paired speedup is required")
    values = list(samples)
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "all_trials_faster": all(value > 1.0 for value in values),
        "paired_trial_values": values,
    }


def permuted_provider_order(
    provider_names: Sequence[str],
    *,
    trial: int,
    round_index: int,
    seed: int,
) -> tuple[str, ...]:
    names = tuple(provider_names)
    if not names:
        raise ValueError("at least one provider is required")
    if len(names) > 6:
        raise ValueError("permutation order supports at most six providers")
    orders = tuple(itertools.permutations(names))
    return orders[(seed + trial + round_index) % len(orders)]
