"""Full-serving metric safety; this does not authorize correctness or promotion.

Unlike the old static-selector report, every workload's goodput and latency tails
block the metric verdict. Freeze this contract before any performance trials.
All records enter paired-allocation, independent within-arm block bootstrap.
"""

import math

import numpy as np

WORKLOADS = ("guarded_prefix", "balanced_prefix", "short_prefill", "decode", "mixed")
METRICS = (
    "output_tokens_per_second",
    "strict_slo_goodput",
    "TTFT-p50",
    "TTFT-p95",
    "TTFT-p99",
    "TPOT-p50",
    "TPOT-p95",
    "TPOT-p99",
)
HIGHER_IS_BETTER = frozenset(METRICS[:2])
FLOOR = 0.99
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 2026100143


def evaluate_metrics(allocations):
    """Input: three allocations, each arm/workload/metric -> four block values.

    Source/model/workload/cache/coverage/error/Graph/parity checks are separate
    mandatory gates. A metric PASS cannot substitute for any of those checks.
    Zero goodput is an unresolved SLO/ratio condition, never an epsilon ratio.
    """
    if len(allocations) != 3:
        raise ValueError("Exactly three independent paired allocations required")
    for allocation in allocations:
        if set(allocation) != {"pristine", "candidate"}:
            raise ValueError("Both complete arms required")
        for arm in allocation.values():
            if set(arm) != set(WORKLOADS):
                raise ValueError("Every workload required")
            for workload in arm.values():
                if set(workload) != set(METRICS):
                    raise ValueError("Every throughput/goodput/latency metric required")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    process = rng.integers(0, 3, size=(BOOTSTRAP_DRAWS, 3))
    base_blocks = rng.integers(0, 4, size=(BOOTSTRAP_DRAWS, 3, 4))
    candidate_blocks = rng.integers(0, 4, size=(BOOTSTRAP_DRAWS, 3, 4))
    comparisons, distribution = [], []
    for workload in WORKLOADS:
        for metric in METRICS:
            base = np.asarray(
                [a["pristine"][workload][metric] for a in allocations], dtype=np.float64
            )
            candidate = np.asarray(
                [a["candidate"][workload][metric] for a in allocations], dtype=np.float64
            )
            if (
                base.shape != (3, 4)
                or candidate.shape != (3, 4)
                or not np.isfinite(base).all()
                or not np.isfinite(candidate).all()
                or np.any(base <= 0)
                or np.any(candidate <= 0)
            ):
                raise ValueError(
                    "Four complete positive finite blocks required; no epsilon or trimming"
                )
            sign = 1 if metric in HIGHER_IS_BETTER else -1
            base_log, candidate_log = np.log(base), np.log(candidate)
            point = math.exp(sign * float(candidate_log.mean() - base_log.mean()))
            boot = sign * (
                candidate_log[process[:, :, None], candidate_blocks].mean(axis=(1, 2))
                - base_log[process[:, :, None], base_blocks].mean(axis=(1, 2))
            )
            ci = np.exp(np.quantile(boot, [0.025, 0.975])).tolist()
            comparisons.append(
                {
                    "workload": workload,
                    "metric": metric,
                    "favorable_ratio": point,
                    "ci95": ci,
                    "point_floor_pass": point >= FLOOR,
                }
            )
            distribution.append(boot)
    joint = float(np.exp(np.quantile(np.stack(distribution).min(axis=0), 0.025)))
    throughput = [c for c in comparisons if c["metric"] == "output_tokens_per_second"]
    prefix = next(c for c in throughput if c["workload"] == "guarded_prefix")
    throughput_gain = math.exp(sum(math.log(c["favorable_ratio"]) for c in throughput) / 5)
    checks = {
        "all_40_workload_metric_points_at_least_0_99": all(
            c["point_floor_pass"] for c in comparisons
        ),
        "all_40_joint_min_lcb95_at_least_0_99": joint >= FLOOR,
        "throughput_geomean_above_one": throughput_gain > 1,
        "guarded_prefix_throughput_lcb95_above_one": prefix["ci95"][0] > 1,
    }
    return {
        "metric_pass": all(checks.values()),
        "requirements": checks,
        "comparisons": comparisons,
        "joint_min_lcb95": joint,
        "throughput_geomean": throughput_gain,
        "trimmed_blocks": 0,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "correctness_qualified": False,
        "full_http_qualified": False,
        "qualification_authority": False,
    }
