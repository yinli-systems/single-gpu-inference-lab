"""A throughput win cannot hide a workload goodput or latency-tail regression."""

import copy

import pytest

from research.selector_v4.serving.metric_gate import METRICS, WORKLOADS, evaluate_metrics


def good():
    base = {w: {m: [10.0] * 4 for m in METRICS} for w in WORKLOADS}
    candidate = copy.deepcopy(base)
    for w in WORKLOADS:
        candidate[w]["output_tokens_per_second"] = [12.0] * 4
    return [copy.deepcopy({"pristine": base, "candidate": candidate}) for _ in range(3)]


def test_all_metrics_and_joint_safety_are_required_even_with_throughput_gain():
    result = evaluate_metrics(good())
    assert result["metric_pass"] and len(result["comparisons"]) == 40
    assert result["joint_min_lcb95"] >= 0.99 and not result["full_http_qualified"]


@pytest.mark.parametrize(
    "metric,value", [("strict_slo_goodput", 9.8), ("TTFT-p99", 10.3), ("TPOT-p95", 10.3)]
)
def test_one_workload_regression_blocks_the_complete_metric_verdict(metric, value):
    allocations = good()
    for a in allocations:
        a["candidate"]["mixed"][metric] = [value] * 4
    result = evaluate_metrics(allocations)
    assert result["throughput_geomean"] > 1 and not result["metric_pass"]
    assert not result["requirements"]["all_40_workload_metric_points_at_least_0_99"]


def test_independent_process_variation_blocks_uncertain_nonregression():
    allocations = good()
    for a, value in zip(allocations, [8.0, 10.0, 12.0]):
        a["candidate"]["decode"]["TTFT-p99"] = [value] * 4
    result = evaluate_metrics(allocations)
    assert result["requirements"]["all_40_workload_metric_points_at_least_0_99"]
    assert not result["requirements"]["all_40_joint_min_lcb95_at_least_0_99"]


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf")])
def test_invalid_or_zero_goodput_never_gets_an_epsilon_ratio(value):
    allocations = good()
    allocations[0]["candidate"]["decode"]["strict_slo_goodput"][0] = value
    with pytest.raises(ValueError, match="no epsilon or trimming"):
        evaluate_metrics(allocations)


def test_missing_allocation_workload_metric_or_block_is_rejected():
    with pytest.raises(ValueError, match="three"):
        evaluate_metrics(good()[:2])
    allocations = good()
    del allocations[0]["candidate"]["decode"]
    with pytest.raises(ValueError, match="workload"):
        evaluate_metrics(allocations)
    allocations = good()
    del allocations[0]["candidate"]["decode"]["TPOT-p99"]
    with pytest.raises(ValueError, match="metric"):
        evaluate_metrics(allocations)
    allocations = good()
    allocations[0]["candidate"]["decode"]["TPOT-p99"] = [10.0] * 3
    with pytest.raises((ValueError, TypeError)):
        evaluate_metrics(allocations)
