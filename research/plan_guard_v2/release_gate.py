"""Fail-closed release gate for the plan-aware resource policy.

The gate consumes only precomputed holdout cells and their conditional bootstrap
log-ratio draws.  It does not infer model quality or extrapolate to other GPUs.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _summary(indices: list[int], mode: str, cells: list[dict[str, Any]], drawcache: dict[tuple[int, str], np.ndarray]) -> dict[str, Any]:
    if not indices:
        return {
            "count": 0,
            "ratio": None,
            "CI95": None,
            "worst_point_ratio": None,
            "simultaneous_worst_CI95": None,
            "controls_failed": 0,
        }
    ratios = [float(cells[i]["comparisons"][mode]["ratio"]) for i in indices]
    _require(all(math.isfinite(x) and x > 0 for x in ratios), "invalid gate ratio")
    matrix = np.stack([np.asarray(drawcache[(i, mode)], dtype=float) for i in indices])
    _require(matrix.ndim == 2 and matrix.shape[1] >= 1000 and np.isfinite(matrix).all(), "invalid gate draws")
    aggregate = matrix.mean(axis=0)
    minimum = matrix.min(axis=0)
    return {
        "count": len(indices),
        "ratio": float(math.exp(sum(math.log(x) for x in ratios) / len(ratios))),
        "CI95": [float(x) for x in np.exp(np.quantile(aggregate, [0.025, 0.975]))],
        "worst_point_ratio": min(ratios),
        # Conditional joint bootstrap of the minimum across all selected cells.
        # This is not a hardware-population guarantee.
        "simultaneous_worst_CI95": [float(x) for x in np.exp(np.quantile(minimum, [0.025, 0.975]))],
        "controls_failed": sum(not cells[i]["comparisons"][mode]["controls_resolve"] for i in indices),
    }


def evaluate_release_gate(cells: list[dict[str, Any]], drawcache: dict[tuple[int, str], np.ndarray], numerics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    primary = [i for i, c in enumerate(cells) if c["calls"] == 16 and c["metric"] == "run_device_us"]
    _require(primary, "missing Graph16 run-device cells")
    _require(all("selected" in cells[i] for i in primary), "selector decision absent")
    selected = [i for i in primary if cells[i]["selected"]]
    unselected = [i for i in primary if not cells[i]["selected"]]

    policy = _summary(primary, "guarded", cells, drawcache)
    selected_summary = _summary(selected, "guarded", cells, drawcache)
    unselected_summary = _summary(unselected, "guarded", cells, drawcache)
    off = _summary(primary, "off", cells, drawcache)

    numerical_exact = bool(numerics) and all(
        int(v["qualifications"]) == int(v["exact_full_outputs"])
        and float(v["max_abs_vs_pristine"]) == 0.0
        for v in numerics.values()
    )
    requirements = {
        "numerical_exact": numerical_exact,
        "selected_nonempty": selected_summary["count"] > 0,
        "selected_controls_resolve": selected_summary["count"] > 0 and selected_summary["controls_failed"] == 0,
        "selected_worst_point_at_least_0_99": selected_summary["count"] > 0 and selected_summary["worst_point_ratio"] >= 0.99,
        "selected_joint_worst_lcb_at_least_0_99": selected_summary["count"] > 0 and selected_summary["simultaneous_worst_CI95"][0] >= 0.99,
        "selected_aggregate_lcb_above_1": selected_summary["count"] > 0 and selected_summary["CI95"][0] > 1.0,
        "policy_worst_point_at_least_0_99": policy["worst_point_ratio"] >= 0.99,
        "off_overlay_worst_point_at_least_0_99": off["worst_point_ratio"] >= 0.99,
    }
    return {
        "pass": all(requirements.values()),
        "requirements": requirements,
        "policy": policy,
        "selected": selected_summary,
        "unselected": unselected_summary,
        "off_overlay": off,
        "scope": "Frozen release cases on one GPU family; joint bootstrap is conditional on measured process/block variation and is not a device-population guarantee.",
    }
