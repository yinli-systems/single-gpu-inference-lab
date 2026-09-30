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


def subset_gate_inputs(
    cells: list[dict[str, Any]],
    drawcache: dict[tuple[int, str], np.ndarray],
    excluded_cases: set[str],
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], np.ndarray]]:
    """Return a re-indexed sensitivity subset without changing the primary gate.

    This helper is analysis-only. It never mutates cells, selector decisions, or the
    frozen measurement manifest. Every retained comparison arm must have draws.
    """
    _require(bool(excluded_cases), "empty sensitivity exclusion")
    retained = [i for i, cell in enumerate(cells) if cell.get("case") not in excluded_cases]
    observed = {str(cell.get("case")) for cell in cells if cell.get("case") in excluded_cases}
    _require(observed == excluded_cases, "sensitivity exclusion not present in measured cells")
    _require(retained and len(retained) < len(cells), "invalid sensitivity subset")
    subset = [cells[i] for i in retained]
    remapped: dict[tuple[int, str], np.ndarray] = {}
    for new_index, old_index in enumerate(retained):
        modes = set(cells[old_index].get("comparisons", {}))
        _require({"off", "guarded"} <= modes, "incomplete comparison arms")
        for mode in modes:
            key = (old_index, mode)
            _require(key in drawcache, "missing sensitivity draws")
            remapped[(new_index, mode)] = np.asarray(drawcache[key], dtype=float)
    return subset, remapped


def evaluate_release_gate(cells: list[dict[str, Any]], drawcache: dict[tuple[int, str], np.ndarray], numerics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    primary = [i for i, c in enumerate(cells) if c["calls"] == 16 and c["metric"] == "run_device_us"]
    all_execution = [i for i, c in enumerate(cells) if c["calls"] in (0, 1, 16) and c["metric"] == "run_device_us"]
    _require(primary and all_execution, "missing deployment-mode run-device cells")
    _require(all("selected" in cells[i] for i in all_execution), "selector decision absent")
    selected = [i for i in primary if cells[i]["selected"]]
    unselected = [i for i in primary if not cells[i]["selected"]]
    selected_all = [i for i in all_execution if cells[i]["selected"]]

    policy = _summary(primary, "guarded", cells, drawcache)
    selected_summary = _summary(selected, "guarded", cells, drawcache)
    unselected_summary = _summary(unselected, "guarded", cells, drawcache)
    off = _summary(primary, "off", cells, drawcache)
    all_mode_policy = _summary(all_execution, "guarded", cells, drawcache)
    all_mode_selected = _summary(selected_all, "guarded", cells, drawcache)

    numerical_exact = bool(numerics) and all(
        int(v["qualifications"]) == int(v["exact_full_outputs"])
        and float(v["max_abs_vs_pristine"]) == 0.0
        for v in numerics.values()
    )
    requirements = {
        "numerical_exact": numerical_exact,
        "selected_nonempty": selected_summary["count"] > 0,
        "graph16_selected_controls_resolve": selected_summary["count"] > 0 and selected_summary["controls_failed"] == 0,
        "graph16_selected_worst_point_at_least_0_99": selected_summary["count"] > 0 and selected_summary["worst_point_ratio"] >= 0.99,
        "graph16_selected_joint_worst_lcb_at_least_0_99": selected_summary["count"] > 0 and selected_summary["simultaneous_worst_CI95"][0] >= 0.99,
        "graph16_selected_aggregate_lcb_above_1": selected_summary["count"] > 0 and selected_summary["CI95"][0] > 1.0,
        "all_execution_selected_controls_resolve": all_mode_selected["count"] > 0 and all_mode_selected["controls_failed"] == 0,
        "all_execution_selected_worst_point_at_least_0_99": all_mode_selected["count"] > 0 and all_mode_selected["worst_point_ratio"] >= 0.99,
        "all_execution_selected_joint_worst_lcb_at_least_0_99": all_mode_selected["count"] > 0 and all_mode_selected["simultaneous_worst_CI95"][0] >= 0.99,
        "policy_worst_point_at_least_0_99": policy["worst_point_ratio"] >= 0.99,
        "all_execution_policy_worst_point_at_least_0_99": all_mode_policy["worst_point_ratio"] >= 0.99,
        "off_overlay_worst_point_at_least_0_99": off["worst_point_ratio"] >= 0.99,
    }
    return {
        "pass": all(requirements.values()),
        "requirements": requirements,
        "policy": policy,
        "selected": selected_summary,
        "unselected": unselected_summary,
        "all_execution_policy": all_mode_policy,
        "all_execution_selected": all_mode_selected,
        "off_overlay": off,
        "scope": "Frozen fresh cases on one GPU family; eager, Graph1 and Graph16 are separate tactic identities. Joint bootstrap is conditional on measured process/block variation, not a device-population guarantee.",
    }

