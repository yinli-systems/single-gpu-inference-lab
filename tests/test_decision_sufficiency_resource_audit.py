import importlib.util
from pathlib import Path


def _load_audit_module():
    path = (
        Path(__file__).parents[1]
        / "research"
        / "decision_sufficiency"
        / "audit_resource_summary.py"
    )
    spec = importlib.util.spec_from_file_location("decision_sufficiency_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _cell(layout, ratio, ci):
    return {
        "case": "same-geometry",
        "dtype": "float16",
        "layout": layout,
        "split": "unsplit",
        "calls": 16,
        "metric": "run_device_us",
        "comparisons": {
            "cap": {
                "ratio": ratio,
                "CI95": list(ci),
                "controls_resolve": True,
            }
        },
    }


def test_resource_audit_separates_representation_from_execution_context():
    audit = _load_audit_module()
    manifest = {
        "cases": [
            {
                "id": "same-geometry",
                "q": [64, 128],
                "cached": [512, 1024],
            }
        ]
    }
    summary = {
        "complete": True,
        "gpu": "synthetic",
        "stage": "unit-test",
        "cells": [
            # The same request geometry prefers opposite actions in two layouts.
            _cell("ragged", 1.20, (1.15, 1.25)),
            _cell("paged", 0.80, (0.76, 0.84)),
        ],
    }
    result = audit.audit(manifest, summary, relative_margin=0.01)

    geometry_only = result["representations"]["paired_geometry"]
    assert geometry_only["opposite_action_collision_pairs"] == 1
    assert geometry_only["max_normalized_minimax_regret_point"] > 0
    assert geometry_only["max_conservative_normalized_minimax_regret"] > 0

    with_layout = result["representations"]["paired_plus_dtype_layout"]
    assert with_layout["opposite_action_collision_pairs"] == 0


def test_resource_audit_refuses_unresolved_controls():
    audit = _load_audit_module()
    manifest = {
        "cases": [{"id": "same-geometry", "q": [64], "cached": [512]}]
    }
    bad = _cell("ragged", 1.20, (1.15, 1.25))
    bad["comparisons"]["cap"]["controls_resolve"] = False
    summary = {
        "complete": True,
        "gpu": "synthetic",
        "stage": "unit-test",
        "cells": [bad],
    }
    result = audit.audit(manifest, summary)
    assert all(
        row["opposite_action_collision_pairs"] == 0
        for row in result["representations"].values()
    )
