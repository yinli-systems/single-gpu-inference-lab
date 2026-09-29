"""Audit a resource-policy summary for exact opposite-action feature collisions.

This consumes already-produced JSON.  It does not run GPU work, refit a policy,
or mutate a frozen campaign.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from l20_stack.decision_sufficiency import (
    Geometry,
    ResolvedDecision,
    aggregate_sums,
    analytical_work,
    feature_with_context,
    find_opposite_action_collisions,
    marginal_moments,
    paired_multiset,
    vidur_prefill_lookup_key,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def geometry_index(manifest: dict) -> dict[str, Geometry]:
    out = {}
    for case in manifest["cases"]:
        case_id = case["id"]
        require(case_id not in out, "duplicate case id")
        out[case_id] = Geometry(tuple(case["q"]), tuple(case["cached"]))
    return out


def representation_functions() -> dict[str, Callable[[Geometry, dict], tuple]]:
    return {
        "aggregate_sums": lambda g, c: aggregate_sums(g),
        "analytical_work": lambda g, c: analytical_work(g),
        "vidur64": lambda g, c: vidur_prefill_lookup_key(g, 64),
        "marginal_moments": lambda g, c: marginal_moments(g),
        "paired_geometry": lambda g, c: paired_multiset(g),
        "paired_plus_dtype": lambda g, c: feature_with_context(
            paired_multiset(g), dtype=c["dtype"]
        ),
        "paired_plus_dtype_layout": lambda g, c: feature_with_context(
            paired_multiset(g), dtype=c["dtype"], layout=c["layout"]
        ),
        "paired_plus_execution_context": lambda g, c: feature_with_context(
            paired_multiset(g),
            dtype=c["dtype"],
            layout=c["layout"],
            split=c["split"],
        ),
    }


def audit(
    manifest: dict,
    summary: dict,
    *,
    action: str = "cap",
    calls: int = 16,
    metric: str = "run_device_us",
    relative_margin: float = 0.01,
) -> dict:
    cases = geometry_index(manifest)
    require(summary.get("complete") is True, "summary is not complete")
    require(action != "pristine", "candidate action must differ from pristine")

    selected = []
    for cell in summary.get("cells", []):
        if cell["calls"] != calls or cell["metric"] != metric:
            continue
        require(cell["case"] in cases, "summary case missing from manifest")
        comparison = cell["comparisons"][action]
        selected.append((cell, comparison))

    require(selected, "no matching cells")
    reps = {}
    for name, feature_fn in representation_functions().items():
        rows = []
        resolved = {"pristine": 0, action: 0, "unresolved": 0}
        for cell, comparison in selected:
            g = cases[cell["case"]]
            state_id = "|".join(
                [
                    cell["case"],
                    cell["dtype"],
                    cell["layout"],
                    cell["split"],
                    "calls=" + str(calls),
                    metric,
                ]
            )
            row = ResolvedDecision(
                state_id=state_id,
                feature_key=feature_fn(g, cell),
                action_a="pristine",
                action_b=action,
                ratio_a_over_b=float(comparison["ratio"]),
                ci_low=float(comparison["CI95"][0]),
                ci_high=float(comparison["CI95"][1]),
                controls_resolve=bool(comparison["controls_resolve"]),
                context=tuple(
                    sorted(
                        {
                            "case": cell["case"],
                            "dtype": cell["dtype"],
                            "layout": cell["layout"],
                            "split": cell["split"],
                        }.items()
                    )
                ),
            )
            pref = row.preference(relative_margin)
            resolved[pref if pref is not None else "unresolved"] += 1
            rows.append(row)

        witnesses = find_opposite_action_collisions(rows, relative_margin)
        serialized = [
            {
                "feature_key": list(w.feature_key),
                "left": {
                    "state_id": w.left.state_id,
                    "preference": w.left.preference(relative_margin),
                    "ratio": w.left.ratio_a_over_b,
                    "CI95": [w.left.ci_low, w.left.ci_high],
                    "context": dict(w.left.context),
                },
                "right": {
                    "state_id": w.right.state_id,
                    "preference": w.right.preference(relative_margin),
                    "ratio": w.right.ratio_a_over_b,
                    "CI95": [w.right.ci_low, w.right.ci_high],
                    "context": dict(w.right.context),
                },
                "normalized_minimax_regret_lower_bound": (
                    w.normalized_minimax_regret_lower_bound
                ),
            }
            for w in witnesses
        ]
        reps[name] = {
            "rows": len(rows),
            "resolved_preferences": resolved,
            "opposite_action_collision_pairs": len(serialized),
            "max_normalized_minimax_regret_lower_bound": (
                max(
                    (
                        x["normalized_minimax_regret_lower_bound"]
                        for x in serialized
                    ),
                    default=0.0,
                )
            ),
            "witnesses": serialized,
        }

    return {
        "kind": "exact representation / resolved decision audit",
        "gpu": summary.get("gpu"),
        "stage": summary.get("stage"),
        "candidate_action": action,
        "calls": calls,
        "metric": metric,
        "relative_margin": relative_margin,
        "normalization": (
            "Per state, pristine cost is normalized to 1 and candidate cost "
            "to 1/(pristine/candidate ratio). Regret bounds are dimensionless, "
            "not microseconds."
        ),
        "qualification": (
            "A decision is resolved only when its A/A controls resolve the "
            "campaign threshold and the entire reported 95% interval clears "
            "the requested action margin."
        ),
        "non_claim": (
            "A collision proves insufficiency only for the named representation, "
            "action set, metric, frozen evidence and decision rule. Absence of a "
            "collision is not proof of sufficiency."
        ),
        "representations": reps,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--action", default="cap")
    p.add_argument("--calls", type=int, default=16)
    p.add_argument("--metric", default="run_device_us")
    p.add_argument("--relative-margin", type=float, default=0.01)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError("preserve prior audit")
    result = audit(
        json.loads(a.manifest.read_text()),
        json.loads(a.summary.read_text()),
        action=a.action,
        calls=a.calls,
        metric=a.metric,
        relative_margin=a.relative_margin,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                name: {
                    "resolved_preferences": row["resolved_preferences"],
                    "opposite_action_collision_pairs": row[
                        "opposite_action_collision_pairs"
                    ],
                    "max_normalized_minimax_regret_lower_bound": row[
                        "max_normalized_minimax_regret_lower_bound"
                    ],
                }
                for name, row in result["representations"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
