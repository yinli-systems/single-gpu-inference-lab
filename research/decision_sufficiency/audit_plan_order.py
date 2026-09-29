"""Retrospective decision-sufficiency audit for the plan-order experiment.

The experiment is already exposed.  This script validates its checked-in raw
archive through the original analysis loader, recomputes per-cell paired
intervals, and asks a new question: do exact representation collisions require
opposite choices between native order and heavy-first order?

No GPU work is launched and no result is treated as new confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys
import tarfile
import tempfile
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_extract_runs(archive: Path, destination: Path) -> None:
    """Extract the raw evidence archive without accepting links/path traversal."""
    with tarfile.open(archive, "r:gz") as tf:
        members = []
        root = destination.resolve()
        for member in tf.getmembers():
            if member.issym() or member.islnk():
                raise ValueError("raw evidence archive contains a link")
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError("raw evidence archive escapes destination")
            members.append(member)
        tf.extractall(destination, members=members)


def load_original_analysis(repo_root: Path):
    bench = repo_root / "benchmarks" / "results" / "plan-order-mechanism"
    analysis_dir = bench / "analysis"
    sys.path.insert(0, str(analysis_dir))
    sys.path.insert(0, str(bench))
    return importlib.import_module("analyze"), bench


def representations() -> dict[str, Callable[[Geometry], tuple]]:
    return {
        "aggregate_sums": aggregate_sums,
        "analytical_work": analytical_work,
        "vidur64": lambda g: vidur_prefill_lookup_key(g, 64),
        "marginal_moments": marginal_moments,
        "paired_geometry": paired_multiset,
    }


def audit(
    repo_root: Path,
    raw_archive: Path,
    *,
    relative_margin: float = 0.01,
) -> dict:
    original, bench = load_original_analysis(repo_root)
    cases = json.loads((bench / "cases.json").read_text())
    case_map = {
        (row["case"], row["state"]): row
        for row in cases
    }
    if len(case_map) != len(cases):
        raise ValueError("duplicate plan-order case/state")

    with tempfile.TemporaryDirectory(prefix="decision-sufficiency-plan-order-") as tmp:
        raw_root = Path(tmp)
        safe_extract_runs(raw_archive, raw_root)
        grouped, measurement_commit = original.load(raw_root, "test")

        report = {
            "kind": "retrospective exact representation / resolved plan-order decision audit",
            "measurement_commit": measurement_commit,
            "raw_archive_sha256": sha256(raw_archive),
            "relative_margin": relative_margin,
            "action_a": "native",
            "action_b": "heavy_first",
            "metric": "device_us",
            "stage": "already-exposed plan-order test evidence",
            "normalization": (
                "Per state, native point cost is normalized to 1 and heavy-first "
                "point cost to 1/(native/heavy-first ratio). Conservative action "
                "gaps use the least-favorable endpoint of each conditional 95% ratio "
                "interval. Values are dimensionless, not microseconds."
            ),
            "statistical_scope": (
                "Per-cell intervals reuse the original three-process x six-block "
                "hierarchical bootstrap. A/A equivalence reuses the original 90% "
                "margin max(3 us, 2% of native median). No familywise or hardware-"
                "population guarantee is added by this retrospective audit."
            ),
            "non_claim": (
                "All geometries and timings were exposed before this question was "
                "asked. A witness is mechanism-generating evidence only; independent "
                "confirmation requires a newly frozen collision suite."
            ),
            "GPUs": {},
        }

        for gpu, runs in sorted(grouped.items()):
            env = runs[0]["env"]
            rows_by_rep = {name: [] for name in representations()}
            cells = []

            for case in cases:
                pair = (case["case"], case["state"])
                g = Geometry(tuple(case["query"]), tuple(case["cached"]))
                for dtype in env["dtype"]:
                    for split in ("auto", "unsplit"):
                        cell = (pair, dtype, split)
                        for mode in ("eager", "graph"):
                            native = [
                                run["index"][
                                    (pair, dtype, split, block, "identity", mode)
                                ]["device_us"]
                                for run in runs
                                for block in range(run["env"]["blocks"])
                            ]
                            native_median = statistics.median(native)

                            aa_grid = original.paired_grid(
                                runs,
                                [cell],
                                "identity_repeat",
                                mode,
                                difference=True,
                            )
                            aa = original.interval(aa_grid, confidence=0.90)
                            epsilon = max(3.0, 0.02 * native_median)
                            aa_equivalent = (
                                aa["CI"][0] >= -epsilon
                                and aa["CI"][1] <= epsilon
                            )

                            ratio_grid = original.paired_grid(
                                runs, [cell], "heavy_first", mode
                            )
                            ratio = original.interval(
                                ratio_grid, confidence=0.95, log=True
                            )

                            cell_record = {
                                "case": case["case"],
                                "state": case["state"],
                                "holdout": bool(case["holdout"]),
                                "dtype": dtype,
                                "split": split,
                                "mode": mode,
                                "native_median_us": native_median,
                                "ratio": ratio["ratio"],
                                "CI95": ratio["CI"],
                                "AA90": aa["CI"],
                                "AA_epsilon_us": epsilon,
                                "AA_equivalent": aa_equivalent,
                            }
                            cells.append(cell_record)

                            for name, feature_fn in representations().items():
                                key = feature_with_context(
                                    feature_fn(g),
                                    dtype=dtype,
                                    split=split,
                                    mode=mode,
                                )
                                rows_by_rep[name].append(
                                    ResolvedDecision(
                                        state_id="|".join(
                                            [
                                                case["case"],
                                                case["state"],
                                                dtype,
                                                split,
                                                mode,
                                            ]
                                        ),
                                        feature_key=key,
                                        action_a="native",
                                        action_b="heavy_first",
                                        ratio_a_over_b=ratio["ratio"],
                                        ci_low=ratio["CI"][0],
                                        ci_high=ratio["CI"][1],
                                        controls_resolve=aa_equivalent,
                                        context=tuple(
                                            sorted(
                                                {
                                                    "case": case["case"],
                                                    "state": case["state"],
                                                    "holdout": bool(case["holdout"]),
                                                    "dtype": dtype,
                                                    "split": split,
                                                    "mode": mode,
                                                }.items()
                                            )
                                        ),
                                    )
                                )

            rep_results = {}
            for name, rows in rows_by_rep.items():
                witnesses = find_opposite_action_collisions(
                    rows, relative_margin=relative_margin
                )
                serialized = []
                for w in witnesses:
                    left_context = dict(w.left.context)
                    right_context = dict(w.right.context)
                    serialized.append(
                        {
                            "feature_key": list(w.feature_key),
                            "left": {
                                "state_id": w.left.state_id,
                                "preference": w.left.preference(relative_margin),
                                "ratio": w.left.ratio_a_over_b,
                                "CI95": [w.left.ci_low, w.left.ci_high],
                                "context": left_context,
                            },
                            "right": {
                                "state_id": w.right.state_id,
                                "preference": w.right.preference(relative_margin),
                                "ratio": w.right.ratio_a_over_b,
                                "CI95": [w.right.ci_low, w.right.ci_high],
                                "context": right_context,
                            },
                            "evidence_pair": (
                                "holdout-only"
                                if left_context["holdout"] and right_context["holdout"]
                                else "discovery-only"
                                if not left_context["holdout"]
                                and not right_context["holdout"]
                                else "mixed-exposed"
                            ),
                            "normalized_minimax_regret_point": (
                                w.normalized_minimax_regret_point
                            ),
                            "conservative_normalized_minimax_regret": (
                                w.conservative_normalized_minimax_regret
                            ),
                        }
                    )

                resolved = {"native": 0, "heavy_first": 0, "unresolved": 0}
                for row in rows:
                    pref = row.preference(relative_margin)
                    resolved[pref if pref is not None else "unresolved"] += 1
                rep_results[name] = {
                    "rows": len(rows),
                    "resolved_preferences": resolved,
                    "opposite_action_collision_pairs": len(serialized),
                    "discovery_only_pairs": sum(
                        x["evidence_pair"] == "discovery-only" for x in serialized
                    ),
                    "holdout_only_pairs": sum(
                        x["evidence_pair"] == "holdout-only" for x in serialized
                    ),
                    "max_conservative_normalized_minimax_regret": max(
                        (
                            x["conservative_normalized_minimax_regret"]
                            for x in serialized
                        ),
                        default=0.0,
                    ),
                    "witnesses": serialized,
                }

            report["GPUs"][gpu] = {
                "processes": len(runs),
                "cells": cells,
                "representations": rep_results,
            }

    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--raw-archive", type=Path)
    p.add_argument("--relative-margin", type=float, default=0.01)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    repo_root = a.repo_root.resolve()
    raw_archive = (
        a.raw_archive.resolve()
        if a.raw_archive
        else repo_root
        / "benchmarks"
        / "results"
        / "plan-order-mechanism"
        / "raw-evidence.tar.gz"
    )
    if a.out.exists():
        raise FileExistsError("preserve prior audit")
    result = audit(
        repo_root,
        raw_archive,
        relative_margin=a.relative_margin,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    compact = {}
    for gpu, g in result["GPUs"].items():
        compact[gpu] = {
            name: {
                "opposite_action_collision_pairs": row[
                    "opposite_action_collision_pairs"
                ],
                "discovery_only_pairs": row["discovery_only_pairs"],
                "holdout_only_pairs": row["holdout_only_pairs"],
                "max_conservative_normalized_minimax_regret": row[
                    "max_conservative_normalized_minimax_regret"
                ],
            }
            for name, row in g["representations"].items()
        }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
