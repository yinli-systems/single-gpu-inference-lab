"""Separate full-candidate completeness from available-arm regret.

Retrospective diagnosis only. Never authorizes a tactic, certificate or job.
Unknown Resource latency stays unknown; a Native winner cannot fill that gap.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def evaluate(rows, *, p99_target=None):
    if not rows:
        raise ValueError("Nonempty complete record population required")
    if p99_target is not None and (
        type(p99_target) not in (int, float)
        or not math.isfinite(p99_target)
        or p99_target < 0
    ):
        raise ValueError("Explicit finite nonnegative prospective target required")
    evaluated, identities = [], set()
    for row in rows:
        identity = tuple(row["identity"])
        if identity in identities:
            raise ValueError("Duplicate record identity")
        identities.add(identity)
        candidates = row["candidate_latencies_us"]
        if set(candidates) - {"native", "resource"} or not positive(
            candidates.get("native")
        ):
            raise ValueError("Named candidate universe with measured Native required")
        if any(not positive(v) for v in candidates.values()) or not positive(
            row["chosen_latency_us"]
        ):
            raise ValueError("Invalid measured latency; missing is not zero")
        complete = set(candidates) == {"native", "resource"}
        # Policy is an independently timed instance of the selected path. Keep
        # it in the available oracle, matching the original excess definition.
        regret = (
            row["chosen_latency_us"]
            / min(*candidates.values(), row["chosen_latency_us"])
            - 1
        )
        evaluated.append(
            {
                "identity": list(identity),
                "missing_candidates": sorted({"native", "resource"} - set(candidates)),
                "available_arm_regret": regret,
                "full_candidate_regret": regret if complete else None,
            }
        )
    complete_count = sum(not r["missing_candidates"] for r in evaluated)
    complete = complete_count == len(evaluated)
    p99 = (
        float(np.quantile([r["full_candidate_regret"] for r in evaluated], 0.99))
        if complete
        else None
    )
    return {
        "state": "COMPLETE_CANDIDATE_MEASUREMENTS"
        if complete
        else "INSUFFICIENT_CANDIDATE_COVERAGE",
        "population_records": len(evaluated),
        "complete_candidate_records": complete_count,
        "missing_candidate_records": len(evaluated) - complete_count,
        "coverage": complete_count / len(evaluated),
        "available_arm_p99": float(
            np.quantile([r["available_arm_regret"] for r in evaluated], 0.99)
        ),
        "full_candidate_p99": p99,
        "prospective_p99_target": p99_target,
        "target_satisfied": complete and p99_target is not None and p99 <= p99_target,
        "records": evaluated,
        "qualification_authority": False,
        "candidate_universe": ["native", "resource"],
        "scope": "Observed population only; no unseen-geometry or production optimality claim",
    }


def from_audit(path, expected_sha256):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Bound tail audit changed")
    audit = json.loads(raw)
    result = {}
    for gpu, card in audit["cards"].items():
        rows = []
        for r in card["records"]:
            means = r["arm_geomean_us"]
            candidates = {"native": means["native"]}
            if r["oracle_includes_resource"] is True:
                candidates["resource"] = means["oracle"]
            rows.append(
                {
                    "identity": [r["case"], r["key"], r["rep"]],
                    "candidate_latencies_us": candidates,
                    "chosen_latency_us": means["policy"],
                }
            )
        if len(rows) != card["complete_records"]:
            raise ValueError("All original records required")
        result[gpu] = evaluate(rows)
    return {
        "source_audit_sha256": expected_sha256,
        "cards": result,
        "dual_full_candidate_evidence_complete": all(
            c["missing_candidate_records"] == 0 for c in result.values()
        ),
        "default_promotion": False,
        "serving_promotion": False,
        "gpu_jobs_dispatched": 0,
        "frozen_thresholds_changed": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("audit", type=Path)
    p.add_argument("sha256")
    p.add_argument("output", type=Path)
    args = p.parse_args()
    result = from_audit(args.audit, args.sha256)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                g: {k: v for k, v in c.items() if k != "records"}
                for g, c in result["cards"].items()
            },
            indent=2,
        )
    )
