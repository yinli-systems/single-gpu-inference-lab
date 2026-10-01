"""Prospective public-path full-call training; separate stage authorization required."""

import hashlib
import json
import math

import numpy as np

CONTRACT = {
    "scope": "PUBLIC_NATIVE_RESOURCE_4_3",
    "blocks_per_process": 24,
    "independent_training_processes": 3,
    "independent_policy_processes": 3,
    "independent_pristine_processes": 3,
    "target_window_us": 384000,
    "minimum_window_us": 120000,
    "training_geomean": 1.05,
    "training_lcb95": 1.01,
    "block_floor": 0.99,
    "duplicate_control_tolerance": 0.005,
    "bootstrap_draws": 20000,
    "managed_profiling_repeat": 256,
    "native_conditioning_seconds": 30,
    "freshness_ledger_required": True,
    "canary_authority": False,
    "release_authority": False,
    "serving_authority": False,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def full_call_choice(training, held_out, identity):
    """Two other training processes decide; scoring data is never an argument.

    The primary artifact is fixed by rotation, not the fastest observed process.
    Every window, including failures and native winners, remains in the ledger.
    A resource choice additionally requires real accepted prepared certificates
    and actual v2 resource winners in both contributing training processes.
    """
    if set(training) != {0, 1, 2} or held_out not in training:
        raise ValueError("Exactly three independent training artifacts required")
    primary = (held_out + 1) % 3
    secondary = (held_out + 2) % 3
    selected = [training[primary], training[secondary]]
    rows = np.asarray([row for item in selected for row in item["paired_blocks"]], dtype=np.float64)
    if rows.shape != (48, 4) or not np.isfinite(rows).all() or np.any(rows <= 0):
        raise ValueError("Two complete 24-block training processes required")
    ratios = np.sqrt(rows[:, 0] * rows[:, 3] / (rows[:, 1] * rows[:, 2]))
    controls = [rows[:, 0] / rows[:, 3], rows[:, 1] / rows[:, 2]]
    seed = int(digest([identity, held_out, CONTRACT])[:8], 16)
    index = np.random.default_rng(seed).integers(0, len(rows), size=(20000, len(rows)))
    gain_ci = np.exp(np.quantile(np.log(ratios)[index].mean(axis=1), [0.025, 0.975])).tolist()
    control_cis = [
        np.exp(np.quantile(np.log(c)[index].mean(axis=1), [0.05, 0.95])).tolist() for c in controls
    ]
    gain = math.exp(float(np.log(ratios).mean()))
    checks = {
        "real_prepared_certificates": all(x["certificate_accepted"] for x in selected),
        "actual_managed_resource_winners": all(x["managed_tactic"] == 1 for x in selected),
        "exact_output_lse": all(x["exact"] for x in selected),
        "resource_bound_every_candidate_call": all(
            x["every_resource_call_bound"] for x in selected
        ),
        "gain_mean": gain >= 1.05,
        "gain_lcb95": gain_ci[0] >= 1.01,
        "block_floor": float(ratios.min()) >= 0.99,
        "duplicate_controls": all(lo >= 1 / 1.005 and hi <= 1.005 for lo, hi in control_cis),
    }
    result = {
        "scope": CONTRACT["scope"],
        "identity": identity,
        "held_out": held_out,
        "training_processes": [primary, secondary],
        "primary_artifact": primary,
        "choice": "resource" if all(checks.values()) else "native",
        "geomean": gain,
        "gain_ci95": gain_ci,
        "control_ci90": control_cis,
        "block_minimum": float(ratios.min()),
        "checks": checks,
        "thresholds": CONTRACT,
        "bootstrap_seed": seed,
        "blocks": len(rows),
        "training_artifact_sha256": {str(i): digest(training[i]) for i in (primary, secondary)},
    }
    result["checksum"] = digest(result)
    return result
