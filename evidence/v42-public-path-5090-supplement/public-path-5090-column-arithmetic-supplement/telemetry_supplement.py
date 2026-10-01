"""Actual NVIDIA clock columns; preserve the original whole-job health gate."""

import csv
import math

import numpy as np


def read_clock_csv(path, uuid=None):
    active = []
    expected_uuid = None if uuid is None else (uuid if uuid.startswith("GPU-") else "GPU-" + uuid)
    with path.open() as stream:
        reader = csv.DictReader(stream)
        names = {name: name.split("[")[0].strip() for name in reader.fieldnames or []}
        clocks = [k for k, v in names.items() if v in ("clocks.sm", "clocks.current.sm")]
        utilization = [k for k, v in names.items() if v == "utilization.gpu"]
        uuids = [k for k, v in names.items() if v == "uuid"]
        if len(clocks) != 1 or len(utilization) != 1 or len(uuids) != 1:
            return {
                "stable": False,
                "reason": "Missing or ambiguous actual NVIDIA columns",
                "active_samples": 0,
            }
        for row in reader:
            if expected_uuid is not None and row[uuids[0]].strip() != expected_uuid:
                return {
                    "stable": False,
                    "reason": "Telemetry GPU UUID differs",
                    "active_samples": len(active),
                }
            try:
                used = float(row[utilization[0]].split()[0])
                if not math.isfinite(used) or not 0 <= used <= 100:
                    raise ValueError("Invalid GPU utilization")
                if used < 90:
                    continue
                value = float(row[clocks[0]].split()[0])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("Invalid active GPU clock")
                active.append(value)
            except (ValueError, AttributeError, TypeError, IndexError):
                return {
                    "stable": False,
                    "reason": "Invalid actual active telemetry",
                    "active_samples": len(active),
                }
    if len(active) < 10:
        return {
            "stable": False,
            "reason": "insufficient active samples",
            "active_samples": len(active),
        }
    lo, hi = np.quantile(active, [0.05, 0.95])
    return {
        "stable": bool(lo > 0 and hi / lo <= 1.05),
        "active_samples": len(active),
        "p05_mhz": float(lo),
        "p95_mhz": float(hi),
    }
