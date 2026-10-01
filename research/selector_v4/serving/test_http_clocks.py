"""Warmup clocks cannot hide insufficient, invalid or unstable scored telemetry."""

import json

import pytest

from research.selector_v4.serving.http_analysis import clock_evidence


def records(tmp_path, *, count=10, unstable=False, fault=False):
    rows = []
    for index in range(30):
        row = {
            "unix_started": index,
            "unix_finished": index + 0.01,
            "exit_code": 0,
            "uuid": "GPU-actual",
            "utilization": 100,
            "sm_mhz": 2500,
        }
        if index < 20:
            row["sm_mhz"] = 2400
        elif unstable and index % 2:
            row["sm_mhz"] = 2200
        if fault and index == 22:
            row["error"] = "actual probe failed"
        rows.append(row)
    path = tmp_path / "telemetry.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path, {"scored": {"unix_started": 20, "elapsed": count}}


def test_only_actual_scored_samples_enter_clock_gate(tmp_path):
    path, blocks = records(tmp_path)
    assert clock_evidence(path, blocks, "GPU-actual") == {
        "active_samples": 10,
        "p05_mhz": 2500,
        "p95_mhz": 2500,
    }


@pytest.mark.parametrize("fault", ["too-few", "unstable", "probe-error", "wrong-uuid"])
def test_clock_failure_cannot_be_hidden_by_good_warmup_samples(tmp_path, fault):
    path, blocks = records(
        tmp_path,
        count=5 if fault == "too-few" else 10,
        unstable=fault == "unstable",
        fault=fault == "probe-error",
    )
    with pytest.raises(ValueError):
        clock_evidence(path, blocks, "GPU-other" if fault == "wrong-uuid" else "GPU-actual")
