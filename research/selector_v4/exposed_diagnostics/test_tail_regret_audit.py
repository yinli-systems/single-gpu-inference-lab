import hashlib
import json

import pytest

from research.selector_v4.exposed_diagnostics.tail_regret_audit import arm_means, verified_json


def windows():
    rows = []
    arms = ["native", "oracle", "policy"]
    for block in range(24):
        order = arms[block % 3 :] + arms[: block % 3]
        for position, arm in enumerate(order + order[::-1]):
            latency = 1000.0 if arm != "oracle" else 800.0
            rows.append(
                {
                    "arm": arm,
                    "block": block,
                    "position": position,
                    "exact": True,
                    "elapsed_us": latency * 200,
                    "kernel_calls": 200,
                    "wall_us_per_call": latency,
                }
            )
    return rows


def test_native_fallback_can_have_regret_without_native_slowdown():
    means = arm_means(windows())
    assert means["native"] / means["policy"] == 1
    assert means["policy"] / min(means.values()) - 1 == pytest.approx(0.25)


@pytest.mark.parametrize("fault", ["missing", "order", "inexact", "short", "denominator"])
def test_rejects_incomplete_or_changed_windows(fault):
    rows = windows()
    if fault == "missing":
        rows.pop()
    elif fault == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif fault == "inexact":
        rows[0]["exact"] = False
    elif fault == "short":
        rows[0]["elapsed_us"] = 1
    else:
        rows[0]["kernel_calls"] = 201
    with pytest.raises(ValueError):
        arm_means(rows)


def test_immutable_input_tamper_rejected():
    raw = json.dumps({"choice": "native"}).encode()
    expected = hashlib.sha256(raw).hexdigest()
    assert verified_json(raw, expected)["choice"] == "native"
    with pytest.raises(ValueError, match="hash"):
        verified_json(raw.replace(b"native", b"resource"), expected)
