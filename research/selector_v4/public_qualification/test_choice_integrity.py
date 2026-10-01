"""Exact checksum/training/winner/gates; tiny roundoff cannot change a decision."""

import copy
import math

import pytest

from research.selector_v4.public_qualification.choice_integrity import verify_choice_record
from research.selector_v4.public_qualification.contract import digest, full_call_choice


def record():
    training = {
        i: {
            "paired_blocks": [[110.0, 100.0, 100.0, 110.0]] * 24,
            "certificate_accepted": True,
            "managed_tactic": 1,
            "exact": True,
            "every_resource_call_bound": True,
        }
        for i in range(3)
    }
    return full_call_choice(training, 0, "synthetic-CPU-only")


def seal(value):
    value["checksum"] = digest({k: v for k, v in value.items() if k != "checksum"})


def test_adjacent_float_roundoff_preserves_original_checksum_and_strict_decision():
    expected = record()
    actual = copy.deepcopy(expected)
    actual["gain_ci95"][0] = math.nextafter(actual["gain_ci95"][0], math.inf)
    seal(actual)
    result = verify_choice_record(actual, expected)
    assert result["roundoff_fields"] == ["gain_ci95"]
    assert actual["choice"] == expected["choice"] == "resource"


def test_one_ulp_at_floor_cannot_change_the_strict_gate_or_promote_a_winner():
    expected = record()
    expected["block_minimum"] = 0.99
    seal(expected)
    actual = copy.deepcopy(expected)
    actual["block_minimum"] = math.nextafter(0.99, 0)
    actual["checks"]["block_floor"] = False
    actual["choice"] = "native"
    seal(actual)
    with pytest.raises(ValueError, match="provenance/decision"):
        verify_choice_record(actual, expected)


@pytest.mark.parametrize(
    "fault", ["checksum", "source", "winner", "threshold", "check", "large-roundoff", "nan"]
)
def test_rehashed_changed_source_gate_or_winner_cannot_use_roundoff_exception(fault):
    expected = record()
    actual = copy.deepcopy(expected)
    if fault == "checksum":
        actual["checksum"] = "forged"
    elif fault == "source":
        actual["training_artifact_sha256"]["1"] = "different-input"
    elif fault == "winner":
        actual["choice"] = "native"
    elif fault == "threshold":
        actual["thresholds"]["block_floor"] = 0.98
    elif fault == "check":
        actual["checks"]["block_floor"] = False
    elif fault == "large-roundoff":
        for _ in range(5):
            actual["gain_ci95"][0] = math.nextafter(actual["gain_ci95"][0], math.inf)
    elif fault == "nan":
        actual["gain_ci95"][0] = math.nan
    if fault not in ("checksum", "nan"):
        seal(actual)
    with pytest.raises(ValueError):
        verify_choice_record(actual, expected)
