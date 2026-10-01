"""Prevent holdout leakage and incorrect artifact selection at the new boundary."""

import copy

import pytest

from research.selector_v4.exposed_diagnostics.public_path_contract import full_call_choice


def observations():
    return {
        i: {
            "paired_blocks": [[2.0, 1.7, 1.7, 2.0] for _ in range(24)],
            "certificate_accepted": True,
            "managed_tactic": 1,
            "exact": True,
            "every_resource_call_bound": True,
        }
        for i in range(3)
    }


def test_excluded_training_fold_cannot_change_choice_or_primary_artifact():
    data = observations()
    before = full_call_choice(data, 0, "source-bound-test")
    data[0] = {"corrupt_and_extremely_fast": True}
    assert full_call_choice(data, 0, "source-bound-test") == before
    assert before["choice"] == "resource" and before["primary_artifact"] == 1
    # A faster secondary process cannot replace the preregistered primary.
    data[2] = observations()[2]
    data[2]["paired_blocks"] = [[2.0, 1.4, 1.4, 2.0] for _ in range(24)]
    assert full_call_choice(data, 0, "source-bound-test")["primary_artifact"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("certificate_accepted", False),
        ("managed_tactic", -1),
        ("exact", False),
        ("every_resource_call_bound", False),
    ],
)
def test_either_training_artifact_failure_requires_native(field, value):
    data = observations()
    data[2][field] = value
    assert full_call_choice(data, 0, "source-bound-test")["choice"] == "native"


def test_duplicate_drift_and_single_bad_block_are_retained_and_reject():
    data = observations()
    data[1]["paired_blocks"] = [[2.02, 1.7, 1.7, 2.0] for _ in range(24)]
    result = full_call_choice(data, 0, "source-bound-test")
    assert result["choice"] == "native" and not result["checks"]["duplicate_controls"]
    data = observations()
    data[2]["paired_blocks"][9] = [2.0, 2.1, 2.1, 2.0]
    result = full_call_choice(data, 0, "source-bound-test")
    assert result["choice"] == "native" and result["blocks"] == 48
    assert not result["checks"]["block_floor"]


def test_incomplete_or_nonfinite_windows_cannot_become_training_evidence():
    data = observations()
    short = copy.deepcopy(data)
    short[1]["paired_blocks"].pop()
    with pytest.raises(ValueError):
        full_call_choice(short, 0, "source-bound-test")
    data[1]["paired_blocks"][0][0] = float("nan")
    with pytest.raises(ValueError):
        full_call_choice(data, 0, "source-bound-test")
