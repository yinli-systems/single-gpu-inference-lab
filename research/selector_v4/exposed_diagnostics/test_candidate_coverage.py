import math

import pytest

from research.selector_v4.exposed_diagnostics.candidate_coverage import evaluate


def row(candidates, chosen=100):
    return {
        "identity": [0, "cell", 0],
        "candidate_latencies_us": candidates,
        "chosen_latency_us": chosen,
    }


def test_zero_available_regret_never_fills_missing_resource():
    r = evaluate([row({"native": 100})], p99_target=0.01)
    assert r["available_arm_p99"] == 0
    assert r["full_candidate_p99"] is None
    assert r["state"] == "INSUFFICIENT_CANDIDATE_COVERAGE"
    assert r["target_satisfied"] is False


def test_complete_but_high_regret_cannot_pass_target():
    r = evaluate([row({"native": 100, "resource": 80})], p99_target=0.01)
    assert r["full_candidate_p99"] == 0.25
    assert not r["target_satisfied"]


def test_no_implicit_target_or_global_optimality():
    r = evaluate([row({"native": 100, "resource": 80}, chosen=80)])
    assert r["full_candidate_p99"] == 0
    assert not r["target_satisfied"]
    assert not r["qualification_authority"]


@pytest.mark.parametrize("bad", [0, -1, math.nan, math.inf, True, None])
def test_missing_candidate_must_not_be_encoded_as_invalid_latency(bad):
    with pytest.raises(ValueError):
        evaluate([row({"native": 100, "resource": bad})])


def test_duplicate_windows_cannot_inflate_coverage():
    r = row({"native": 100, "resource": 80})
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate([r, r])
