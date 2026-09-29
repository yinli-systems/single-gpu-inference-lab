import math

import pytest

from l20_stack.decision_sufficiency import (
    Geometry,
    ResolvedDecision,
    aggregate_sums,
    analytical_work,
    binary_minimax_regret_lower_bound,
    find_opposite_action_collisions,
    marginal_moments,
    ordered_pairs,
    paired_multiset,
    vidur_prefill_lookup_key,
)


def test_pairing_swap_aliases_aggregate_but_changes_work():
    a = Geometry((16, 32), (64, 128))
    b = Geometry((16, 32), (128, 64))
    assert aggregate_sums(a) == aggregate_sums(b)
    assert vidur_prefill_lookup_key(a) == vidur_prefill_lookup_key(b)
    assert paired_multiset(a) != paired_multiset(b)
    assert ordered_pairs(a) != ordered_pairs(b)
    assert analytical_work(a) != analytical_work(b)


def test_augmented_moments_reconstruct_joint_work_information():
    a = Geometry((16, 32), (64, 128))
    b = Geometry((16, 32), (128, 64))
    assert marginal_moments(a) != marginal_moments(b)
    assert a.attention_work != b.attention_work


def test_binary_minimax_regret_exact_two_state_result():
    # Left requires native; right requires cap.
    left = {"native": 1.0, "cap": 2.0}
    right = {"native": 3.0, "cap": 1.0}
    # Equalizing (1-p)*1 and p*2 gives 2/3.
    got = binary_minimax_regret_lower_bound(left, right, "native", "cap")
    assert math.isclose(got, 2.0 / 3.0, rel_tol=1e-12)


def test_same_best_action_is_not_an_identifiability_witness():
    with pytest.raises(ValueError, match="opposite actions"):
        binary_minimax_regret_lower_bound(
            {"native": 1.0, "cap": 2.0},
            {"native": 1.5, "cap": 2.5},
            "native",
            "cap",
        )


def _row(state_id, key, ratio, lo, hi, controls=True):
    return ResolvedDecision(
        state_id=state_id,
        feature_key=key,
        action_a="native",
        action_b="cap",
        ratio_a_over_b=ratio,
        ci_low=lo,
        ci_high=hi,
        controls_resolve=controls,
    )


def test_preference_requires_control_and_full_interval_margin():
    assert _row("cap", ("same",), 1.10, 1.05, 1.15).preference(0.01) == "cap"
    assert _row("native", ("same",), 0.90, 0.85, 0.95).preference(0.01) == "native"
    assert _row("crosses", ("same",), 1.02, 0.99, 1.04).preference(0.01) is None
    assert _row("aa-fail", ("same",), 1.10, 1.05, 1.15, False).preference(0.01) is None


def test_exact_collision_with_opposite_actions_produces_positive_lower_bound():
    rows = [
        _row("left", ("geometry-only", 7), 0.90, 0.88, 0.92),
        _row("right", ("geometry-only", 7), 1.20, 1.16, 1.24),
        _row("different-key", ("geometry-only", 8), 1.30, 1.25, 1.35),
    ]
    witnesses = find_opposite_action_collisions(rows, relative_margin=0.01)
    assert len(witnesses) == 1
    w = witnesses[0]
    assert {w.left.state_id, w.right.state_id} == {"left", "right"}
    assert w.normalized_minimax_regret_point > 0
    assert w.conservative_normalized_minimax_regret > 0
    assert (
        w.conservative_normalized_minimax_regret
        <= w.normalized_minimax_regret_point
    )


def test_unresolved_or_different_representation_is_not_promoted():
    rows = [
        _row("left", ("a",), 0.90, 0.88, 0.92),
        _row("different-feature", ("b",), 1.20, 1.16, 1.24),
        _row("unresolved", ("a",), 1.20, 0.99, 1.24),
    ]
    assert find_opposite_action_collisions(rows) == []
