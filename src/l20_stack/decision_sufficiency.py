"""Decision-sufficiency primitives for execution-cost research.

This module does not predict latency.  It audits whether a chosen observation
representation can distinguish measured states that require different actions.
All comparisons are exact at the representation layer; approximate equality is
an explicitly separate research question.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Optional, Sequence


FeatureKey = tuple[object, ...]


@dataclass(frozen=True)
class Geometry:
    """Request-level chunk/cached-KV geometry."""

    query: tuple[int, ...]
    cached: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.query or len(self.query) != len(self.cached):
            raise ValueError("query/cached must be non-empty and aligned")
        if any(type(x) is not int or x < 0 for x in (*self.query, *self.cached)):
            raise ValueError("geometry values must be non-negative integers")
        if any(q <= 0 for q in self.query):
            raise ValueError("query chunks must be positive")

    @property
    def n(self) -> int:
        return len(self.query)

    @property
    def total_query(self) -> int:
        return sum(self.query)

    @property
    def total_cached(self) -> int:
        return sum(self.cached)

    @property
    def attention_work(self) -> int:
        return sum(
            q * k + q * (q + 1) // 2
            for q, k in zip(self.query, self.cached)
        )


def aggregate_sums(g: Geometry) -> FeatureKey:
    return (g.n, g.total_query, g.total_cached)


def analytical_work(g: Geometry) -> FeatureKey:
    return (g.attention_work,)


def marginal_moments(g: Geometry) -> FeatureKey:
    total = tuple(q + k for q, k in zip(g.query, g.cached))
    return (
        g.n,
        g.total_query,
        g.total_cached,
        sum(q * q for q in g.query),
        sum(k * k for k in g.cached),
        sum(x * x for x in total),
    )


def paired_multiset(g: Geometry) -> FeatureKey:
    return tuple(sorted(zip(g.query, g.cached)))


def ordered_pairs(g: Geometry) -> FeatureKey:
    return tuple(zip(g.query, g.cached))


def vidur_prefill_lookup_key(g: Geometry, kv_granularity: int = 64) -> FeatureKey:
    """Pinned-Vidur-style prefill lookup coordinate.

    Mirrors the audited shape of Vidur's prefill key:
    quantize each cached-KV size, sum them, and use the rounded L2 norm of
    query-chunk sizes squared.  This is a feature-construction audit, not a
    reproduction of Vidur's trained execution-time predictor.
    """
    if type(kv_granularity) is not int or kv_granularity <= 0:
        raise ValueError("kv_granularity must be a positive integer")
    quantized_kv = sum(
        ((k + kv_granularity - 1) // kv_granularity) * kv_granularity
        for k in g.cached
    )
    chunk_coord = round(math.sqrt(sum(q * q for q in g.query))) ** 2
    return (quantized_kv, chunk_coord)


@dataclass(frozen=True)
class ResolvedDecision:
    """One measured state with a predeclared, statistically resolved preference."""

    state_id: str
    feature_key: FeatureKey
    action_a: str
    action_b: str
    ratio_a_over_b: float
    ci_low: float
    ci_high: float
    controls_resolve: bool
    context: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        vals = (self.ratio_a_over_b, self.ci_low, self.ci_high)
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in vals):
            raise ValueError("ratio and interval must be finite positive numbers")
        if self.ci_low > self.ci_high:
            raise ValueError("invalid interval")
        if self.action_a == self.action_b:
            raise ValueError("actions must differ")

    def preference(self, relative_margin: float = 0.01) -> Optional[str]:
        """Return the faster action only when controls and the CI resolve the margin."""
        if not 0 <= relative_margin < 1:
            raise ValueError("relative_margin must be in [0,1)")
        if not self.controls_resolve:
            return None
        if self.ci_low > 1.0 + relative_margin:
            return self.action_b
        if self.ci_high < 1.0 / (1.0 + relative_margin):
            return self.action_a
        return None

    def normalized_costs(self) -> Mapping[str, float]:
        """Return costs normalized to action_a=1.

        If r = cost(a)/cost(b), then cost(b)=1/r.  These normalized costs are
        suitable for dimensionless regret audits only; they are not microseconds.
        """
        return {self.action_a: 1.0, self.action_b: 1.0 / self.ratio_a_over_b}


@dataclass(frozen=True)
class CollisionWitness:
    feature_key: FeatureKey
    left: ResolvedDecision
    right: ResolvedDecision
    normalized_minimax_regret_lower_bound: float

    def __post_init__(self) -> None:
        if self.left.feature_key != self.right.feature_key:
            raise ValueError("not a representation collision")
        if self.normalized_minimax_regret_lower_bound <= 0:
            raise ValueError("regret lower bound must be positive")


def binary_minimax_regret_lower_bound(
    left: Mapping[str, float],
    right: Mapping[str, float],
    action_a: str,
    action_b: str,
) -> float:
    """Minimax regret for two indistinguishable states with opposite best actions.

    The policy may randomize between action_a and action_b but observes the same
    representation in both states.  Costs are arbitrary positive values in a
    common unit (or a documented normalized unit).
    """
    for costs in (left, right):
        if set((action_a, action_b)) - set(costs):
            raise ValueError("missing action cost")
        if any(not math.isfinite(float(costs[a])) or float(costs[a]) <= 0 for a in (action_a, action_b)):
            raise ValueError("action costs must be finite and positive")

    left_best = min((action_a, action_b), key=left.__getitem__)
    right_best = min((action_a, action_b), key=right.__getitem__)
    if left_best == right_best:
        raise ValueError("states do not require opposite actions")

    if left_best == action_a:
        delta_left = float(left[action_b] - left[action_a])
        delta_right = float(right[action_a] - right[action_b])
    else:
        delta_left = float(left[action_a] - left[action_b])
        delta_right = float(right[action_b] - right[action_a])

    if delta_left <= 0 or delta_right <= 0:
        raise ValueError("non-positive action gap")
    return delta_left * delta_right / (delta_left + delta_right)


def find_opposite_action_collisions(
    rows: Iterable[ResolvedDecision],
    relative_margin: float = 0.01,
) -> list[CollisionWitness]:
    """Find exact feature collisions whose resolved best actions are opposite."""
    groups: dict[FeatureKey, list[ResolvedDecision]] = {}
    for row in rows:
        if row.preference(relative_margin) is None:
            continue
        groups.setdefault(row.feature_key, []).append(row)

    out: list[CollisionWitness] = []
    for key, members in groups.items():
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                if (left.action_a, left.action_b) != (right.action_a, right.action_b):
                    continue
                if left.preference(relative_margin) == right.preference(relative_margin):
                    continue
                lb = binary_minimax_regret_lower_bound(
                    left.normalized_costs(),
                    right.normalized_costs(),
                    left.action_a,
                    left.action_b,
                )
                out.append(CollisionWitness(key, left, right, lb))
    return sorted(
        out,
        key=lambda x: (
            -x.normalized_minimax_regret_lower_bound,
            x.left.state_id,
            x.right.state_id,
        ),
    )


def feature_with_context(
    base: Sequence[object],
    **context: object,
) -> FeatureKey:
    """Append explicitly named execution context to a representation."""
    return tuple(base) + tuple(sorted(context.items()))
