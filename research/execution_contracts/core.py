"""Exact representation witnesses and cost accounting, not a live dispatcher.

A witness is conditional on caller-supplied cost bounds. This module does not
turn nominal confidence intervals into simultaneous/statistical guarantees.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def number(value: Any, name: str, *, positive: bool = False) -> float:
    require(type(value) in (int, float), f"{name}: real number required")
    result = float(value)
    require(math.isfinite(result) and (result > 0 if positive else result >= 0),
            f"{name}: invalid finite nonnegative value")
    return result


@dataclass(frozen=True)
class Geometry:
    query: tuple[int, ...]
    cached: tuple[int, ...]

    def __post_init__(self) -> None:
        require(type(self.query) is tuple and type(self.cached) is tuple,
                "geometry must be immutable tuples")
        require(len(self.query) > 0 and len(self.query) == len(self.cached),
                "geometry lengths differ or are empty")
        require(all(type(x) is int and x > 0 for x in self.query), "invalid queries")
        require(all(type(x) is int and x >= 0 for x in self.cached), "invalid cache")

    @property
    def work(self) -> int:
        return sum(q*k + q*(q+1)//2 for q, k in zip(self.query, self.cached))

    @property
    def independent_work(self) -> Fraction:
        return (Fraction(sum(self.query)*sum(self.cached), len(self.query))
                + sum(q*(q+1)//2 for q in self.query))

    def features(self, representation: str) -> tuple:
        aggregate = (len(self.query), sum(self.query), sum(self.cached))
        if representation == "aggregate":
            return aggregate
        if representation == "marginals":
            return tuple(sorted(self.query)), tuple(sorted(self.cached))
        if representation == "joint_work":
            return (*aggregate, sum(x*x for x in self.query),
                    sum(x*x for x in self.cached), self.work)
        if representation == "unordered_pairs":
            return tuple(sorted(zip(self.query, self.cached)))
        if representation == "ordered_pairs":
            return tuple(zip(self.query, self.cached))
        raise ValueError(f"unknown representation: {representation}")


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    def __post_init__(self) -> None:
        low = number(self.low, "low", positive=True)
        high = number(self.high, "high", positive=True)
        require(low <= high, "reversed bounds")


def reversal_witness(left: Geometry, right: Geometry, representation: str,
                     left_context: str, right_context: str,
                     left_costs: tuple[Interval, Interval],
                     right_costs: tuple[Interval, Interval]) -> dict:
    """Two actions, same declared feature vector/context, opposite strict winners.

    Costs must share a unit and boundary, represented in the context identity.
    This is an algebraic certificate conditional on bounds, NOT GPU evidence.
    """
    require(type(left_context) is str and bool(left_context), "empty context")
    require(left_context == right_context, "incompatible execution context")
    require(left.features(representation) == right.features(representation),
            "different feature vectors: not an indistinguishability witness")
    require(len(left_costs) == len(right_costs) == 2, "exactly two actions required")
    require(all(isinstance(x, Interval) for x in (*left_costs, *right_costs)),
            "intervals required")
    gaps = (left_costs[1].low-left_costs[0].high,
            right_costs[0].low-right_costs[1].high)
    require(min(gaps) > 0, "opposite winners not separated by supplied bounds")
    return {"representation": representation, "features": left.features(representation),
            "context": left_context, "gap_lower_bounds": list(gaps),
            "two_action_minimax_regret_lower_bound": min(gaps)/(1. + min(gaps)/max(gaps)),
            "scope": "single decision, two actions, no extra observations; conditional on bounds",
            "statistical_guarantee": False, "measured_gpu_witness": False}


def net_value(*, guaranteed_reuses: int | None, native_lower_us: float,
              candidate_upper_us: float, probe_us: float, setup_us: float,
              dispatch_per_call_us: float, sunk_us: float = 0.0,
              context_matches: bool, qualified: bool) -> dict:
    """Counterfactual arithmetic only; never changes an execution path.

    Include *all* paid costs even on fallback. Future reuse is an explicit
    precondition, never inferred from layer count or past graph repetitions.
    """
    native = number(native_lower_us, "native", positive=True)
    candidate = number(candidate_upper_us, "candidate", positive=True)
    probe = number(probe_us, "probe")
    setup = number(setup_us, "setup")
    dispatch = number(dispatch_per_call_us, "dispatch")
    sunk = number(sunk_us, "sunk")
    require(type(context_matches) is bool and type(qualified) is bool, "bool gates required")
    require(guaranteed_reuses is None or (type(guaranteed_reuses) is int and guaranteed_reuses > 0),
            "reuse must be unknown or a positive integer")
    paid = probe + setup + sunk
    require(math.isfinite(paid), "paid-cost overflow")
    try:
        net = None if guaranteed_reuses is None else guaranteed_reuses*(native-candidate-dispatch)-paid
    except OverflowError as exc:
        raise ValueError("net-value overflow") from exc
    require(net is None or math.isfinite(net), "net-value overflow")
    reason = ("context_mismatch" if not context_matches else "unqualified" if not qualified
              else "unknown_reuse" if guaranteed_reuses is None
              else "no_positive_net_bound" if net <= 0 else "positive_conditional_net_bound")
    return {"decision": "candidate_eligible" if reason == "positive_conditional_net_bound" else "native",
            "reason": reason, "paid_cost_us": paid, "net_lower_bound_us": net,
            "fallback_cost_us": paid, "runtime_promoted": False,
            "scope": "arithmetic conditional on supplied bounds and reuse, not a timing guarantee"}
