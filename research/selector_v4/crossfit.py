from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Mapping, Sequence
import numpy as np
from .eligibility import EligibilityDecision
from .identity import TacticIdentity
from .safe_tuner import SelectionReceipt, choose_tactic
from .schema import DEFAULT_THRESHOLDS, DecisionThresholds, TACTIC_CAP

@dataclass(frozen=True)
class RepeatEvidence:
    treatment_ratios: tuple[float, ...]
    native_controls: tuple[float, ...]
    cap_controls: tuple[float, ...]

@dataclass(frozen=True)
class FoldResult:
    held_out_repeat: int
    receipt: SelectionReceipt
    held_out_geomean: float
    held_out_minimum: float
    policy_geomean: float
    policy_minimum: float
    held_out_log_ratios: tuple[float, ...]


def _geo(values: Sequence[float]) -> float:
    return math.exp(sum(math.log(float(x)) for x in values) / len(values))

def crossfit_cell(
    identity: TacticIdentity,
    eligibility: EligibilityDecision,
    repeats: Mapping[int, RepeatEvidence],
    *,
    exact_outputs: bool,
    thresholds: DecisionThresholds = DEFAULT_THRESHOLDS,
) -> tuple[FoldResult, ...]:
    if sorted(repeats) != [0, 1, 2]:
        raise ValueError("crossfit requires exactly three process repeats")
    results: list[FoldResult] = []
    for held_out in range(3):
        train = [repeats[i] for i in range(3) if i != held_out]
        receipt = choose_tactic(
            identity, eligibility,
            [x for item in train for x in item.treatment_ratios],
            [x for item in train for x in item.native_controls],
            [x for item in train for x in item.cap_controls],
            exact_outputs=exact_outputs, thresholds=thresholds,
        )
        test = repeats[held_out]
        selected = receipt.tactic == TACTIC_CAP
        policy = test.treatment_ratios if selected else tuple(1.0 for _ in test.treatment_ratios)
        results.append(FoldResult(
            held_out_repeat=held_out, receipt=receipt,
            held_out_geomean=_geo(test.treatment_ratios),
            held_out_minimum=min(test.treatment_ratios),
            policy_geomean=_geo(policy), policy_minimum=min(policy),
            held_out_log_ratios=tuple(math.log(x) for x in test.treatment_ratios),
        ))
    return tuple(results)

def simultaneous_min_lcb(folds: Sequence[FoldResult], draws: int = 10000, seed: int = 99173) -> float:
    selected = [fold for fold in folds if fold.receipt.tactic == TACTIC_CAP]
    if not selected:
        return 1.0
    rng = np.random.default_rng(seed)
    matrix = []
    for fold in selected:
        logs = np.asarray(fold.held_out_log_ratios, dtype=np.float64)
        indices = rng.integers(0, len(logs), size=(draws, len(logs)))
        matrix.append(logs[indices].mean(axis=1))
    return float(np.exp(np.quantile(np.stack(matrix).min(axis=0), 0.025)))
