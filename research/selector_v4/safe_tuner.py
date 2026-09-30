from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib, math
from typing import Any, Iterable, Sequence
import numpy as np
from .eligibility import EligibilityDecision
from .identity import TacticIdentity
from .schema import DEFAULT_THRESHOLDS, DecisionThresholds, TACTIC_CAP, TACTIC_NATIVE

@dataclass(frozen=True)
class SelectionReceipt:
    identity_key: str
    tactic: str
    passed: bool
    reason: str
    eligible: bool
    eligibility_reasons: tuple[str, ...]
    exact_outputs: bool
    blocks: int
    geomean: float | None
    lcb95: float | None
    minimum: float | None
    native_control_ci90: tuple[float, float] | None
    cap_control_ci90: tuple[float, float] | None
    thresholds: dict[str, Any]
    bootstrap_seed: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["eligibility_reasons"] = list(self.eligibility_reasons)
        for key in ("native_control_ci90", "cap_control_ci90"):
            if value[key] is not None:
                value[key] = list(value[key])
        return value

def _as_logs(ratios: Iterable[float], label: str) -> np.ndarray:
    arr = np.asarray(tuple(float(x) for x in ratios), dtype=np.float64)
    if arr.ndim != 1 or not len(arr) or not np.isfinite(arr).all() or np.any(arr <= 0):
        raise ValueError(f"invalid {label} ratios")
    return np.log(arr)

def _seed(identity: TacticIdentity) -> int:
    return int(hashlib.sha256((identity.key + ":safe-tuner").encode()).hexdigest()[:16], 16) % (2**32)

def _bootstrap_mean(logs: np.ndarray, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(logs), size=(draws, len(logs)), endpoint=False)
    return logs[indices].mean(axis=1)

def _control_ci(log_ratios: Sequence[float], draws: int, seed: int) -> tuple[float, float]:
    logs = _as_logs(log_ratios, "control")
    values = np.exp(np.quantile(_bootstrap_mean(logs, draws, seed), (0.05, 0.95)))
    return float(values[0]), float(values[1])

def _control_passes(ci: tuple[float, float], tolerance: float) -> bool:
    return ci[0] >= 1.0 / (1.0 + tolerance) and ci[1] <= 1.0 + tolerance

def choose_tactic(
    identity: TacticIdentity,
    eligibility: EligibilityDecision,
    native_over_cap_ratios: Sequence[float],
    native_control_ratios: Sequence[float],
    cap_control_ratios: Sequence[float],
    *,
    exact_outputs: bool,
    thresholds: DecisionThresholds = DEFAULT_THRESHOLDS,
) -> SelectionReceipt:
    identity.validate()
    thresholds.validate()
    seed = _seed(identity)
    base = dict(
        identity_key=identity.key,
        eligible=eligibility.eligible,
        eligibility_reasons=eligibility.reasons,
        exact_outputs=bool(exact_outputs),
        thresholds=thresholds.to_dict(),
        bootstrap_seed=seed,
    )
    if not eligibility.eligible:
        return SelectionReceipt(tactic=TACTIC_NATIVE, passed=True, reason="static_ineligible", blocks=0,
            geomean=None, lcb95=None, minimum=None, native_control_ci90=None, cap_control_ci90=None, **base)
    logs = _as_logs(native_over_cap_ratios, "treatment")
    if len(logs) < thresholds.min_training_blocks:
        return SelectionReceipt(tactic=TACTIC_NATIVE, passed=False, reason="insufficient_training_blocks", blocks=len(logs),
            geomean=None, lcb95=None, minimum=None, native_control_ci90=None, cap_control_ci90=None, **base)
    native_ci = _control_ci(native_control_ratios, thresholds.bootstrap_draws, seed ^ 0xA551)
    cap_ci = _control_ci(cap_control_ratios, thresholds.bootstrap_draws, seed ^ 0xC4A9)
    geomean = float(np.exp(logs.mean()))
    lcb95 = float(np.exp(np.quantile(_bootstrap_mean(logs, thresholds.bootstrap_draws, seed), 0.025)))
    minimum = float(np.exp(logs.min()))
    checks = {
        "exact_outputs": exact_outputs,
        "native_control": _control_passes(native_ci, thresholds.control_equivalence),
        "cap_control": _control_passes(cap_ci, thresholds.control_equivalence),
        "geomean": geomean >= thresholds.train_geomean,
        "lcb95": lcb95 >= thresholds.train_lcb95,
        "minimum": minimum >= thresholds.train_block_min,
    }
    passed = all(checks.values())
    failed = [name for name, ok in checks.items() if not ok]
    return SelectionReceipt(
        tactic=TACTIC_CAP if passed else TACTIC_NATIVE,
        passed=passed,
        reason="qualified_cap" if passed else "native_fallback:" + ",".join(failed),
        blocks=len(logs), geomean=geomean, lcb95=lcb95, minimum=minimum,
        native_control_ci90=native_ci, cap_control_ci90=cap_ci, **base,
    )
