from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
from .cache import SafeTacticCache
from .eligibility import evaluate_eligibility
from .identity import TacticIdentity
from .schema import TACTIC_CAP, TACTIC_NATIVE

@dataclass(frozen=True)
class AppliedTactic:
    tactic: str
    cache_hit: bool
    reason: str
    identity_key: str


def validate_wrapper(identity: TacticIdentity, entry: dict[str, Any], wrapper: Any) -> bool:
    try:
        if getattr(wrapper, "_backend", None) != "fa2" or getattr(wrapper, "_jit_module", None) is not None:
            return False
        module = getattr(wrapper, "_cached_module", None)
        if (module is None or not callable(getattr(wrapper, "run_resource", None)) or
                not callable(getattr(module, identity.operation["layout"]+"_run_resource", None)) or
                getattr(module, "resource_kernel_isolation", False) is not True):
            return False
        if entry["tactic"] == TACTIC_CAP and not evaluate_eligibility(identity).eligible:
            return False
        return True
    except Exception:
        return False

def apply_cached_tactic(
    wrapper: Any,
    identity: TacticIdentity,
    cache: SafeTacticCache,
    validator: Callable[[TacticIdentity, dict[str, Any], Any], bool] = validate_wrapper,
) -> AppliedTactic:
    """Apply a validated tactic before plan/capture; every failure is native."""
    wrapper._sgi_resource_policy = 0
    wrapper._sgi_tactic_run = wrapper.run
    entry = cache.lookup(identity, validator=lambda i, e: validator(i, e, wrapper))
    if entry is None:
        return AppliedTactic(TACTIC_NATIVE, False, "cache_miss_or_revalidation_failure", identity.key)
    tactic = entry["tactic"]
    if tactic == TACTIC_CAP:
        wrapper._sgi_resource_policy = 1
        wrapper._sgi_tactic_run = wrapper.run_resource
    elif tactic != TACTIC_NATIVE:
        wrapper._sgi_resource_policy = 0
        return AppliedTactic(TACTIC_NATIVE, False, "unknown_tactic", identity.key)
    return AppliedTactic(tactic, True, "validated_cache_hit", identity.key)

def publish_and_apply(wrapper: Any, identity: TacticIdentity, cache: SafeTacticCache,
                      receipt: dict[str, Any], provenance: dict[str, Any]) -> AppliedTactic:
    cache.publish(identity, receipt, provenance=provenance)
    cache.reload()
    return apply_cached_tactic(wrapper, identity, cache)


def selected_run(wrapper: Any) -> Callable:
    """Resolve before capture/serving; the official wrapper.run is never replaced."""
    return getattr(wrapper, "_sgi_tactic_run", wrapper.run)
