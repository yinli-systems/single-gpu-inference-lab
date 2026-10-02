"""Separate fixed-schedule parity; ordinary concurrent workloads stay frozen."""

import copy

from research.release_qualification.serving.workloads import work_specs
from research.selector_v4.public_qualification.gates import need

PARITY_SCOPE = "fixed_single_request_with_radix_cache_and_cuda_graph"


def verify_configuration(info, stage):
    """Check resolved worker arguments before any warmup or scored request."""
    need(
        not info["disable_cuda_graph"] and not info["disable_overlap_schedule"],
        "Actual Graph/overlap serving required",
    )
    need(not info["disable_radix_cache"], "Actual radix cache must remain enabled")
    need(
        bool(info["enable_deterministic_inference"]) == (stage == "deterministic_control"),
        "Actual separately declared global deterministic mode",
    )
    if stage in ("parity", "fixed_parity"):
        need(info["max_running_requests"] == 1, "Actual fixed single-request server schedule")


def specs(prefix, block, *, stage):
    result = work_specs(prefix, block)
    if stage in ("parity", "fixed_parity"):
        result = copy.deepcopy(result)
        for work in result.values():
            work["concurrency"] = 1
    return result
