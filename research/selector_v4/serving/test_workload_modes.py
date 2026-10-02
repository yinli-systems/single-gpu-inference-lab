"""Fixed scheduling preserves all cached input/output cells and fails closed."""

import copy

import pytest

from research.release_qualification.serving.workloads import prefix_tokens, work_specs
from research.selector_v4.serving.workload_modes import specs, verify_configuration


@pytest.mark.parametrize("block", range(4))
def test_only_parity_scheduling_changes_and_all_cells_remain(block):
    original = work_specs(prefix_tokens(), block)
    assert specs(prefix_tokens(), block, stage="functional") == original
    assert specs(prefix_tokens(), block, stage="performance") == original
    fixed = specs(prefix_tokens(), block, stage="parity")
    assert fixed == specs(prefix_tokens(), block, stage="fixed_parity")
    assert set(fixed) == set(original)
    for name, work in fixed.items():
        assert work["concurrency"] == 1
        expected = copy.deepcopy(original[name])
        expected["concurrency"] = 1
        assert work == expected
    assert any(work["concurrency"] > 1 for work in original.values())


def resolved():
    return {
        "disable_cuda_graph": False,
        "disable_overlap_schedule": False,
        "disable_radix_cache": False,
        "enable_deterministic_inference": False,
        "max_running_requests": 1,
    }


def test_resolved_cached_single_request_mode_is_required_before_measurement():
    verify_configuration(resolved(), "parity")


@pytest.mark.parametrize(
    "field,value",
    [
        ("disable_cuda_graph", True),
        ("disable_overlap_schedule", True),
        ("disable_radix_cache", True),
        ("enable_deterministic_inference", True),
        ("max_running_requests", 16),
    ],
)
def test_resolved_incompatible_mode_is_rejected(field, value):
    info = resolved()
    info[field] = value
    with pytest.raises(ValueError):
        verify_configuration(info, "parity")
