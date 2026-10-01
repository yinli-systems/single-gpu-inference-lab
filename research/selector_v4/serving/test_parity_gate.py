"""A changed middle token/logprob cannot be hidden by a matching final token."""

import copy
import json

import pytest

from research.selector_v4.serving.parity_gate import compare_blocks


def block():
    tokens = [7, 9, 11]
    values = [[-0.1, token, None] for token in tokens]
    tops = [[[-0.1 - j, token + j, None] for j in range(5)] for token in tokens]
    request = {
        "id": "logical0",
        "complete": True,
        "tokens": tokens,
        "payload": {"input_ids": [123], "sampling_params": {"max_new_tokens": 3}},
        "raw_events": [
            {
                "data": json.dumps(
                    {"meta_info": {"output_token_logprobs": values, "output_top_logprobs": tops}}
                )
            },
            {"data": "[DONE]"},
        ],
    }
    return {"complete": True, "errors": [], "workload_sha256": "synthetic", "requests": [request]}


def test_complete_natural_and_logprob_comparisons_remain_separate_from_qualification():
    base = block()
    result = compare_blocks(base, copy.deepcopy(base), require_logprobs=True)
    assert result["parity_pass"] and result["tokens_compared"] == 3
    assert not result["historical_token_divergence_resolved"] and not result["full_http_qualified"]


def test_middle_token_and_logprob_changes_block_parity_with_all_positions_retained():
    base = block()
    candidate = copy.deepcopy(base)
    candidate["requests"][0]["tokens"][1] = 10
    result = compare_blocks(base, candidate, require_logprobs=False)
    assert not result["parity_pass"] and result["mismatches"][0]["positions"] == [1]
    candidate = copy.deepcopy(base)
    events = candidate["requests"][0]["raw_events"]
    parsed = json.loads(events[0]["data"])
    parsed["meta_info"]["output_token_logprobs"][1][0] -= 0.0001
    events[0]["data"] = json.dumps(parsed)
    assert not compare_blocks(base, candidate, require_logprobs=True)["parity_pass"]


@pytest.mark.parametrize(
    "fault",
    ["missing-logprobs", "duplicate-request", "truncated-token", "changed-input", "request-error"],
)
def test_incomplete_or_changed_request_cannot_get_a_parity_pass(fault):
    base = block()
    candidate = copy.deepcopy(base)
    if fault == "missing-logprobs":
        candidate["requests"][0]["raw_events"] = []
    elif fault == "duplicate-request":
        candidate["requests"].append(copy.deepcopy(candidate["requests"][0]))
    elif fault == "truncated-token":
        candidate["requests"][0]["tokens"].pop()
    elif fault == "changed-input":
        candidate["requests"][0]["payload"]["input_ids"] = [124]
    else:
        candidate["errors"] = [{"type": "HTTP500"}]
    with pytest.raises(ValueError):
        compare_blocks(base, candidate, require_logprobs=True)
