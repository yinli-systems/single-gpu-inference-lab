import copy

import pytest

from research.selector_v4.serving.slo_report import summarize
from research.selector_v4.serving.test_http_analysis import measured


def test_full_requests_input_coverage_and_zero_goodput_retained():
    work, block = measured()
    r = summarize(block, work)
    assert len(r["slo_threshold_grid"]) == 30
    assert r["slo_threshold_grid"][0]["requests_meeting_both"] == 0
    assert r["slo_threshold_grid"][0]["strict_goodput_requests_per_second"] == 0
    assert r["logical_input_tokens_per_second"] >= r["uncached_input_tokens_per_second"] > 0
    assert not r["qualification_authority"]


def test_success_only_or_changed_cache_data_cannot_produce_slo_report():
    work, original = measured()
    b = copy.deepcopy(original)
    b["requests"].pop()
    with pytest.raises(ValueError):
        summarize(b, work)
    b = copy.deepcopy(original)
    b["requests"][0]["cached_tokens"] = 1000000
    with pytest.raises(ValueError):
        summarize(b, work)
