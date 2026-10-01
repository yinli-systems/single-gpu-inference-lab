"""Transport partition and error integrity; synthetic, no actual HTTP authority."""

import json

import pytest

from research.selector_v4.serving.http_stream import TokenStream


def event(ids, count, *, finished=False, cached=8192):
    meta = {"completion_tokens": count, "cached_tokens": cached}
    if finished:
        meta["finish_reason"] = {"type": "length", "length": count}
    return b"data: " + json.dumps({"output_ids": ids, "meta_info": meta}).encode() + b"\r\n\r\n"


def test_network_partitioning_preserves_full_token_stream_and_arrival_times():
    stream = TokenStream(3)
    first = event([7], 1)
    for part in (first[:7], first[7:31], first[31:]):
        stream.feed(part, 0.2)
    stream.feed(event([7, 9], 2), 0.3)  # Cumulative server response.
    stream.feed(event([11], 3, finished=True), 0.5)  # Incremental response.
    stream.feed(b"data: [DONE]\n\n", 0.51)
    result = stream.complete(expected_cached_tokens=8192)
    assert result["tokens"] == [7, 9, 11]
    assert result["token_times"] == [0.2, 0.3, 0.5]
    assert result["tpot"] == pytest.approx(0.15)


def test_repeated_cumulative_event_cannot_manufacture_new_token_or_timestamp():
    stream = TokenStream(2)
    stream.feed(event([7], 1), 0.2)
    stream.feed(event([7], 1), 0.25)
    stream.feed(event([7, 9], 2, finished=True), 0.4)
    stream.feed(b"data: [DONE]\n\n", 0.41)
    assert stream.complete()["token_times"] == [0.2, 0.4]


@pytest.mark.parametrize("ending", [b"", b"data: [DONE]\n", b"data: [DO"])
def test_disconnect_after_last_token_is_retained_as_incomplete(ending):
    stream = TokenStream(2)
    stream.feed(event([7], 1), 0.2)
    stream.feed(event([9], 2, finished=True) + ending, 0.4)
    with pytest.raises(ValueError, match="Truncated|Incomplete"):
        stream.complete()
    assert stream.tokens == [7, 9] and len(stream.events) == 2


def test_error_event_and_conflicting_cumulative_prefix_are_retained():
    stream = TokenStream(2)
    stream.feed(event([7], 1), 0.2)
    with pytest.raises(ValueError, match="prefix mismatch"):
        stream.feed(event([8, 9], 2), 0.4)
    assert len(stream.events) == 2 and stream.tokens == [7]
    error = TokenStream(2)
    with pytest.raises(ValueError, match="stream error"):
        error.feed(b'data: {"error":"worker failed"}\n\n', 0.2)
    assert "worker failed" in error.events[0]["data"]


def test_cache_failure_and_batched_delivery_do_not_get_epsilon_metrics():
    stream = TokenStream(2)
    stream.feed(event([7], 1, cached=0), 0.2)
    stream.feed(event([9], 2, finished=True, cached=0), 0.4)
    stream.feed(b"data: [DONE]\n\n", 0.41)
    with pytest.raises(ValueError, match="cache hit"):
        stream.complete(expected_cached_tokens=8192)
    coalesced = TokenStream(2)
    coalesced.feed(event([7, 9], 2, finished=True) + b"data: [DONE]\n\n", 0.4)
    with pytest.raises(ValueError, match="Nonpositive"):
        coalesced.complete()


def test_bounds_and_data_after_done_block_invalid_metrics():
    bounded = TokenStream(2, maximum_event_bytes=16)
    with pytest.raises(ValueError, match="Bounded"):
        bounded.feed(b"data: " + b"x" * 17, 0.2)
    stream = TokenStream(2)
    stream.feed(b"data: [DONE]\n\n", 0.2)
    with pytest.raises(ValueError, match="after terminal"):
        stream.feed(event([7], 1), 0.4)
