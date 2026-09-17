"""GPU regression tests for the two-stage top-logprobs reduction.

The primitive defect: a vocabulary tile whose every logit is ``-inf`` (an
``allowed_token_ids`` or grammar mask that empties a whole tile) produced
``block_max == -inf`` and ``exp(-inf - (-inf)) == NaN`` in the partial kernel.
That NaN poisoned the row's global log-normalizer even when other tiles held
valid logits. Before the fix, every row of a Qwen-sized batch masked to its
first eight tokens came back NaN.

These tests need a CUDA device and Triton; they are skipped elsewhere.
"""

import math

import pytest

from l20_stack.ops.triton_sampling import (
    logprob_topk_launch_config,
    top_logprobs,
    top_logprobs_reference,
    vllm_top_logprobs_out,
    vllm_top_logprobs_reference,
)

LOG_1_PLUS_E = math.log(1.0 + math.e)


def _cuda():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch


def _vllm_workspaces(torch, batch, vocab, top_n, block_vocab_override=None):
    config = logprob_topk_launch_config(
        vocab, top_n, batch=batch, block_vocab_override=block_vocab_override
    )
    blocks = config.blocks_per_row
    dev = "cuda"
    return {
        "partial_values": torch.empty((batch, blocks, top_n), device=dev, dtype=torch.float32),
        "partial_tokens": torch.empty((batch, blocks, top_n), device=dev, dtype=torch.int64),
        "partial_max": torch.empty((batch, blocks), device=dev, dtype=torch.float32),
        "partial_sum_exp": torch.empty((batch, blocks), device=dev, dtype=torch.float32),
        "partial_ranks": torch.empty((batch, blocks), device=dev, dtype=torch.int32),
    }


def test_all_masked_tile_does_not_poison_row_reduction():
    torch = _cuda()
    vocab, top_n = 2048, 2
    logits = torch.full((2, vocab), float("-inf"), device="cuda")
    # Row 0: live logits only in tile 0; tiles 1-3 fully masked.
    logits[0, 0], logits[0, 1] = 0.0, 1.0
    # Row 1: live logits only in tile 2; tiles 0, 1, 3 fully masked.
    logits[1, 1200], logits[1, 1201] = 0.0, 1.0
    values, tokens = top_logprobs(logits, top_n=top_n, block_vocab_override=512)
    assert not torch.isnan(values).any(), values
    expected = torch.tensor(
        [[1.0 - LOG_1_PLUS_E, -LOG_1_PLUS_E]] * 2, device="cuda", dtype=torch.float32
    )
    torch.testing.assert_close(values, expected, atol=1e-6, rtol=0)
    assert tokens.tolist() == [[1, 0], [1201, 1200]]


@pytest.mark.parametrize("vocab", [32_000, 128_256, 151_936])
def test_allowed_token_mask_matches_reference_at_default_tile_size(vocab):
    torch = _cuda()
    batch, top_n, live = 8, 5, 8
    gen = torch.Generator(device="cuda").manual_seed(1234)
    logits = torch.full((batch, vocab), float("-inf"), device="cuda")
    logits[:, :live] = torch.randn((batch, live), device="cuda", generator=gen)
    values, tokens = top_logprobs(logits, top_n=top_n)
    ref_values, ref_tokens = top_logprobs_reference(logits, top_n=top_n)
    assert not torch.isnan(values).any()
    torch.testing.assert_close(values, ref_values, atol=1e-5, rtol=0)
    assert torch.equal(tokens, ref_tokens)


def test_vllm_path_survives_all_masked_tiles():
    torch = _cuda()
    batch, vocab, top_n = 4, 151_936, 5
    gen = torch.Generator(device="cuda").manual_seed(7)
    logits = torch.full((batch, vocab), float("-inf"), device="cuda")
    # Live tokens scattered far apart so most tiles are fully masked.
    live = torch.tensor([3, 5_000, 70_000, 151_000], device="cuda")
    logits[:, live] = torch.randn((batch, live.numel()), device="cuda", generator=gen)
    token_ids = live[torch.randint(0, live.numel(), (batch,), device="cuda", generator=gen)]
    ws = _vllm_workspaces(torch, batch, vocab, top_n)
    out_tokens = torch.empty((batch, top_n + 1), device="cuda", dtype=torch.int32)
    out_logprobs = torch.empty((batch, top_n + 1), device="cuda", dtype=torch.float32)
    out_ranks = torch.empty((batch,), device="cuda", dtype=torch.int32)
    vllm_top_logprobs_out(
        logits, token_ids, out_tokens, out_logprobs, out_ranks, top_n=top_n, **ws
    )
    ref_tokens, ref_logprobs, ref_ranks = vllm_top_logprobs_reference(
        logits, token_ids, top_n=top_n
    )
    # Only four live tokens exist, so columns beyond them are -inf ties whose
    # token order is unspecified; compare the live prefix exactly.
    assert not torch.isnan(out_logprobs[:, : live.numel() + 1]).any()
    torch.testing.assert_close(
        out_logprobs[:, : live.numel() + 1], ref_logprobs[:, : live.numel() + 1], atol=1e-5, rtol=0
    )
    assert torch.equal(out_tokens[:, : live.numel() + 1], ref_tokens[:, : live.numel() + 1])
    assert torch.equal(out_ranks, ref_ranks)


def test_entirely_invalid_row_is_nan_and_isolated():
    """A row with no live logit has no normalizer; it returns NaN like
    ``torch.log_softmax`` and must not disturb its neighbours."""
    torch = _cuda()
    vocab, top_n = 2048, 2
    logits = torch.full((3, vocab), float("-inf"), device="cuda")
    logits[0, 0], logits[0, 1] = 0.0, 1.0
    logits[2, 10], logits[2, 11] = 0.0, 1.0
    values, tokens = top_logprobs(logits, top_n=top_n, block_vocab_override=512)
    assert torch.isnan(values[1]).all()
    expected = torch.tensor([1.0 - LOG_1_PLUS_E, -LOG_1_PLUS_E], device="cuda")
    torch.testing.assert_close(values[0], expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(values[2], expected, atol=1e-6, rtol=0)
    assert tokens[0].tolist() == [1, 0]
    assert tokens[2].tolist() == [11, 10]


def test_non_contiguous_logits_are_rejected():
    torch = _cuda()
    wide = torch.randn((4, 2 * 4096), device="cuda")
    strided = wide[:, ::2]  # shape [4, 4096], stride 2: flat pointer math would be wrong
    assert not strided.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        top_logprobs(strided, top_n=2)
    top_logprobs(strided.contiguous(), top_n=2)  # the explicit copy is accepted
