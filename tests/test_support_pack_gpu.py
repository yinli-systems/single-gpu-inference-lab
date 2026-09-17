"""GPU tests for compact sampling-support packing with joint logprob outputs."""

import pytest

from l20_stack.ops.triton_support_pack import (
    support_pack,
    support_pack_launch_config,
    support_pack_reference,
)


def _cuda():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch


def _topk_masked_logits(torch, batch, vocab, k_per_row, gen):
    """Random logits with only each row's top-k kept finite (post top-k filter)."""

    logits = torch.randn((batch, vocab), device="cuda", generator=gen)
    out = torch.full_like(logits, -float("inf"))
    for row, k in enumerate(k_per_row):
        idx = torch.topk(logits[row], k).indices
        out[row, idx] = logits[row, idx]
    return out


def _compare(torch, got, ref, counts, max_support):
    ids, cnt, ovf, logz, slp, sup = got
    r_ids, r_cnt, r_ovf, r_logz, r_slp, r_sup = ref
    assert torch.equal(cnt, r_cnt)
    assert torch.equal(ovf, r_ovf)
    active = cnt > 0
    torch.testing.assert_close(logz[active], r_logz[active], atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(slp[active], r_slp[active], atol=1e-5, rtol=1e-6)
    for row in range(cnt.shape[0]):
        n = min(int(cnt[row]), max_support)
        assert torch.equal(ids[row, :n], r_ids[row, :n]), row
        if sup is not None and n:
            torch.testing.assert_close(sup[row, :n], r_sup[row, :n], atol=1e-5, rtol=1e-6)


def test_launch_config_is_shape_driven():
    assert support_pack_launch_config(151_936)["block_vocab"] == 8192
    assert support_pack_launch_config(32_000)["block_vocab"] == 2048
    with pytest.raises(ValueError):
        support_pack_launch_config(0)


@pytest.mark.parametrize("vocab", [32_000, 151_936])
@pytest.mark.parametrize("max_support", [64, 256])
def test_matches_reference_after_topk_filter(vocab, max_support):
    torch = _cuda()
    batch = 16
    gen = torch.Generator(device="cuda").manual_seed(11)
    ks = [1, 2, 5, 20, 50, max_support, max_support - 1, 7] * 2
    logits = _topk_masked_logits(torch, batch, vocab, ks, gen)
    num_sampled = torch.ones(batch, device="cuda", dtype=torch.int32)
    sampled = torch.stack([torch.nonzero(torch.isfinite(logits[r]))[0, 0] for r in range(batch)])
    got = support_pack(logits, num_sampled, sampled, max_support=max_support, return_support_logprobs=True)
    ref = support_pack_reference(logits, num_sampled, sampled, max_support=max_support)
    _compare(torch, got, ref, got[1], max_support)
    assert not bool(got[2].any())
    assert got[1].tolist() == ks


def test_overflow_is_flagged_not_truncated_silently():
    torch = _cuda()
    batch, vocab, max_support = 4, 151_936, 64
    gen = torch.Generator(device="cuda").manual_seed(3)
    logits = _topk_masked_logits(torch, batch, vocab, [64, 65, 1000, 3], gen)
    num_sampled = torch.ones(batch, device="cuda", dtype=torch.int32)
    sampled = torch.zeros(batch, device="cuda", dtype=torch.int64)
    for r in range(batch):
        sampled[r] = torch.nonzero(torch.isfinite(logits[r]))[0, 0]
    ids, cnt, ovf, logz, slp, _ = support_pack(logits, num_sampled, sampled, max_support=max_support)
    assert cnt.tolist() == [64, 65, 1000, 3]
    assert ovf.tolist() == [0, 1, 1, 0]
    # logz / sampled logprob are still exact over the full support even when overflowed
    ref = support_pack_reference(logits, num_sampled, sampled, max_support=max_support)
    torch.testing.assert_close(logz, ref[3], atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(slp, ref[4], atol=1e-5, rtol=1e-6)
    # the first max_support ids of an overflowed row are still the lowest ids kept
    assert torch.equal(ids[2], ref[0][2])


def test_inactive_rows_are_skipped_and_isolated():
    torch = _cuda()
    batch, vocab, max_support = 3, 32_000, 64
    gen = torch.Generator(device="cuda").manual_seed(5)
    logits = _topk_masked_logits(torch, batch, vocab, [10, 10, 10], gen)
    num_sampled = torch.tensor([1, 0, 1], device="cuda", dtype=torch.int32)
    sampled = torch.tensor([int(torch.nonzero(torch.isfinite(logits[r]))[0, 0]) for r in range(batch)], device="cuda")
    ids, cnt, ovf, _logz, slp, _ = support_pack(logits, num_sampled, sampled, max_support=max_support)
    assert cnt.tolist() == [10, 0, 10]
    assert ovf.tolist() == [0, 0, 0]
    ref = support_pack_reference(logits, num_sampled, sampled, max_support=max_support)
    for r in (0, 2):
        assert torch.equal(ids[r, :10], ref[0][r, :10])
        torch.testing.assert_close(slp[r], ref[4][r], atol=1e-5, rtol=1e-6)


def test_support_ids_match_upstream_finite_rule_with_partial_tiles():
    """Support must be exactly the finite logits, including +inf exclusion and
    entries in the ragged last tile of the vocabulary."""
    torch = _cuda()
    vocab, max_support = 151_936, 64
    logits = torch.full((1, vocab), -float("inf"), device="cuda")
    keep = torch.tensor([0, 4095, 4096, 151_935, 151_000, 77_777], device="cuda")
    logits[0, keep] = torch.arange(keep.numel(), device="cuda", dtype=torch.float32)
    logits[0, 5] = float("inf")  # +inf is not part of the support, matching upstream
    num_sampled = torch.ones(1, device="cuda", dtype=torch.int32)
    sampled = torch.tensor([151_935], device="cuda")
    ids, cnt, _ovf, logz, _slp, sup = support_pack(logits, num_sampled, sampled, max_support=max_support, return_support_logprobs=True)
    assert cnt.tolist() == [6]
    assert ids[0, :6].tolist() == sorted(keep.tolist())
    ref = support_pack_reference(logits, num_sampled, sampled, max_support=max_support)
    torch.testing.assert_close(logz, ref[3], atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(sup[0, :6], ref[5][0, :6], atol=1e-5, rtol=1e-6)


def test_rejects_bad_contracts():
    torch = _cuda()
    logits = torch.randn((2, 4096), device="cuda")
    ns = torch.ones(2, device="cuda", dtype=torch.int32)
    sm = torch.zeros(2, device="cuda", dtype=torch.int64)
    with pytest.raises(ValueError, match="power of two"):
        support_pack(logits, ns, sm, max_support=48)
    with pytest.raises(ValueError, match="contiguous"):
        support_pack(torch.randn((2, 8192), device="cuda")[:, ::2], ns, sm, max_support=64)
