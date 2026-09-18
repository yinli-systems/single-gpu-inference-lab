"""Compact sampling-support packing with joint logprob outputs.

Context. vLLM's sampling-mask feature (``--return-sampling-mask``) returns, for
every generated token, the set of vocabulary IDs that survived the sampling
filters (top-k/top-p/min-p/penalties/bias) so an RL trainer can replay the
behaviour policy's truncated distribution. The upstream implementation
re-scans the processed logits into a bit-packed ``[B, ceil(V/8)]`` mask,
copies that bitmap to the host every step, and unpacks the full vocabulary
with ``np.unpackbits`` before taking ``nonzero``. The feature also requires a
bounded ``top_k``, so the surviving support is at most ``top_k`` tokens; the
bitmap path nevertheless moves and unpacks ``V`` bits per row per step.

This module packs the same support as compact token IDs ``[B, K]`` with a
per-row count, in one pass over the processed logits, and computes the
processed distribution's log-normalizer in the same pass. From that a caller
gets, together and consistently:

* the exact surviving support (token IDs, in ascending vocabulary order),
* ``count`` per row, with an explicit overflow flag when the support exceeds
  ``K`` (the caller must then fall back; nothing is silently truncated),
* the processed logprob of the sampled token, ``z[t] - logsumexp(z)``,
* optionally the processed logprobs of every support token ``[B, K]``.

Contract. "Support" means finite processed logits, matching upstream's
``keep = isfinite(logit)`` rule. The log-normalizer is over the same finite
set. Rows with ``num_sampled == 0`` (chunked-prefill rows that produced no
token) get ``count = 0`` and are skipped. A row with an empty support yields
NaN logprobs, as ``torch.log_softmax`` would; it cannot occur after a valid
top-k filter, and the caller-visible count of zero makes it detectable.
"""

from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None

try:  # pragma: no cover - optional dependency
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


def support_pack_launch_config(vocab_size: int) -> dict:
    """Tile size and warps for one-row-per-program support packing."""

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    block = 8192 if vocab_size >= 65536 else 2048
    return {"block_vocab": block, "num_warps": 8 if block >= 4096 else 4, "num_stages": 1}


if triton is not None:

    @triton.jit
    def _support_pack_kernel(
        logits_ptr,
        logits_row_stride,
        num_sampled_ptr,
        sampled_ptr,
        out_ids_ptr,
        out_counts_ptr,
        out_overflow_ptr,
        out_logz_ptr,
        out_sampled_logprob_ptr,
        out_support_logprobs_ptr,
        VOCAB: tl.constexpr,
        MAX_SUPPORT: tl.constexpr,
        BLOCK: tl.constexpr,
        WRITE_SUPPORT_LOGPROBS: tl.constexpr,
    ):
        row = tl.program_id(0)
        active = tl.load(num_sampled_ptr + row) > 0
        row_ptr = logits_ptr + row * logits_row_stride

        # Pass 1: count, running max and shifted sum (online log-sum-exp), and
        # compact token IDs. IDs are emitted in ascending order because the
        # scan walks the vocabulary in order and the prefix sum is stable.
        running = tl.zeros((), dtype=tl.int32)
        run_max = tl.full((), -float("inf"), tl.float32)
        run_sum = tl.zeros((), dtype=tl.float32)
        for start in range(0, VOCAB, BLOCK):
            offsets = start + tl.arange(0, BLOCK)
            in_vocab = offsets < VOCAB
            values = tl.load(row_ptr + offsets, mask=in_vocab, other=-float("inf")).to(tl.float32)
            keep = (values > -float("inf")) & (values < float("inf")) & in_vocab & active

            # online log-sum-exp over the kept values of this tile
            tile_max = tl.max(tl.where(keep, values, -float("inf")), axis=0)
            new_max = tl.maximum(run_max, tile_max)
            shift = tl.where(new_max == -float("inf"), 0.0, new_max)
            tile_sum = tl.sum(tl.where(keep, tl.exp(values - shift), 0.0), axis=0)
            run_sum = run_sum * tl.exp(tl.where(run_max == -float("inf"), 0.0, run_max) - shift) + tile_sum
            run_max = new_max

            keep_i = keep.to(tl.int32)
            prefix = tl.cumsum(keep_i, axis=0)  # inclusive
            position = running + prefix - 1
            store = keep & (position < MAX_SUPPORT)
            tl.store(
                out_ids_ptr + row * MAX_SUPPORT + position,
                offsets.to(tl.int32),
                mask=store,
            )
            running += tl.sum(keep_i, axis=0)

        tl.store(out_counts_ptr + row, running)
        tl.store(out_overflow_ptr + row, (running > MAX_SUPPORT).to(tl.int32))
        logz = run_max + tl.log(run_sum)
        tl.store(out_logz_ptr + row, logz)

        sampled = tl.load(sampled_ptr + row).to(tl.int64)
        sampled_value = tl.load(row_ptr + sampled).to(tl.float32)
        tl.store(out_sampled_logprob_ptr + row, sampled_value - logz)

        if WRITE_SUPPORT_LOGPROBS:
            # Pass 2 over the compact IDs only (<= MAX_SUPPORT gathers). The
            # IDs were written by other threads of this program; fence first.
            tl.debug_barrier()
            k_offsets = tl.arange(0, MAX_SUPPORT)
            valid_k = k_offsets < tl.minimum(running, MAX_SUPPORT)
            ids = tl.load(out_ids_ptr + row * MAX_SUPPORT + k_offsets, mask=valid_k, other=0)
            vals = tl.load(row_ptr + ids.to(tl.int64), mask=valid_k, other=-float("inf")).to(tl.float32)
            tl.store(
                out_support_logprobs_ptr + row * MAX_SUPPORT + k_offsets,
                vals - logz,
                mask=valid_k,
            )


def _check(name: str, tensor, *, device, dtype=None, shape=None) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}")


def support_pack_out(
    processed_logits,
    num_sampled,
    sampled_token_ids,
    *,
    out_token_ids,
    out_counts,
    out_overflow,
    out_logz,
    out_sampled_logprob,
    out_support_logprobs=None,
    max_support: int,
) -> None:
    """Pack the finite-logit support and joint logprob outputs into caller tensors.

    ``max_support`` must be a power of two (it is used as a Triton block); pass
    the next power of two above the largest ``top_k`` in the batch.
    """

    if torch is None or triton is None:
        raise RuntimeError("support_pack_out requires PyTorch and Triton")
    if processed_logits.ndim != 2:
        raise ValueError("expected processed_logits with shape [batch, vocab]")
    batch, vocab = processed_logits.shape
    if max_support <= 0 or (max_support & (max_support - 1)) != 0:
        raise ValueError("max_support must be a positive power of two")
    device = processed_logits.device
    _check("processed_logits", processed_logits, device=device)
    _check("num_sampled", num_sampled, device=device, dtype=torch.int32, shape=(batch,))
    if sampled_token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("sampled_token_ids must be int32 or int64")
    _check("sampled_token_ids", sampled_token_ids, device=device, shape=(batch,))
    _check("out_token_ids", out_token_ids, device=device, dtype=torch.int32, shape=(batch, max_support))
    _check("out_counts", out_counts, device=device, dtype=torch.int32, shape=(batch,))
    _check("out_overflow", out_overflow, device=device, dtype=torch.int32, shape=(batch,))
    _check("out_logz", out_logz, device=device, dtype=torch.float32, shape=(batch,))
    _check("out_sampled_logprob", out_sampled_logprob, device=device, dtype=torch.float32, shape=(batch,))
    write_support = out_support_logprobs is not None
    if write_support:
        _check(
            "out_support_logprobs",
            out_support_logprobs,
            device=device,
            dtype=torch.float32,
            shape=(batch, max_support),
        )
    else:
        out_support_logprobs = out_logz  # unused placeholder pointer
    if batch == 0:
        return
    config = support_pack_launch_config(int(vocab))
    _support_pack_kernel[(batch,)](
        processed_logits,
        processed_logits.stride(0),
        num_sampled,
        sampled_token_ids,
        out_token_ids,
        out_counts,
        out_overflow,
        out_logz,
        out_sampled_logprob,
        out_support_logprobs,
        VOCAB=int(vocab),
        MAX_SUPPORT=int(max_support),
        BLOCK=config["block_vocab"],
        WRITE_SUPPORT_LOGPROBS=write_support,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


def support_pack(
    processed_logits,
    num_sampled,
    sampled_token_ids,
    *,
    max_support: int,
    return_support_logprobs: bool = False,
):
    """Allocate outputs and call :func:`support_pack_out`.

    Returns ``(token_ids [B, K] int32, counts [B] int32, overflow [B] int32,
    logz [B] f32, sampled_logprob [B] f32, support_logprobs [B, K] f32 | None)``.
    """

    batch = processed_logits.shape[0]
    device = processed_logits.device
    out_token_ids = torch.empty((batch, max_support), device=device, dtype=torch.int32)
    out_counts = torch.empty((batch,), device=device, dtype=torch.int32)
    out_overflow = torch.empty((batch,), device=device, dtype=torch.int32)
    out_logz = torch.empty((batch,), device=device, dtype=torch.float32)
    out_sampled_logprob = torch.empty((batch,), device=device, dtype=torch.float32)
    out_support_logprobs = (
        torch.empty((batch, max_support), device=device, dtype=torch.float32)
        if return_support_logprobs
        else None
    )
    support_pack_out(
        processed_logits,
        num_sampled,
        sampled_token_ids,
        out_token_ids=out_token_ids,
        out_counts=out_counts,
        out_overflow=out_overflow,
        out_logz=out_logz,
        out_sampled_logprob=out_sampled_logprob,
        out_support_logprobs=out_support_logprobs,
        max_support=max_support,
    )
    return out_token_ids, out_counts, out_overflow, out_logz, out_sampled_logprob, out_support_logprobs


def support_pack_reference(processed_logits, num_sampled, sampled_token_ids, *, max_support: int):
    """Dense PyTorch reference with the same outputs and contract."""

    if torch is None:
        raise RuntimeError("support_pack_reference requires PyTorch")
    logits = processed_logits.float()
    batch, vocab = logits.shape
    active = (num_sampled > 0).unsqueeze(1)
    keep = torch.isfinite(logits) & active
    counts = keep.sum(dim=1).to(torch.int32)
    overflow = (counts > max_support).to(torch.int32)
    token_ids = torch.full((batch, max_support), -1, device=logits.device, dtype=torch.int32)
    support_logprobs = torch.full(
        (batch, max_support), float("nan"), device=logits.device, dtype=torch.float32
    )
    masked = torch.where(keep, logits, torch.full_like(logits, -float("inf")))
    logz = torch.logsumexp(masked, dim=1)
    sampled_logprob = logits.gather(1, sampled_token_ids.long().unsqueeze(1)).squeeze(1) - logz
    for row in range(batch):
        ids = torch.nonzero(keep[row], as_tuple=False).flatten()[:max_support]
        token_ids[row, : ids.numel()] = ids.to(torch.int32)
        support_logprobs[row, : ids.numel()] = logits[row, ids] - logz[row]
    return token_ids, counts, overflow, logz, sampled_logprob, support_logprobs
