# Stage 1 notes: where the sampling-mask compatibility penalty actually sits (vLLM v0.29.0)

Working notes; the measured artifacts live under `benchmarks/results/`.

## Code path (v0.29.0, `vllm/v1/worker/gpu/sample/`)

`Sampler.__init__` sets `use_flashinfer = not return_sampling_mask and flashinfer_sampler_supported()`,
so `--return-sampling-mask` disables the FlashInfer fused sampler for the whole engine. Per batch,
`Sampler.sample` additionally leaves the fused path when any request is greedy, has an explicit seed,
or wants processed logprobs. The fallback is `apply_top_k_top_p` (Qrita-style Triton for batch >= 8,
sort-based PyTorch otherwise) followed by `gumbel_sample` over the full vocabulary.

With the mask on, every step then runs `SamplingMaskTensors.from_logits`: a Triton pass over the
processed logits that writes a `[B, ceil(V/8)]` uint8 bitmap plus counts, an async D2H copy of that
bitmap, and on the host (`tolists`) `np.unpackbits` over the full vocabulary followed by `np.nonzero`
to build the CSR `(token_ids, offsets)` that the scheduler slices per request.

The mask is exposed only on the token-in/token-out endpoint `/inference/v1/generate` (non-streaming
response, `choices[].sampling_mask`) and through `LLM(return_sampling_mask=True)`; the OpenAI
completions/chat routes never return it. The feature requires `temperature > 0` and `top_k > 0`
per request, and rejects speculative decoding, diffusion models, and custom logits processors
(`vllm/v1/engine/input_processor.py`).

## Consequence

`top_k > 0` bounds the surviving support at `top_k` tokens, yet the bitmap path moves `V/8` bytes per
row and unpacks `V` bits per row on the host every step regardless of `top_k`. The host stage is the
cost that scales with batch: see `benchmarks/results/l20-support-pack-path/` (156 ms per step at
B=256, V=151936 on the L20 host) and the serving A/B in `benchmarks/results/l20-vllm-sampling-mask-ab/`.

Logprobs (`logprobs=1`) on the native server are a separate, smaller cost: the fused upstream
`_topk_log_softmax_kernel` keeps the GPU side cheap, and the visible overhead is API-server CPU
(detokenizing/serialising per-token logprob objects), which shows up as coalesced streaming chunks
(`tokens_per_chunk` > 1 in the harness output) rather than slower engine steps.

## What the compact patch changes

`integrations/vllm/vllm-v0.29.0-compact-sampling-mask.patch` keeps the same keep-rule (finite
processed logits of rows that sampled) but emits ascending token IDs into `[B, K]` with `K` the next
power of two above the batch's largest `top_k` (read from host-side `SamplingStates.top_k.np`, so no
device sync), plus counts and an overflow flag. `tolists` becomes a ragged slice. Overflow raises
instead of truncating. `VLLM_SAMPLING_MASK_COMPACT=0` restores the bitmap layout, and batches whose
largest `top_k` exceeds `VLLM_SAMPLING_MASK_COMPACT_MAX_K` (default 1024) also fall back to it.

It does not re-enable the FlashInfer sampler; the mask still needs the processed logits that
`apply_top_k_top_p` materialises. Whether that remaining cost matters is the next measurement.
