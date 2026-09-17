# Top-Logprobs Correctness Notice — 2026-09

## Status

Fixed in commit `8a987d2`. The fused top-logprobs primitive
(`top_logprobs_out`, `vllm_top_logprobs_out`) returned NaN for any row whose
logits left an entire vocabulary tile at `-inf`. The A100 microbenchmark
artifact describes the pre-fix source and is annotated accordingly; the fixed
kernel was remeasured on the L20 in
[`benchmarks/results/l20-fused-top-logprobs-2026-09/`](../benchmarks/results/l20-fused-top-logprobs-2026-09/README.md).

## What Was Wrong

Both partial kernels computed a per-tile log-sum-exp as

```python
block_max = tl.max(values, axis=0)
block_sum = tl.sum(tl.exp(values - block_max), axis=0)
```

A tile whose every element is `-inf` has `block_max == -inf`, so the
subtraction evaluates `-inf - (-inf) = NaN` and `block_sum` is NaN. In the
reduce kernel that NaN survives `exp(block_max - global_max) * block_sum`
(`0 * NaN`) and enters the row's `log_denom`, so every returned logprob for the
row is NaN even when other tiles hold valid logits.

Any request that masks the vocabulary to a small allowed set — `allowed_token_ids`,
grammar-constrained decoding, or a hard `min_tokens`/EOS mask over a large
vocabulary — empties whole 1024-token tiles and hits this path.

## Reproduction

Measured on an NVIDIA L20 at the primitive's default tile sizes, eight live
tokens per row, batch 8:

| Vocab | Tiles per row | NaN rows before fix | NaN rows after fix | Max abs error after fix |
| ---: | ---: | ---: | ---: | ---: |
| 32000 | 32 | 8/8 | 0/8 | 4.77e-7 |
| 128256 | 126 | 8/8 | 0/8 | 2.38e-7 |
| 151936 | 149 | 8/8 | 0/8 | 2.38e-7 |

## Fix and Contract

- An all-`-inf` tile now contributes the log-sum-exp identity
  (`max = -inf`, `sum = 0`) by shifting with a finite value instead of the
  tile max.
- A row with **no** live logit has no normalizer. It returns NaN logprobs,
  matching `torch.log_softmax`, and does not disturb neighbouring rows. Its
  token IDs are unspecified. Callers that can produce such a row must mask it
  before or after the primitive; the primitive does not invent a distribution.
- `top_logprobs_out` and `vllm_top_logprobs_out` now reject tensors that are
  not on the same CUDA device or not contiguous, because the kernels use flat
  pointer arithmetic and would otherwise read or write the wrong memory
  silently.

Regression coverage lives in
[`tests/test_top_logprobs_masked_tiles_gpu.py`](../tests/test_top_logprobs_masked_tiles_gpu.py).

## Related Hardening In The Same Change

The opt-in vLLM top-k/top-p hook could synchronize the host through
`torch.all(...)` and `.item()` when a caller did not pass pre-resolved scalar
metadata. That fallback is now opt-in (`allow_host_sync=True`); otherwise a
CUDA parameter tensor is traced as ineligible with
`unresolved_<name>_requires_host_sync`. The vLLM installer already passes
scalars resolved from CPU-side state, so its fast path is unchanged. See
[`tests/test_l20_topk_topp_scalar_metadata_contract.py`](../tests/test_l20_topk_topp_scalar_metadata_contract.py).

## Provenance Rule Introduced

Benchmark artifacts pin the kernel-source hash. From this change on, a kernel
edit does not silently break or silently inherit an artifact: `summary.json`
must gain a dated `kernel_source_supersessions` entry naming the new hash and
stating whether performance was remeasured
(`tests/test_benchmark_protocol.py`).
