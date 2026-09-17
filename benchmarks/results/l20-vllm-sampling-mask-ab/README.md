# vLLM sampling-mask serving A/B on L20: native vs bitmap mask vs compact mask

> **Status (2026-09-18): independent replication, not a novel fix.** Upstream
> [vllm-project/vllm#54901](https://github.com/vllm-project/vllm/pull/54901) (merged 2026-09-04,
> shipped in 0.29.1) diagnosed the same host-side `np.unpackbits` bottleneck and landed the same
> `top_k`-bounded compact layout, sized by the batch's largest `top_k`. This work was done against
> v0.29.0 without knowledge of that PR. What this artifact adds is a replication on a different GPU
> family (L20 vs GB200), host, and models, plus the batch-invariant token equivalence and the
> bitmap-vs-bitmap repeatability control. The patch in `integrations/vllm/` applies to 0.29.0 only
> and should not be used on 0.29.1+. Upstream's own description notes that the bit-packed mask is
> still produced and copied to the host as the overflow fallback; that residual is measured in
> [`../l20-support-pack-path/`](../l20-support-pack-path/README.md).

End-to-end serving evidence for the compact sampling-mask layout
([patch](../../../integrations/vllm/vllm-v0.29.0-compact-sampling-mask.patch))
against vLLM v0.29.0's shipped bitmap layout, with the native engine (no mask)
as the control. This is the deployment-level counterpart of the operator-level
[support-pack path microbenchmark](../l20-support-pack-path/README.md).

**Claim boundary.** One L20 (SM89, 46 GB), one host (8 CPU cores), vLLM
0.29.0 with FlashInfer 0.6.18, two Qwen models, one rollout-shaped workload.
Numbers are medians over three interleaved rounds unless stated; min–max is
shown. No claim is made about other hosts, tensor parallelism, speculative
decoding, or workloads with variable completion lengths.

## Setup

- Engine: `vllm serve <model> --max-model-len 2048 --gpu-memory-utilization 0.85 --max-num-seqs 256 --seed 0`
  plus per-condition flags below. Every server condition is a fresh process.
- Workload: 128 deterministic prompts (~96 words), `max_tokens 256`, `ignore_eos`, so every
  condition generates exactly 32,768 tokens per round; concurrency 64; `temperature 1.0`,
  `top_k 50`, `top_p 0.95`; per-request `seed`. Warm-up of 16 prompts per request condition.
- Request conditions: `gen` (no logprobs) and `logprobs` (`logprobs=1`). Rounds alternate the
  order of request conditions.
- API: `/inference/v1/generate` (token in, token out, non-streaming) is the only route that
  returns `sampling_mask` in 0.29.0 and is the shape an RL rollout collector uses; the
  streaming `/v1/completions` run adds inter-token latency.
- Harness: [`scripts/measure_vllm_feature_cost.py`](../../../scripts/measure_vllm_feature_cost.py);
  aggregation: [`scripts/summarize_feature_cost.py`](../../../scripts/summarize_feature_cost.py).

| Server condition | Flags / env | What it measures |
| --- | --- | --- |
| `native` | — | control; FlashInfer fused sampler active |
| `mask_bitmap` | `--return-sampling-mask --logprobs-mode processed_logprobs`, `VLLM_SAMPLING_MASK_COMPACT=0` | upstream 0.29.0 behaviour (bitmap + host unpack) |
| `mask_compact` | same flags, `VLLM_SAMPLING_MASK_COMPACT=1` | patched layout |
| `native_fi_off` | `VLLM_USE_FLASHINFER_SAMPLER=0` | cost of the top-k/top-p + Gumbel fallback alone |
| `native_processed` | `--logprobs-mode processed_logprobs` | cost of processed-logprobs mode alone |

Software: Python 3.12.3, vLLM 0.29.0, PyTorch 2.13.0+cu130, Triton 3.7.1, FlashInfer 0.6.18
(sampler JIT-built with CUDA 13.0 nvcc); driver 580.159.04. Repository commits `6d0bbc2`
(main campaign) and `ef2eaa4` (batch-invariant equivalence and decomposition), clean trees.

## Results

Tables are generated into [`tables.md`](tables.md) from the raw files; the headline rows:

### Qwen2.5-0.5B-Instruct, generate API

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e median | mask mean size |
| --- | --- | ---: | ---: | ---: | ---: |
| `native` | `gen` | 18,566 (17,886–18,684) | 1.000x | 865 ms | — |
| `native` | `logprobs` | 13,403 (12,846–13,458) | 0.722x | 1,203 ms | — |
| `mask_bitmap` | `gen` | 1,582 (1,578–1,586) | **0.085x** | 10,252 ms | 14.7 |
| `mask_bitmap` | `logprobs` | 1,497 (1,486–1,509) | 0.081x | 10,838 ms | 15.0 |
| `mask_compact` | `gen` | 14,323 (14,120–14,649) | **0.771x** | 1,098 ms | 14.7 |
| `mask_compact` | `logprobs` | 10,076 (9,928–10,379) | 0.543x | 1,561 ms | 14.7 |

### Qwen3-4B, generate API

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e median | mask mean size |
| --- | --- | ---: | ---: | ---: | ---: |
| `native` | `gen` | 3,538 (3,250–3,540) | 1.000x | 4,608 ms | — |
| `native` | `logprobs` | 3,338 (3,323–3,345) | 0.944x | 4,859 ms | — |
| `mask_bitmap` | `gen` | 1,583 (1,526–1,585) | **0.447x** | 10,282 ms | 3.2 |
| `mask_bitmap` | `logprobs` | 1,484 (1,460–1,493) | 0.419x | 10,906 ms | 3.2 |
| `mask_compact` | `gen` | 3,418 (3,177–3,476) | **0.966x** | 4,680 ms | 3.1 |
| `mask_compact` | `logprobs` | 3,238 (3,211–3,266) | 0.915x | 4,969 ms | 3.2 |

### Qwen2.5-0.5B-Instruct, streaming completions API (inter-token latency)

| Server | Request | tok/s | ITL median | ITL mean | tokens per SSE chunk |
| --- | --- | ---: | ---: | ---: | ---: |
| `native` | `gen` | 17,954 | 3.21 ms | 3.27 ms | 1.00 |
| `native` | `logprobs` | 14,948 | 34.74 ms | 33.93 ms | 8.37 |
| `mask_bitmap` | `gen` | 1,591 | 39.61 ms | 39.55 ms | 1.00 |
| `mask_compact` | `gen` | 14,882 | 3.56 ms | 3.97 ms | 1.01 |
| `mask_compact` | `logprobs` | 12,117 | 26.07 ms | 28.02 ms | 5.54 |

### Decomposition on the native engine (0.5B, generate API)

| Server | `gen` tok/s | `logprobs` tok/s |
| --- | ---: | ---: |
| `native` | 18,360 | 13,128 |
| `native_fi_off` | 18,599 | 13,311 |
| `native_processed` | 18,656 | 13,348 |

## Reading the numbers

1. **The bitmap mask path is a host-side floor, independent of the model.** Both the 0.5B
   and the 4B model settle at ~1.58K tok/s (~10.3 s per round) with the mask on, matching
   the 35–70 ms per step of `np.unpackbits` + `np.nonzero` at batch 32–128 in the
   microbenchmark. Every request pays it, including `gen` requests that never read the mask.
2. **The compact layout removes almost all of it.** 0.5B: 0.085x → 0.771x of native (9.1x
   faster than bitmap); 4B: 0.447x → 0.966x of native (2.2x faster than bitmap). Streaming
   ITL on 0.5B goes from 39.6 ms back to 3.56 ms (native 3.21 ms).
3. **Disabling the FlashInfer sampler is not the expensive part here.** `native_fi_off` and
   `native_processed` are within noise of `native` on this GPU/batch, so the remaining
   0.77x on the 0.5B model is mask work proper: the pack kernel, the D2H copy, the CSR
   slicing in the scheduler, and JSON serialisation of ~15 IDs per generated token in the
   API server. On the 4B model, where each decode step is GPU-bound, that residual is 3%.
4. **`logprobs=1` is its own compatibility cost on the API path, not in the engine.** The
   engine-side fused `_topk_log_softmax_kernel` is cheap; the 0.72x on the 0.5B model comes
   from per-token logprob objects being built and serialised on the API server (visible as
   8 tokens coalesced per streaming chunk on the native server). On the 4B model it is 0.94x.

## Equivalence

Kernel level: the compact and bitmap kernels produce identical CSR lists for the same
processed logits ([tests](../../../tests/test_support_pack_gpu.py),
[patched-vLLM test](../../../tests/test_vllm_compact_sampling_mask_patch.py),
and the assertion in every microbenchmark trial).

Serving level, per-request seeds, `VLLM_BATCH_INVARIANT=1` (`raw/qwen25-05b-bi-*`):

| Comparison | token sequences identical | masks identical |
| --- | ---: | ---: |
| bitmap vs compact, `gen` | 128 / 128 | 123 / 128 |
| bitmap vs compact, `logprobs` | 128 / 128 | 124 / 128 |
| bitmap run A vs bitmap run B, `gen` (control) | 128 / 128 | 123 / 128 |

Two runs of the *unpatched* bitmap server disagree on as many masks (5 of 128) as
bitmap-vs-compact does under the same protocol, so this experiment does not attribute the
remaining five differences to the compact representation. Observed properties of those
differences, without a traced cause: all at position 0 (the prefill step) or at a
concurrency-wave boundary; 1–5 tail tokens next to the top-p cut-off; for the same request
they go in opposite directions between the `gen` and `logprobs` runs; the sampled token was
inside both masks in every case. A plausible mechanism is that the top-p tail depends on the
batch it was computed in (`apply_top_k_top_p` uses a Triton kernel at batch >= 8 and a
sort-based PyTorch path below, which `VLLM_BATCH_INVARIANT` does not cover), but that has not
been verified per request. Without batch-invariant mode, token sequences
themselves diverge between any two servers after a few dozen tokens
(`raw/*equivalence*.json`), which is expected and unrelated to the mask layout.

## Files

- `raw/qwen25-05b-generate.json`, `raw/qwen3-4b-generate.json`, `raw/qwen25-05b-completions.json` — main campaign (commit `6d0bbc2`).
- `raw/qwen25-05b-decomp-generate.json`, `raw/qwen25-05b-bi-generate.json`, `raw/qwen25-05b-bi-equivalence-*.json`, `raw/qwen25-05b-bi-bitmap-{a,b}.json`, `raw/bitmap-a-vs-b.json` — commit `ef2eaa4`.
- `raw/*equivalence*.json` — comparer output; `summary.json` — aggregate with raw-file hashes; `tables.md` — generated tables.
- Server logs for every condition are kept on the measurement host (`~/inference/results/campaign*-*/`), not checked in.
