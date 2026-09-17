# Where the rest of the sampled-token logprob cost sits after the flat response (L20)

Follow-up to [`../l20-flat-token-logprobs/`](../l20-flat-token-logprobs/README.md). With the flat
`token_logprobs` response in place, a Qwen2.5-0.5B server still runs at 0.80–0.87x of plain
generation at concurrency 64–512 (4B: 0.99x). This artifact bounds the remaining layers by
switching each one off, without proposing any of the switches as a feature.

Workload as before (256-token rollouts, `ignore_eos`, `/inference/v1/generate`,
`return_token_logprobs`), vLLM 0.29.0 + #54901, commit `598db2d`, clean tree.

## 1. Engine-side kernels (env-gated experiment patch)

`vllm-main-exp-logprob-engine-skips.patch` adds two switches to `compute_topk_scores`:
`VLLM_EXP_SKIP_RANKS=1` returns zero ranks instead of running `_ranks_kernel` (a third
full-vocabulary pass per row), and `VLLM_EXP_SKIP_LOGPROBS_ENGINE=1` returns zero scores and ranks
without launching any logprob kernel, leaving the D2H, IPC, output-processor and response path
intact. (The 2,650-byte responses confirm the switch engaged: zero logprobs serialise shorter.)

| Model / conc | `gen` | `token_logprobs` | + skip `_ranks_kernel` | + skip all logprob kernels |
| --- | ---: | ---: | ---: | ---: |
| 0.5B / 256 | 23,301 ± 757 | 18,516 ± 442 (0.79x) | 18,798 ± 285 (0.81x) | 19,473 ± 131 (0.84x) |
| 0.5B / 64 | 18,467 ± 414 | 16,373 ± 62 (0.89x) | 16,002 ± 801 | 16,632 ± 35 |
| 4B / 64 | 3,527 ± 138 | 3,491 (0.99x) | 3,493 | 3,527 |

Skipping `_ranks_kernel` is within noise everywhere; skipping every engine-side logprob kernel is
worth at most ~5 points (0.5B, conc 256) and nothing at conc 64 or on the 4B model. **A rank-less
or fused logprob kernel is not a serving improvement on this stack and is not pursued.**

## 2. Per-token detokenization in the output processor (`--skip-tokenizer-init`)

| 0.5B / conc | `gen` API CPU | `token_logprobs` tok/s, API CPU | + skip tokenizer tok/s, API CPU |
| --- | ---: | ---: | ---: |
| 256 | 0.36 | 19,314, **0.73** | 19,367, **0.39** |
| 64 | 0.28 | 16,665, 0.60 | 16,402, 0.38 |

`LogprobsProcessor` detokenises every logprob entry (`convert_ids_list_to_tokens` +
`_verify_tokens`) even though `/inference/v1/generate` never emits decoded tokens (it prints
`token_id:<id>`). With the flat response that detokenisation is nearly all of the API server's
remaining per-token CPU (0.73 → 0.39 of a core, back to the `gen` level) — **but throughput does not
move**, because the API server is no longer the limiter.

## 3. What is left, and where

With objects gone (PR 1) and kernels and detokenisation ruled out, the 0.5B gap that remains
(19.5K vs 23.3K at conc 256, ~16%; ~1% on 4B) sits on the **engine-core process**, which is pinned
at ~1.0 core with or without logprobs: `LogprobsTensors.tolists()` (three `.cpu().numpy()` arrays
per step), msgpack encoding of `LogprobsLists`, and the scheduler's per-request slicing of those
lists every step. That is the "sampled-score-only transport" target: one float per token, one
array per step, no token-id or rank arrays for requests that only asked for the sampled score. Its
upper bound is ~16% on a CPU-bound 0.5B host and ~1% on a GPU-bound 4B; it has not been built.

Separately, skipping logprob detokenisation on `/inference/v1/generate` is a throughput-neutral
CPU saving (~0.35 core on this host) that only matters where the API server is the bottleneck
(several engines behind one frontend); it is small and independent of PR 1.

## Files

- `raw/engine-skips-*.json`, `raw/skip-tokenizer-*.json`; `summary.json`, `tables.md` generated.
- `vllm-main-exp-logprob-engine-skips.patch` — the experiment switches (not a feature).
