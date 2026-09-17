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

## 3. Transport ladder: it is the per-request slicing, not the D2H

`vllm-main-exp-transport-ladder.patch` adds two more switches. `VLLM_EXP_SKIP_LOGPROB_SLICING=1`
keeps the async D2H and `LogprobsTensors.tolists()` but makes the scheduler skip
`logprobs.slice_request(...)` for every request (so nothing per-request is built, encoded, or
processed downstream). `VLLM_EXP_DROP_LOGPROBS_AFTER_D2H=1` additionally skips `tolists()`. The
endpoint tolerates the missing values (empty `token_logprobs`) under the experiment.
`raw/transport-ladder-*.json`, Qwen2.5-0.5B, `return_token_logprobs` requests:

| Concurrency | `gen` | PR 1 full path | + skip per-request slicing | + drop after D2H |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 18,609 ± 330 | 16,337 ± 300 (0.88x) | **17,925 ± 280 (0.96x)** | 17,946 (0.96x) |
| 256 | 22,448 ± 480 | 19,065 ± 160 (0.85x) | **22,431 ± 60 (1.00x)** | 22,147 (0.99x) |

With slicing skipped, API-server CPU falls to the `gen` level (0.29–0.37 of a core) and the
engine core is no faster or slower than `gen`. So the remaining sampled-token logprob cost is
**entirely** in the per-request path that starts at `slice_request`: one `LogprobsLists` (three
numpy arrays) per request per step, its msgpack encoding into `EngineCoreOutput`, and the output
processor's per-token work on the API side. The D2H copy and `tolists()` are free at this scale.

Upper bound for a sampled-score-only transport (one float per generated token, no token-id or rank
arrays, no per-token objects): **+15 points at concurrency 256, +8 at 64** on the 0.5B model.

## 4. What is left, and where

With objects gone (PR 1), kernels and detokenisation ruled out, and section 3 locating the rest
in the per-request `LogprobsLists` path, the sampled-score-only transport is the next change: for
requests that asked only for the sampled token's logprob, the scheduler should hand the output
processor one float per generated token instead of a three-array `LogprobsLists`, and the output
processor should keep a float list instead of a `FlatLogprobs`. Measured upper bound +15 points
(0.5B, c256), +8 (c64); ~1% on the GPU-bound 4B.

Separately, skipping logprob detokenisation on `/inference/v1/generate` is a throughput-neutral
CPU saving (~0.35 core on this host) that only matters where the API server is the bottleneck
(several engines behind one frontend); it is small and independent of PR 1.

## Files

- `raw/engine-skips-*.json`, `raw/skip-tokenizer-*.json`; `summary.json`, `tables.md` generated.
- `vllm-main-exp-logprob-engine-skips.patch` — the experiment switches (not a feature).
