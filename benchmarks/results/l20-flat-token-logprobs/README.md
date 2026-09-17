# Flat `token_logprobs` on the token-in/token-out endpoint (L20 upper-bound → exact feature)

Follow-up to [`../l20-logprobs-feature-cost/`](../l20-logprobs-feature-cost/README.md), which located
the `logprobs=1` cost on the API server (per-token `Logprob` dicts → per-token Pydantic objects →
JSON) rather than in the engine or detokenization. This artifact measures how much a flat output
contract recovers, and checks that it is exact.

**Change under test** (`vllm-main-flat-token-logprobs-experiment.patch`, ~25 lines, env-gated
`VLLM_GENERATE_FLAT_TOKEN_LOGPROBS=1`): on `/inference/v1/generate`, when a request sets
`logprobs=0` (sampled-token logprob only) and `flat_logprobs=True` (vLLM's existing engine-side
flat representation, `SamplingParams.flat_logprobs`), the response carries
`choices[].token_logprobs: list[float]` — one float per generated token, read directly from
`FlatLogprobs.logprobs` at each position's start index — and omits the OpenAI-style `logprobs`
object. Everything else (engine, sampler, mask path) is untouched.

Setup as in the sibling artifacts: vLLM 0.29.0 + #54901, 128 prompts × 256 tokens (`ignore_eos`),
concurrency 64, `top_k 50 / top_p 0.95`, 3 interleaved rounds, commit `f587fe4`, clean tree.

## Ladder on the native server, Qwen2.5-0.5B (tok/s, median; min–max)

| Request | plain server | server with flat emission |
| --- | ---: | ---: |
| `gen` | 18,613 (17,922–18,682) | 18,555 (17,910–18,566) |
| `logprobs=1` (earlier artifacts) | 12,939 — 0.70x | 13,338 |
| `logprobs=0` | 13,718 — 0.74x | 13,911 |
| `logprobs=0, flat_logprobs=True` | 13,490 — 0.73x | **16,596 — 0.89x** |

The existing `flat_logprobs` knob alone recovers nothing on this endpoint, because
`_create_tokens_logprobs` still materialises per-token objects from it; emitting the flat array
directly recovers 19 points of the 30-point gap. The remainder is the engine-core share
(`LogprobsTensors.tolists` + IPC, ~0.35 s per 32K tokens in the CPU split) plus the flat list
and JSON floats themselves.

## The RL contract: sampling mask + sampled-token logprob together

| Model | Server | `gen` | `logprobs=0, flat` | vs native `gen` |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-0.5B | `mask_upstream` (#54901) | 16,372 | 11,756 | 0.63x |
| Qwen2.5-0.5B | `mask_upstream` + flat emission | 15,991 | **14,045** | **0.75x** |
| Qwen3-4B | `mask_upstream` (#54901) | 3,486 | 3,308 | 0.94x |
| Qwen3-4B | `mask_upstream` + flat emission | 3,468 | **3,446** | **0.98x** |

(native `gen`: 18,613 for 0.5B, 3,534 for 4B, from the sibling artifacts.) On the 4B model the
combined mask + logprob rollout contract reaches 0.98x of plain generation; on the 0.5B model
0.75x, with the remaining cost now split between the mask frontend path and the engine-side
logprob transport.

## Exactness

`raw/flat-vs-openai-style-equivalence.json`: one server (`VLLM_BATCH_INVARIANT=1`, mask on),
16 seeded prompts, 64 tokens each, sequential requests; for each prompt the same request once with
`logprobs=0` (OpenAI-style objects) and once with `logprobs=0, flat_logprobs=True` (flat array).
Token sequences identical 16/16; `token_logprobs` length equals the generated length 16/16;
maximum absolute difference between the flat values and `content[i].logprob` is **0.0**; the flat
response carries no `logprobs` object.

## Final form: `return_token_logprobs` request field

The env switch was replaced by a request-level field (`GenerateRequest.return_token_logprobs`),
which sets `flat_logprobs=True` server-side, defaults `logprobs` to 0, keeps the OpenAI objects
when `logprobs > 0`, and rejects `stream=True` with a 400
(`vllm-main-return-token-logprobs.patch`, with four endpoint tests that pass against the
installed server). Re-measured with the field (`raw/*-final-field.json`, commit `89019c8`):

| Model | Server | `gen` | `logprobs=0` | `return_token_logprobs` |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-0.5B | `native` | 18,514 | 13,710 (0.74x) | **15,189 (0.82x)**; rounds 15,043–16,774 |
| Qwen2.5-0.5B | `mask_upstream` | 16,504 | 11,416 (0.62x) | **13,920 (0.75x)** |
| Qwen3-4B | `mask_upstream` | 3,431 | 3,315 (0.94x of native) | **3,458 (0.98x of native)** |

The 0.5B native median is lower than the env-gated run (15.2K vs 16.6K) with a wider spread
across rounds (15.0–16.8K); the mask-server and 4B numbers reproduce the earlier runs. The
conservative statement is the median: +11 points on 0.5B native, +13 points on 0.5B with the
mask, +4 points on 4B with the mask.

## Concurrency sweep (final field, commit `598db2d`)

Both models, native and `--return-sampling-mask` servers, concurrency 1 / 64 / 256 / 512
(prompts = 2 × concurrency, 4 at concurrency 1), 256 tokens each, 3 interleaved rounds. "API CPU"
and "core CPU" are the API-server and EngineCore processes' CPU seconds divided by the round's
wall time (fraction of one core, `/proc/<pid>/stat`), median over rounds. `raw/sweep/`.

| Model | Conc | Server | Request | tok/s median | ±sd | vs native gen | e2e p50 | e2e p99 | API CPU | core CPU | bytes/req |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5B | 1 | `native` | `gen` | 430 | 0 | 1.000x | 595 ms | 596 ms | 0.09 | 0.93 | 1,636 |
| 0.5B | 1 | `native` | `logprobs0` | 407 | 1 | 0.947x | 626 ms | 630 ms | 0.10 | 0.98 | 41,163 |
| 0.5B | 1 | `native` | `logprobs0_flat` | 408 | 1 | 0.948x | 626 ms | 629 ms | 0.09 | 0.96 | 41,163 |
| 0.5B | 1 | `native` | `token_logprobs` | 411 | 0 | 0.955x | 623 ms | 624 ms | 0.09 | 0.96 | 6,762 |
| 0.5B | 1 | `mask_upstream` | `gen` | 417 | 0 | 0.969x | 614 ms | 614 ms | 0.06 | 0.96 | 18,877 |
| 0.5B | 1 | `mask_upstream` | `logprobs0` | 395 | 0 | 0.919x | 647 ms | 647 ms | 0.08 | 0.98 | 63,268 |
| 0.5B | 1 | `mask_upstream` | `logprobs0_flat` | 395 | 0 | 0.918x | 647 ms | 647 ms | 0.09 | 0.98 | 63,268 |
| 0.5B | 1 | `mask_upstream` | `token_logprobs` | 397 | 0 | 0.923x | 644 ms | 644 ms | 0.08 | 0.98 | 29,480 |
| 0.5B | 64 | `native` | `gen` | 18,645 | 401 | 1.000x | 873 ms | 887 ms | 0.32 | 0.96 | 1,635 |
| 0.5B | 64 | `native` | `logprobs0` | 13,583 | 823 | 0.728x | 1,186 ms | 1,220 ms | 0.66 | 0.84 | 41,318 |
| 0.5B | 64 | `native` | `logprobs0_flat` | 13,023 | 231 | 0.698x | 1,201 ms | 1,250 ms | 0.64 | 0.83 | 41,323 |
| 0.5B | 64 | `native` | `token_logprobs` | 16,146 | 57 | 0.866x | 1,004 ms | 1,027 ms | 0.58 | 0.98 | 6,832 |
| 0.5B | 64 | `mask_upstream` | `gen` | 16,103 | 385 | 0.864x | 976 ms | 1,035 ms | 0.42 | 0.94 | 20,695 |
| 0.5B | 64 | `mask_upstream` | `logprobs0` | 11,652 | 257 | 0.625x | 1,375 ms | 1,403 ms | 0.67 | 0.80 | 58,219 |
| 0.5B | 64 | `mask_upstream` | `logprobs0_flat` | 11,177 | 342 | 0.599x | 1,370 ms | 1,514 ms | 0.65 | 0.80 | 58,477 |
| 0.5B | 64 | `mask_upstream` | `token_logprobs` | 13,686 | 270 | 0.734x | 1,150 ms | 1,206 ms | 0.59 | 0.95 | 24,816 |
| 0.5B | 256 | `native` | `gen` | 22,942 | 651 | 1.000x | 2,160 ms | 4,090 ms | 0.37 | 0.96 | 1,635 |
| 0.5B | 256 | `native` | `logprobs0` | 15,322 | 496 | 0.668x | 3,089 ms | 6,157 ms | 0.78 | 0.81 | 41,302 |
| 0.5B | 256 | `native` | `logprobs0_flat` | 15,438 | 64 | 0.673x | 3,105 ms | 6,173 ms | 0.77 | 0.81 | 41,291 |
| 0.5B | 256 | `native` | `token_logprobs` | 19,168 | 182 | 0.835x | 2,516 ms | 5,044 ms | 0.72 | 1.02 | 6,818 |
| 0.5B | 256 | `mask_upstream` | `gen` | 19,817 | 436 | 0.864x | 2,579 ms | 5,029 ms | 0.37 | 1.01 | 20,546 |
| 0.5B | 256 | `mask_upstream` | `logprobs0` | 12,976 | 564 | 0.566x | 3,836 ms | 7,600 ms | 0.77 | 0.84 | 58,200 |
| 0.5B | 256 | `mask_upstream` | `logprobs0_flat` | 12,934 | 249 | 0.564x | 3,645 ms | 7,468 ms | 0.75 | 0.80 | 58,074 |
| 0.5B | 256 | `mask_upstream` | `token_logprobs` | 16,410 | 285 | 0.715x | 3,166 ms | 6,253 ms | 0.73 | 1.00 | 25,035 |
| 0.5B | 512 | `native` | `gen` | 23,945 | 807 | 1.000x | 5,097 ms | 8,228 ms | 0.33 | 1.01 | 1,634 |
| 0.5B | 512 | `native` | `logprobs0` | 15,787 | 66 | 0.659x | 7,811 ms | 12,623 ms | 0.79 | 0.82 | 41,317 |
| 0.5B | 512 | `native` | `logprobs0_flat` | 15,583 | 133 | 0.651x | 7,691 ms | 12,731 ms | 0.77 | 0.83 | 41,307 |
| 0.5B | 512 | `native` | `token_logprobs` | 19,264 | 433 | 0.805x | 6,339 ms | 10,228 ms | 0.71 | 1.02 | 6,831 |
| 0.5B | 512 | `mask_upstream` | `gen` | 20,730 | 803 | 0.866x | 5,916 ms | 9,878 ms | 0.44 | 1.01 | 20,867 |
| 0.5B | 512 | `mask_upstream` | `logprobs0` | 13,728 | 64 | 0.573x | 9,240 ms | 14,862 ms | 0.80 | 0.83 | 58,579 |
| 0.5B | 512 | `mask_upstream` | `logprobs0_flat` | 13,767 | 278 | 0.575x | 9,079 ms | 14,841 ms | 0.78 | 0.84 | 58,315 |
| 0.5B | 512 | `mask_upstream` | `token_logprobs` | 16,984 | 144 | 0.709x | 7,261 ms | 11,915 ms | 0.75 | 1.01 | 24,807 |
| 4B | 1 | `native` | `gen` | 82 | 0 | 1.000x | 3,134 ms | 3,136 ms | 0.03 | 0.36 | 1,627 |
| 4B | 1 | `native` | `logprobs0` | 81 | 0 | 0.989x | 3,166 ms | 3,169 ms | 0.05 | 0.34 | 41,544 |
| 4B | 1 | `native` | `logprobs0_flat` | 81 | 0 | 0.989x | 3,167 ms | 3,171 ms | 0.04 | 0.34 | 41,544 |
| 4B | 1 | `native` | `token_logprobs` | 81 | 0 | 0.991x | 3,160 ms | 3,163 ms | 0.05 | 0.36 | 6,943 |
| 4B | 1 | `mask_upstream` | `gen` | 81 | 0 | 0.994x | 3,153 ms | 3,154 ms | 0.04 | 0.34 | 6,456 |
| 4B | 1 | `mask_upstream` | `logprobs0` | 80 | 0 | 0.983x | 3,185 ms | 3,190 ms | 0.05 | 0.31 | 42,430 |
| 4B | 1 | `mask_upstream` | `logprobs0_flat` | 80 | 0 | 0.982x | 3,189 ms | 3,193 ms | 0.05 | 0.36 | 42,430 |
| 4B | 1 | `mask_upstream` | `token_logprobs` | 80 | 0 | 0.985x | 3,181 ms | 3,184 ms | 0.04 | 0.31 | 9,739 |
| 4B | 64 | `native` | `gen` | 3,517 | 136 | 1.000x | 4,644 ms | 4,667 ms | 0.10 | 0.38 | 1,625 |
| 4B | 64 | `native` | `logprobs0` | 3,361 | 11 | 0.956x | 4,833 ms | 4,913 ms | 0.34 | 0.43 | 41,433 |
| 4B | 64 | `native` | `logprobs0_flat` | 3,342 | 9 | 0.950x | 4,850 ms | 4,923 ms | 0.33 | 0.39 | 41,388 |
| 4B | 64 | `native` | `token_logprobs` | 3,498 | 9 | 0.994x | 4,657 ms | 4,712 ms | 0.31 | 0.37 | 6,894 |
| 4B | 64 | `mask_upstream` | `gen` | 3,479 | 125 | 0.989x | 4,688 ms | 4,724 ms | 0.14 | 0.35 | 6,089 |
| 4B | 64 | `mask_upstream` | `logprobs0` | 3,287 | 17 | 0.935x | 4,888 ms | 5,104 ms | 0.34 | 0.39 | 42,207 |
| 4B | 64 | `mask_upstream` | `logprobs0_flat` | 3,330 | 13 | 0.947x | 4,850 ms | 4,977 ms | 0.35 | 0.35 | 42,203 |
| 4B | 64 | `mask_upstream` | `token_logprobs` | 3,419 | 12 | 0.972x | 4,738 ms | 4,811 ms | 0.30 | 0.39 | 9,472 |
| 4B | 256 | `native` | `gen` | 4,324 | 196 | 1.000x | 10,773 ms | 21,518 ms | 0.13 | 0.38 | 1,624 |
| 4B | 256 | `native` | `logprobs0` | 4,236 | 79 | 0.980x | 11,463 ms | 22,854 ms | 0.36 | 0.39 | 41,413 |
| 4B | 256 | `native` | `logprobs0_flat` | 4,213 | 86 | 0.974x | 11,360 ms | 22,942 ms | 0.39 | 0.38 | 41,414 |
| 4B | 256 | `native` | `token_logprobs` | 4,274 | 24 | 0.988x | 10,834 ms | 21,863 ms | 0.33 | 0.43 | 6,882 |
| 4B | 256 | `mask_upstream` | `gen` | 4,285 | 180 | 0.991x | 10,952 ms | 22,048 ms | 0.17 | 0.42 | 6,078 |
| 4B | 256 | `mask_upstream` | `logprobs0` | 4,181 | 14 | 0.967x | 11,766 ms | 23,488 ms | 0.39 | 0.43 | 42,245 |
| 4B | 256 | `mask_upstream` | `logprobs0_flat` | 4,015 | 82 | 0.928x | 11,621 ms | 23,328 ms | 0.33 | 0.43 | 42,233 |
| 4B | 256 | `mask_upstream` | `token_logprobs` | 4,240 | 30 | 0.980x | 11,144 ms | 22,197 ms | 0.30 | 0.44 | 9,531 |
| 4B | 512 | `native` | `gen` | 4,130 | 20 | 1.000x | 29,530 ms | 47,492 ms | 0.12 | 0.33 | 1,626 |
| 4B | 512 | `native` | `logprobs0` | 3,922 | 6 | 0.950x | 30,967 ms | 50,003 ms | 0.37 | 0.40 | 41,408 |
| 4B | 512 | `native` | `logprobs0_flat` | 3,917 | 7 | 0.948x | 31,026 ms | 50,140 ms | 0.36 | 0.39 | 41,410 |
| 4B | 512 | `native` | `token_logprobs` | 4,096 | 10 | 0.992x | 29,695 ms | 47,841 ms | 0.33 | 0.40 | 6,875 |
| 4B | 512 | `mask_upstream` | `gen` | 4,076 | 18 | 0.987x | 29,730 ms | 48,181 ms | 0.16 | 0.38 | 6,077 |
| 4B | 512 | `mask_upstream` | `logprobs0` | 3,877 | 6 | 0.939x | 31,305 ms | 50,663 ms | 0.39 | 0.41 | 42,313 |
| 4B | 512 | `mask_upstream` | `logprobs0_flat` | 3,867 | 4 | 0.936x | 31,408 ms | 50,827 ms | 0.38 | 0.42 | 42,289 |
| 4B | 512 | `mask_upstream` | `token_logprobs` | 4,059 | 9 | 0.983x | 30,019 ms | 48,462 ms | 0.34 | 0.43 | 9,492 |


Reading it:

- **0.5B, concurrency 64–512**: `return_token_logprobs` recovers 14–17 points over `logprobs=0`
  (0.73→0.87, 0.67→0.84, 0.66→0.81 of native) and 11–15 points on the mask server. The
  bottleneck moves back from the API server (0.78 of a core with objects, engine core at 0.81)
  to the engine core (1.0, API 0.72). Response size per request drops 41 KB → 6.8 KB.
- **The API server still spends ~0.35 core more than `gen` with the flat response** (0.72 vs 0.37):
  that is the `LogprobsProcessor` work per token (msgpack decode, `.tolist()`,
  `FlatLogprobs.append_fast` of ids/logprobs/ranks/decoded) which the flat response does not
  remove. It is the next removable layer (a sampled-score-only transport), not part of this change.
- **4B**: `logprobs=0` costs 2–5% at every concurrency; the flat field brings it to 0.99x
  (0.97–0.98x with the mask). p99 latency follows the same pattern.
- **Concurrency 1**: everything is within 5% and the 0.5B engine core already sits at ~0.95 of a
  core at batch 1 — per-step Python overhead is the floor there, and no output-path change moves it.
- `flat_logprobs` alone (engine knob, object response) never helps on this endpoint, at any
  concurrency or model: the objects are rebuilt from it.

## Status

Exact, small, opt-in, composes with #54901. Prepared as an upstream PR (branch
`flat-token-logprobs` on the user's vLLM fork); submission waits on the submitter's DCO sign-off
and line-by-line review, which vLLM's contributing policy requires of the human author. No claim
about other endpoints (the OpenAI `/v1/completions` route keeps its schema) or about streaming.

## Files

- `raw/qwen25-05b-ladder.json`, `raw/qwen25-05b-mask-ladder.json`, `raw/qwen3-4b-mask-ladder.json` — runs; `summary.json`, `tables.md` generated.
- `raw/flat-vs-openai-style-equivalence.json` — exactness check.
- `vllm-main-flat-token-logprobs-experiment.patch` — the env-gated change against vLLM main (applies to 0.29.0 with offsets).
