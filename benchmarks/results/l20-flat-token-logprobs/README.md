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

## Status

Exact, small, opt-in, composes with #54901. Prepared as an upstream PR (branch
`flat-token-logprobs` on the user's vLLM fork); submission waits on the submitter's DCO sign-off
and line-by-line review, which vLLM's contributing policy requires of the human author. No claim
about other endpoints (the OpenAI `/v1/completions` route keeps its schema) or about streaming.

## Files

- `raw/qwen25-05b-ladder.json`, `raw/qwen25-05b-mask-ladder.json`, `raw/qwen3-4b-mask-ladder.json` — runs; `summary.json`, `tables.md` generated.
- `raw/flat-vs-openai-style-equivalence.json` — exactness check.
- `vllm-main-flat-token-logprobs-experiment.patch` — the env-gated change against vLLM main (applies to 0.29.0 with offsets).
