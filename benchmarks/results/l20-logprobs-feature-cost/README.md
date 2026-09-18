# Where `logprobs=1` costs on the RL rollout path (L20, vLLM 0.29.0 + #54901)

Follow-up to [`../l20-vllm-sampling-mask-ab/`](../l20-vllm-sampling-mask-ab/README.md). With the
sampling mask solved upstream by vLLM #54901, the largest remaining feature cost on a small model
is `logprobs=1`: 0.71x native on Qwen2.5-0.5B (0.94x on Qwen3-4B). This artifact locates it.

Workload as before: 128 prompts x 256 tokens (`ignore_eos`), concurrency 64, `/inference/v1/generate`,
`temperature 1.0, top_k 50, top_p 0.95`, 3 interleaved rounds; commit `61c2319`, clean tree.

## 1. Detokenization is not the cost

`--skip-tokenizer-init` makes the token-in/token-out server skip every per-token
`convert_ids_list_to_tokens` / `_verify_tokens` call in `LogprobsProcessor`. It changes nothing:

| Server | `gen` tok/s | `logprobs` tok/s |
| --- | ---: | ---: |
| `native` | 18,544 | 13,188 (0.711x) |
| `native_skip_tokenizer` | 18,612 | 13,379 (0.721x) |
| `mask_upstream` | 15,532 | 11,301 (0.609x) |
| `mask_upstream_skip_tokenizer` | 16,272 | 11,253 (0.607x) |

## 2. The cost is API-server CPU, three quarters of it

Per-process CPU time (`/proc/<pid>/stat`, utime+stime) around one round on the native server,
two repetitions each ([`scripts/measure_vllm_cpu_split.py`](../../../scripts/measure_vllm_cpu_split.py),
`raw/cpu_split.jsonl`):

| Round | wall | API server CPU | engine core CPU | tok/s |
| --- | ---: | ---: | ---: | ---: |
| `gen` | 1.78 s | 0.68 s (38% of a core) | 1.70 s (96%) | 18,474 |
| `logprobs=1` | 2.47 s | **1.96 s** (79%) | 2.06 s (83%) | 13,268 |
| `gen` | 1.76 s | 0.60 s | 1.68 s | 18,618 |
| `logprobs=1` | 2.44 s | **1.95 s** | 2.04 s | 13,452 |

For 32,768 generated tokens, `logprobs=1` adds ~1.3 s of API-server CPU (~40 us per token) and
~0.35 s of engine-core CPU (~11 us per token). On the `gen` rounds the engine core is already the
ceiling at ~96% of one core, so the extra engine-side work also shows up in wall time; the
API-side work is the larger component.

What runs per token on the API side with `logprobs=1` (vLLM 0.29.0, `vllm/v1/engine/logprobs.py`
and `vllm/entrypoints/scale_out/token_in_token_out/serving.py`): msgpack decode of the logprob
arrays, `.tolist()` of ranks/logprobs/token ids, `append_logprobs_for_next_position` building a
`dict[int, Logprob]` with two `Logprob` dataclasses (sampled token + top-1), then
`_create_tokens_logprobs` building a `ChatCompletionLogProbsContent` Pydantic object with a nested
`ChatCompletionLogProb` list, then JSON serialisation of those objects. The engine-side share is
`LogprobsTensors.tolists()` (three D2H arrays per step) and the IPC encode.

None of that is required by an RL trainer, which consumes one float per generated token. This
points at an output-contract change (a flat per-request `token_logprobs` float array on the
token-in/token-out route, bypassing the OpenAI-style objects) rather than a kernel change; the
kernel-side fused `_topk_log_softmax_kernel` upstream is already cheap. Not implemented here.

## 3. Upstream #54901's residual mask cost (path microbenchmark)

Same protocol as [`../l20-support-pack-path/`](../l20-support-pack-path/README.md) with a verbatim
copy of main's `_compact_sampling_mask_kernel` + `tolists` added
(`../l20-support-pack-path/raw-with-main.json`). Main still produces and copies the bitmap as its
overflow fallback and slices rows in a Python loop:

| Batch | v0.29.0 total | main (#54901) total | compact-only total | D2H bytes main / compact-only |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 4.52 ms | 0.113 ms | 0.117 ms | 154 KB / 2 KB |
| 32 | 17.78 ms | 0.147 ms | 0.120 ms | 614 KB / 8 KB |
| 128 | 70.90 ms | 0.315 ms | 0.208 ms | 2.5 MB / 33 KB |
| 256 | 146.71 ms | 0.657 ms | 0.321 ms | 4.9 MB / 67 KB |

The remaining 0.3–0.35 ms per step at batch 128–256 is ~2% of a 4B decode step on this GPU and
would be a small, exact follow-up (copy the bitmap only when a row overflows; vectorise the
slice). It is not pursued unless a workload shows it matters.

## Files

- `raw/qwen25-05b-skiptok-generate.json` — skip-tokenizer A/B; `summary.json`, `tables.md` generated.
- `raw/cpu_split.jsonl` — per-process CPU split.
- `../l20-support-pack-path/raw-with-main.json` — three-way path microbenchmark.
