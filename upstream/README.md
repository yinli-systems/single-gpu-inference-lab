# Upstream optimizations (2026-09)

Changes aimed at vLLM and Transformers, kept here as patches with the harnesses and raw results
that back each claim. Every row separates what was **measured** from what is **not yet known**.

| Directory | Target | What it changes | Measured | Equivalence evidence | Status |
| --- | --- | --- | --- | --- | --- |
| [`vllm-57442-sampled-logprobs/`](vllm-57442-sampled-logprobs/) | vLLM `/inference/v1/generate` data path | Sampled-token logprob as one float per token end to end (scheduler → output processor → response), no per-token `Logprob` objects | Throughput with logprobs **0.67× → 0.95–0.96×** of no-logprobs (Qwen2.5-0.5B, c64/c256), **0.99×** (Qwen3-4B); response 41 KB → 6.8 KB per request (L20) | Max abs diff vs object path 0.0 over 16 seeded sequences, batch-invariant | [vllm#57442](https://github.com/vllm-project/vllm/pull/57442) open, rebased on main + lint fixed; restack onto #58181 (`GenerateLogProbs.sampled`) ready as `0002-*.patch` |
| [`vllm-58181-generate-logprobs/`](vllm-58181-generate-logprobs/) | vLLM generate API (Python + Rust frontends, derender) | `GenerateLogProbs` with integer `token_id` and `rank` instead of `"token_id:N"` strings; derender fills `token`/`bytes` | Protocol change, not a speed claim | Rust `cargo test raw_generate` 9/9; Python 7/7 (CPU box, 0.29.0 + branch source) | [vllm#58181](https://github.com/vllm-project/vllm/pull/58181) open; shape agreed on [#57574](https://github.com/vllm-project/vllm/issues/57574) |
| [`vllm-kv-request-free-prefetch/`](vllm-kv-request-free-prefetch/) | vLLM `OffloadingConnector` | Request-free CPU→GPU prefetch primitive: resolve → capacity check behind a reserve → reserve blocks → existing async load → publish into the prefix cache; never evicts, no predictor, no ranking | Motivating measurement (line E): resume TTFT **−52…−58%** at 4k–16k prefixes when the reload is hidden | 12/12 new tests; offloading-connector suite failure set **identical** to unmodified main in the same env (no regressions) | Invited on [vllm#57103](https://github.com/vllm-project/vllm/issues/57103); branch pushed to the fork, PR not yet opened |
| [`vllm-beam-defer-materialization/`](vllm-beam-defer-materialization/) | vLLM offline beam search | Rank the 2·B² candidates from (parent length + 1, last token, cumulative logprob) and build token/logprob histories only for the B survivors | CPU beam logic per step, B=32 / 4096-token prompt: **52.8 → 1.5 ms**. End to end on L20 (Qwen2.5-0.5B): prompt 2048 B=32 **3.20 → 1.64 s (1.95×)**, prompt 128 B=32 **1.28 → 0.69 s (1.85×)** | 640 full `beam_search` runs byte-identical (CPU); on GPU under `VLLM_BATCH_INVARIANT=1` **bit-identical** tokens and scores, stock = stock = patched | Local; PR not opened |
| [`vllm-beam-grammar-mask/`](vllm-beam-grammar-mask/) | vLLM offline beam search + structured outputs | Unpack the grammar bitmask once to a bool array, test candidates by bit; build the explicit ID list only when ≤ 1024 (what the engine needs). Dense in-string masks (147k of 152k tokens allowed) no longer build a list and a set per beam per step | Grammar work per step at B=32 (real xgrammar, Qwen vocab): **206–217 → 6.3–6.9 ms**. End to end on L20, JSON schema: B=32 **15.6 → 4.1 s (3.76×)**, B=16 **2.29×**, B=8 **1.53×** | 56 real grammar states: identical drop decisions, engine `allowed_token_ids`, membership for every token ID; GPU batch-invariant run **bit-identical** | Local; PR not opened |
| [`transformers-stop-string/`](transformers-stop-string/) | Transformers `StopStringCriteria` preprocessing | Candidate positions via native `find` and first/last-unit indices instead of a Python triple loop, identical output (including the reversed-index semantics and empty-token edge case) | Real Qwen2.5 vocab (151,665 tokens): helper **4.6–12.2×**; full cache miss **2.1–3.8×** (1 stop: 636 → 298 ms; 8 stops: 3.18 → 0.84 s). The cache holds 8 stop-string sets, so varied per-request stop strings miss repeatedly | 336/336 differential checks (real vocab, exhaustive small alphabets, random, degenerate, str mode) | Measured; not submitted |

End-to-end beam numbers come from one A/B on the L20 ([`vllm-beam-e2e-l20/`](vllm-beam-e2e-l20/)):
stock vLLM 0.29.0 and 0.29.0 with both beam patches, **alternated** (stock, patched, stock,
patched), 5 timed calls per cell after a warm-up. Both patches were applied together; in the
unconstrained cells only the first one is exercised (no grammar), in the JSON cells both are.
Stock-vs-stock agreement (e.g. JSON B=32 15.6 vs 15.8 s) rules out drift. Without batch invariance
GPU runs are not bit-reproducible (stock ≠ stock in 11 of 12 cells), so equivalence was tested
separately under `VLLM_BATCH_INVARIANT=1`, where stock = stock exactly and patched = stock exactly.

## Where each piece was run

| Environment | Used for | Caveat |
| --- | --- | --- |
| L20 (vLLM 0.29.0 venv) | End-to-end beam A/B and GPU equivalence | The two beam patches were ported to 0.29.0, whose beam-search file is the same code as `main` apart from docstrings and main's newer abort block; the port is output-identical to the `main` patch on the 640-case harness |
| ParaCloud login node, CPU only | vLLM `main` pure-Python tests and CPU harnesses | Interpreter is vLLM 0.29.0 with `main` source on `PYTHONPATH` and 0.29.0's compiled extensions; ~104 existing connector tests fail there for version-skew reasons, so every claim is a diff against unmodified `main` in the same setup, never an absolute pass count |
| This Mac | Rust `cargo` tests for #58181, Transformers stop-string harness | `cargo` needs `openssl` vendored temporarily (no system OpenSSL) |

## Not known yet

- End-to-end gains on larger models (the 0.5B model maximizes the CPU share of a step; a 4B–7B
  model will show a smaller relative gain for the unconstrained beam cells).
- A server-backed run of the #58181 and prefetch tests on a real vLLM `main` build (CI is the
  first).
- Whether the Transformers stop-string change moves any end-to-end `generate()` latency beyond
  the first request that uses a new stop-string set.
