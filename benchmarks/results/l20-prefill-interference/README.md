# Decode-latency interference from long prefills under a token budget (L20, Qwen3-4B)

Branch B reproducer. vLLM's chunked prefill bounds the *tokens* a step may prefill
(`--max-num-batched-tokens`); the wall-clock cost of that chunk depends on the context it attends
to, so the decoders sharing the step see stalls the budget does not control. This artifact
measures that surface on one L20 before any scheduler change.

**Setup.** Qwen3-4B (`max_model_len 40960`, `--no-enable-prefix-caching`), one `vllm serve` per
budget. 8 background requests decode continuously (128-token random prompt, 4096 output tokens,
`ignore_eos`) on `/inference/v1/generate` with per-token arrival timestamps. Once all are past 64
tokens, 2 long requests (32,768 random tokens, 32 output tokens) are injected together. Reported
per budget: background inter-token latency in the 3 s before injection and during the injection
window (until both long requests finish), fraction of background tokens within 25/50 ms, the
largest single stall, the long requests' TTFT, and background tok/s during the window. Medians of
3 repeats (repeat-to-repeat spread is <2% in every cell). Commit `447136f`, clean tree.
Script: [`scripts/measure_prefill_interference.py`](../../../scripts/measure_prefill_interference.py).

## Result

| budget (tokens) | bg ITL p50 before | during p50 | during p99 | max stall | SLO 25 ms | SLO 50 ms | long TTFT | bg tok/s during |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 13.0 ms | 36.2 ms | 56 ms | 84 ms | 0.22 | 0.85 | 31.7 s | 221 |
| 128 | 12.9 ms | 36.8 ms | 56 ms | 82 ms | 0.20 | 0.82 | 15.3 s | 216 |
| 256 | 12.9 ms | 62.9 ms | 96 ms | 98 ms | 0.13 | 0.34 | 12.5 s | 136 |
| 512 | 13.0 ms | 97.9 ms | 150 ms | 150 ms | 0.21 | 0.24 | 10.0 s | 92 |
| 1024 | 13.0 ms | 147.5 ms | 288 ms | 294 ms | 0.35 | 0.35 | 9.5 s | 59 |
| 2048 | 13.0 ms | 28.8 ms | 568 ms | 568 ms | 0.33 | 0.51 | 9.1 s | 42 |
| 4096 | 13.0 ms | 26.3 ms | 1,078 ms | 1,078 ms | 0.27 | 0.67 | 8.9 s | 33 |
| 8192 | 13.0 ms | 26.0 ms | 2,042 ms | 2,042 ms | 0.23 | 0.80 | 9.0 s | 29 |

## Reading it

- **The undisturbed decode step is 13 ms; no fixed budget keeps it anywhere near that during a
  32k prefill.** Even 64-token chunks push the median to 36 ms, and the 25 ms SLO is met by at
  most a third of tokens under any budget.
- **Step time is strongly nonlinear in chunk size at this context**: 64 and 128 tokens both cost
  ~+23 ms on top of the decode step, 256 costs +50 ms, then it grows roughly linearly (512: +85,
  1024: +135, 2048: ~+555 as a single stall). A token budget cannot express this; a wall-clock
  budget can.
- **Small chunks are not free**: TTFT for the 32k request goes 9 s → 15 s → 32 s at 256 → 128
  → 64 tokens, i.e. prefill efficiency collapses ~3.5x, because each step re-reads the weights and
  the attention kernel runs with tiny query counts. A time-budget controller has to choose the
  *largest* chunk that fits the budget, per context bucket, not simply a small one.
- **Large chunks trade fewer, catastrophic stalls for a good attainment fraction**: at 8192,
  80% of background tokens meet 50 ms because prefill happens in only a few steps — but each of
  those steps stalls every decoder for 2 s. Which policy is "better" depends on whether the
  objective is attainment fraction or tail (p99/max); a controller should be explicit about that.
- Background throughput during the window falls from ~230 tok/s (small chunks) to 29 tok/s
  (8192), while total system work is roughly constant — the budget only moves who waits.

## What this motivates

A per-step prefill chunk chosen from a wall-clock budget derived from the active decoders' latency
target, using an online, self-calibrating step-time model keyed by (context bucket, decode batch,
chunk size) that is refreshed from each executed step. The controller must also be allowed to
choose "no prefill this step" when the budget cannot fit even the smallest useful chunk, and it
should be evaluated on p99/max stall *and* attainment *and* TTFT, across a context sweep (4k–32k),
against every fixed budget above. Not implemented in this artifact.

## Files

- `raw/qwen3-4b-32k.json` (budgets 512–8192), `raw/qwen3-4b-32k-small.json` (64–256): per-repeat
  before/during/after ITL distributions, SLO attainment, long-request TTFT/e2e, server commands.
- `summary.json` — the table above.
