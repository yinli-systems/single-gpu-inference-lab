# Phase 0: does a step-time predictor survive context shift? (L20, Qwen3-4B)

Branch B, first measured question. Candidate direction: risk-calibrated prefill scheduling
(choose the chunk from a wall-clock budget with a probabilistic, not point, cost estimate). Kill
criterion set in advance: if a simple point predictor already keeps its deadline-miss rate at the
target across distribution shift, there is nothing to calibrate. This artifact answers it with
traces, before any scheduler code.

## Instrumentation

`vllm-main-exp-iter-trace.patch` (env `VLLM_EXP_ITER_TRACE=<file>`, with
`--enable-logging-iteration-details`) appends one JSON line per engine iteration: prefill
requests/tokens and the KV depth they attend to (`ctx_kv_max`), decode batch and its KV depth,
and the engine's iteration timer. Because vLLM overlaps execution with scheduling, the timer's own
`ms` is only the residual wait; the decode-facing step time used everywhere below is the gap
between consecutive **result-ready instants** (`t + ms`), which is the cadence decoders see.

Workload: the interference reproducer ([`../l20-prefill-interference/`](../l20-prefill-interference/README.md)),
8 background decoders + 2 injected long prefills, 2 repeats per cell; cells = budget
{512, 2048, 8192} × long context {4k, 8k, 16k, 32k} at 8 decoders, plus decoders {4, 16, 32} at
16k. Commit `f7f4f22`, clean tree. Analysis: [`scripts/analyze_step_cost_traces.py`](../../../scripts/analyze_step_cost_traces.py).

## 1. The same token budget is a different workload at every context

| budget | long context | during p50 | during p99 | max stall | SLO 25 ms | long TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 4k | 15.6 ms | 56 ms | 56 ms | 0.67 | 0.65 s |
| 512 | 8k | 21.3 ms | 73 ms | 73 ms | 0.51 | 1.46 s |
| 512 | 16k | 62.0 ms | 108 ms | 112 ms | 0.35 | 3.68 s |
| 512 | 32k | 97.0 ms | 149 ms | 151 ms | 0.21 | 10.03 s |
| 2048 | 4k → 32k | 14.8 → 28.5 ms | 190 → 568 ms | | 0.87 → 0.32 | 0.67 → 9.15 s |
| 8192 | 4k → 32k | 14.9 → 26.0 ms | 519 → 2,042 ms | | 0.96 → 0.24 | 0.65 → 9.05 s |

(Undisturbed decode ITL is 13.0 ms in every cell.) With a 512-token budget the median decode
interference grows 6.2× from 4k to 32k and 25 ms SLO attainment collapses from 67% to 21%.

## 2. Why: an equal-size chunk gets more expensive as its KV-read depth grows

Prefill-step time from the engine trace, bucketed by the depth the chunk attends to
(`bg8-L32768`; decode batch held at 8, chunk size constant within a row):

| chunk tokens | 0–4k | 4–8k | 8–16k | 16–24k | 24–32k |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 504 | 51 ms | 82 | 95 | 110 | 136 |
| 2040 | 190 ms | 275 | 348 | 450 | 529 |

Decode batch size (4 / 8 / 16 / 32 decoders at 16k) changes these by <5%. The cost coordinate the
token budget lacks is the KV read, not the batch.

## 3. Predictors under shift (static fit on 4k/8k cells, evaluated on 16k/32k)

Least squares on scheduler-visible features. `tokens-only` = `[1, gen_reqs, ctx_tokens, ctx_reqs]`;
`+KV` adds `ctx_tokens × ctx_kv_max`. Residuals on prefill steps (`phase0-static-fit.json`):

| cell | tokens-only p50 / p95 | +KV p50 / p95 | false-safe @100 ms, tokens-only | +KV | +KV + calibration q95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bg8-L16384-chunk512 | +18.7 / +48.2 ms | +13.4 / +36.5 | 8% | 8% | 8% |
| bg8-L32768-chunk512 | +48.8 / +88.7 | +37.2 / +69.1 | **56%** | **56%** | **56%** |
| bg8-L32768-chunk2048 | +153 / +359 | +120 / +278 | — | — | — |

A static fit under-predicts systematically once the context leaves the calibration range, and a
static residual quantile learned at 4–8k (+22 ms) does nothing about it: at 32k every 512-token
step is predicted safe for 100 ms and 56% of them miss. The shift is a **bias**, not variance.

## 4. Online predictor (fit on the trace prefix, predict the next prefill step)

Each prefill step is predicted from a model refitted on all iterations seen so far; the margin is
the 95th percentile of the *online* residual history. `bg8-L32768-chunk512`:

| model | online bias p50 | p95 | p99 | false-safe @100 ms (share deemed safe) | + online q95 margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| tokens-only | +18.4 ms | +50.7 | +58.9 | 55% (92%) | 2% (16%) |
| tokens + KV | **+0.1 ms** | +28.0 | +33.6 | 24% (49%) | **0% (25%)** |

At 16k/512 the same pattern holds (tokens-only 9% → +KV 9% → +margin 0%, with 45–73% deemed safe);
at 32k/2048 even the +KV model keeps a +19 ms online bias, so the linear form under-fits large
chunks (superlinear in chunk × depth).

## Conclusions for the direction

1. The kill criterion is **not** met: a point predictor's deadline-miss rate is far above target
   under context shift (55% at 32k), so the problem is real.
2. Half of it is representation: with the KV-read coordinate and online refitting the bias
   disappears (+18 → +0.1 ms). A scheduler state without KV depth cannot be fixed by any margin.
3. The other half is tail risk: the remaining residual (p95 +28 ms at 32k) still yields 24%
   false-safe decisions; an online residual-quantile margin brings that to 0% at the cost of
   halving the prefill steps deemed safe (49% → 25%). That conservatism is the quantity a
   risk-calibrated controller trades against goodput — and the quantity to report.
4. Model form matters at large chunks; the next iteration should try cost ∝ chunk × (a + b·depth)
   with per-bucket online residuals rather than one global quantile.

Not done here: any scheduler change. The next step is an env-gated controller in the scheduler
(chunk = largest C with predicted(C, state) + q_α(bucket) ≤ decode slack, K = 0 allowed),
evaluated against every fixed budget above on the same reproducer with violation rate at α,
p99/max stall, TTFT and prefill progress.

## Files

- `raw/*.jsonl` — 21 per-iteration traces (cell = `bg<B>-L<ctx>-chunk<budget>`); `raw/*.json` — the
  interference harness outputs for the same runs.
- `phase0-static-fit.json` — static-fit residuals, KV-bucket table, deadline tables.
- `vllm-main-exp-iter-trace.patch` — the tracer (experiment only).
