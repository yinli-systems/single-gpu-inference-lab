# When Token Budgets Lie: Request Geometry in Chunked LLM Prefill

*Technical report on the frozen artifact
[`benchmarks/results/l20-prefill-cost-geometry/`](../benchmarks/results/l20-prefill-cost-geometry/README.md)
(commit `17478ac`). One GPU, one model, one engine version; every number below is reproducible from
the raw traces in that directory.*

## Abstract

Deadline-aware chunked-prefill schedulers price the next iteration from an *aggregate* coordinate —
decode batch size, aggregate KV depth, candidate prefill tokens — and pick the largest prefill
budget that fits a latency bound. We ask whether that coordinate is sufficient on a real engine
(vLLM 0.29, Qwen3-4B, NVIDIA L20) and, where it is not, whether the missing information is worth
carrying into the scheduler. It is sufficient for the shifts usually worried about: context length
(≤16k → 32k), decode load (8 → 32 requests), skewed decode KV at equal total, CUDA-graph capture
boundaries, and fresh short-prompt bursts under default FCFS chunking. It is not sufficient when
several partially-prefilled requests with different KV depths share one iteration: at identical
decode batch, identical total cached context and identical prefill tokens, 1×2048 tokens costs up
to 1.9× more than 8×256, because the attention work is Σ qᵢ·kvᵢ, not (Σqᵢ)(Σkvᵢ). Six partition
series collapse onto one line against Σ qᵢ(kvᵢ + (qᵢ+1)/2). Replacing the aggregate
token × KV interaction with that per-request term reduces geometry-out-of-distribution prediction
error from 341 ms MAE / 1.2 s P95 to 8.3 ms / 17.6 ms, and on the live engine a 100 ms-deadline
controller carrying it delivers 49–180% more safe prefill progress than the same controller on the
aggregate coordinate at essentially zero violations. Against a hindsight-tuned oracle fixed budget
the gain is 5–12%; on FCFS short-prompt bursts it is nil. The contribution is a precise scope: the
representation, not the margin, is what fails, and it fails only for mixed-depth multi-request
steps.

## 1. Question and pre-registered gates

Two hypotheses were fixed before the discriminating experiments:

- **H1 (representation).** Steps with the same aggregate coordinate can differ materially in
  measured cost. Gate: ≥20–25% median difference at equal aggregate; ≥1.5× counts as headline.
- **H2 (control).** A cost model that sees per-request geometry improves a deadline-aware prefill
  controller. Gates: the geometry model must cut P95 under-prediction by ≥50% before any scheduler
  code; a live controller must reach ≤5–7% violations with ≥15% better TTFT or safe prefill
  progress than the strongest safe baseline, over ≥3 interleaved repeats.

Kill rule: if the fair aggregate model stays calibrated under shift (P95 positive residual <5 ms;
nominal 5% miss → ≤7% actual), the geometry hypothesis dies. The aggregate baseline was
deliberately the strongest one a SLOWeave-style scheduler could use, not the `(decode batch,
chunk)` pair.

## 2. Measurement contract

**Instrumentation.** Two per-iteration traces are written by patches to the installed engine
(never upstreamed): the *engine* trace (one row per scheduler iteration: prefill request count,
per-request chunk and KV depth, decode batch and KV depths, totals, the engine's `future.result()`
wait) and the *runner* trace (one row per executed batch: a CUDA event pair around the model
step, padded token count, cudagraph mode). The engine wait is never used as GPU time.

**Join.** Runner rows are joined to engine rows by *sequence*: the runner executes four warm-up
batches before the engine trace starts, after which token counts must match one-to-one. Over 28
cells the match fraction is 1.0000 (e.g. 1189/1189); the only unjoined row per file is the final
in-flight iteration. A nearest-timestamp join is wrong under async scheduling — the engine stamps
about one step after the runner starts the batch — and produced a spurious −25 ms
gap-versus-CUDA discrepancy at 2048-token chunks that vanishes under the sequence join.

**Agreement.** With the correct join, the host-observed result-ready gap equals the model-step
CUDA time at every step class (decode-only 13.2 / 13.2 ms; 512-chunk 74.9 / 73.8; 2048-chunk
248.0 / 248.0, corr 1.000). All costs below are CUDA time.

**Overhead.** Trace on vs off, three interleaved repeats each: throughput within ±0.6%, decode ITL
p50 +1.4%, p95 −0.6%, TTFT +0.1% — inside the <1% / <2% envelope set in advance.

**Environment.** NVIDIA L20 (46 GB, driver 580.159.04), vLLM 0.29.0 + backported #54901,
torch 2.13.0+cu130, Qwen3-4B bf16, `--max-model-len 40960 --max-num-seqs 64
--no-enable-prefix-caching`. Harness: B background decoders (128-token prompts, 4096 output
tokens) plus N injected long prefills; one fresh `vllm serve` per condition; conditions
interleaved; three repeats per cell. Commits, exact commands and per-run provenance blocks are in
the artifact.

## 3. Result 1 — same aggregate coordinate, different cost

![same aggregate geometry](../benchmarks/results/l20-prefill-cost-geometry/figures/same_aggregate_geometry.png)

*Figure 1. Left: model-step CUDA time against the aggregate KV read depth of the step for six
partitions of a fixed prefill budget (1024 tokens as 1×1024 / 2×512 / 4×256; 2048 as 1×2048 /
4×512 / 8×256), 8 decoders. Right: the same steps against Σ qᵢ(kvᵢ + (qᵢ+1)/2).*

Binned by aggregate KV (prefill + decode), with decode batch and prefill tokens held equal:

| aggregate KV | 1×1024 | 4×256 | ratio | 1×2048 | 8×256 | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4–8k | 110.4 ms | 85.4 | 1.29× | 229.3 | 162.0 | 1.42× |
| 8–12k | 138.7 | 91.9 | 1.51× | 285.0 | 169.2 | 1.68× |
| 12–16k | 163.1 | 100.4 | 1.62× | 336.1 | 178.9 | 1.88× |
| 16–20k | 179.8 | 105.5 | 1.70× | — | 181.8 | — |

H1's gate is met from the 8k bin upward. The explanation is not a kernel anomaly: each series
regresses on the proxy alone as `77–78 + 6.4–6.6·M` (1024-token budget) and `151–153 +
6.4–6.7·M` (2048), i.e. one slope within 4%, intercepts set by the token count. The aggregate
coordinate knows Σq and Σkv; the cost depends on Σ qᵢkvᵢ.

### What does *not* move the cost (controls)

| control | result |
| --- | --- |
| context 4k → 32k, 512-chunk | 47.7 → 105.1 ms (2.2×) — real, but an aggregate quantity |
| decode batch 4 → 32 at fixed prefill | 73.6 → 76.3 ms |
| decode KV balanced 8×4k vs skewed 4×7.9k+4×0.3k, equal total | 19.58 vs 19.48 ms (decode-only); 107.1 vs 106.4 ms (512-chunk at 12k) |
| CUDA-graph capture boundary C → C+1 (8, 16, 24) | ≤ +0.7 ms |

One aggregate subtlety worth recording: a single 32k decoder adds +36 ms to a 512-chunk prefill
step at every depth — 1.09 ms per 1k decode KV inside an eager mixed step, five times the 0.2 ms/1k
the same KV costs inside a graph-captured decode-only step — but it is linear in the aggregate
decode-KV sum and therefore not a geometry effect.

## 4. Result 2 — the aggregate predictor aliases geometry; a physical term fixes it

Models are ridge regressions (λ = 0.01 on standardised features) on prefill-containing steps.
**M0** (strongest aggregate coordinate): decode batch, aggregate decode KV, aggregate prefill KV,
prefill tokens, prefill tokens × aggregate prefill KV; plus a conservative P99 lookup table over
(decode batch, tokens/256, KV/4k). **M1**: M0 + geometry statistics (prefill count, KV max/var,
largest chunk, tokens × KV max, decode KV max/var). **M2**: aggregates + prefill count + eager
flag + padded tokens + Σ qᵢ(kvᵢ+(qᵢ+1)/2) *in place of* tokens × aggregate KV. Splits are by
cell, never random.

![OOD residuals](../benchmarks/results/l20-prefill-cost-geometry/figures/ood_residuals.png)

*Figure 2. P50/P95/P99 positive residual (actual − predicted) of each model on the four
distribution-shift splits; dashed 5 ms and dotted 10 ms are the pre-registered gate lines.*

| split (train → test) | model | MAE | P95 abs | P95 positive residual | nominal-5% rule, actual miss @100 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| context ≤16k → 32k | M0 | 3.3 | 12.5 | 11.3 | 0% |
| load B≤16 → 32 | M0 | 3.2 | 5.5 | 5.1 | 0% |
| **geometry: one prefill → 2/4/8** | M0 | **341.1** | **1208.1** | 0.0 (over-prices) | 4% of 4% admitted |
| | M1 | 132.3 | 527.4 | 24.9 | |
| | **M2** | **8.3** | **17.6** | 0.0 | 3% of 6% admitted |
| **geometry, reverse: 2/4/8 → one** | M0 | 37.0 | 126.0 | **126.0** | — |
| | M0 P99 lookup | 34.6 | 119.9 | 119.9 | **34%** |
| | **M2** | 5.5 | 17.2 | **9.1** | 0% |

Three things follow. (i) The kill rule for context and load shift fires: a fair aggregate model
is calibrated there, and an earlier Phase-0 claim of 56% false-safe at 32k — made with a predictor
lacking the KV term — is superseded. (ii) On unseen geometry the aggregate model is wrong by an
order of magnitude in whichever direction its calibration data pushes it: fit on single prefills
it over-prices multi-request steps by up to 1.2 s (and so starves them); fit on multi-request
steps it under-prices single prefills by 126 ms at P95, and a "safe" P99 table turns a nominal
≤1% miss target into 34% actual misses. (iii) Adding geometry statistics (M1) does not work: on
single-prefill data KV max ≡ KV sum and largest chunk ≡ tokens, the fit cannot separate the
duplicated features, and it extrapolates wrongly. The working form is one physically motivated
term replacing the wrong interaction — a representation change, not a feature count.

## 5. Result 3 — live controller on the real engine

An env-gated patch to the vLLM scheduler enumerates the in-progress and waiting prefills (remaining
tokens, KV depth) and the decoders, prices each candidate budget {64 … 8192} split equally over
the prefills through an exported cost model, and takes the largest budget with
`prediction + margin ≤ 100 ms` (smallest candidate if none). Both models were fit **only on
single-prefill steps** — the calibration data an operator realistically has. Fixed budgets run
through the same code path. Workload: 8 decoders + N × 16k prefills injected together; one fresh
server per run; interleaved; three repeats (every range ≤3% of its mean).

![live controller](../benchmarks/results/l20-prefill-cost-geometry/figures/live_controller.png)

*Figure 3. Safe prefill tokens per second (prefill tokens of steps under the deadline ÷ wall time
of the prefill phase), with the fraction of prefill steps over 100 ms.*

| N | controller | violations | prefill p95 | safe tok/s | TTFT |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | fixed 256 (strongest safe fixed) | 0.0% | 62 ms | 5,625 | 11.72 s |
| 4 | fixed 512 | 4.7% | 100 | 6,516 | 9.66 |
| 4 | M0 aggregate | 0.0% | 54 | 4,132 | 15.82 |
| 4 | **M2 geometry** | 0.0% | 80 | **6,171** | **10.65** |
| 8 | fixed 256 | 0.0% | 85 | 4,427 | 29.70 |
| 8 | fixed 512 | 10.7% | 103 | 5,966 | 19.77 |
| 8 | M0 aggregate | 0.0% | 78 | 1,702 | 77.02 |
| 8 | **M2 geometry** | 0.0% | 83 | **4,769** | **27.67** |

Geometry versus aggregate, everything else equal: **+49% / +180% safe progress and −33% / −64%
TTFT at zero violations.** The aggregate controller is not unsafe — it is starved: it prices the
4- and 8-request steps at several times their cost and settles on 64–128-token budgets. This is
the H2 test, and it is decisive.

### 5.1 Calibration round: is it just the margin?

Adding an online margin (the 95th percentile of the last 64 realised residuals per prefill-count
bucket, realised times attributed to their batch in `update_from_output`) and a finer budget grid
to every controller — and `fixed 384` to the baseline for the same reason:

| N | controller | violations | prefill p95 | safe tok/s | TTFT | final margin |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | fixed 384 (hindsight-tuned oracle fixed) | 0.0% | 90 | 5,887 | 11.25 | — |
| 4 | **M2 + online margin** | 0.3% | 93 | **6,592** | **10.04** | 2.8 ms |
| 4 | M0 + online margin | 0.0% | 68 | 5,914 | 11.20 | −59 ms |
| 8 | fixed 384 | 0.0% | 94 | 5,742 | 22.97 | — |
| 8 | **M2 + online margin** | 0.0% | 93 | **6,036** | **21.69** | 0.6 ms |
| 8 | M0 + online margin | 0.0% | 81 | 4,145 (3,704–4,378) | 31.7 (30.2–34.7) | −33 (−108 … +14) |

The online margin uses the slack (M2's p95 goes from 80–87 to 93 ms at ≤0.3% violations) and
brings M2 to +12% / +5% over the oracle fixed budget — still below the pre-registered 15%. The
control answers the reviewer's question: the same margin given to M0 becomes a −59 ms bias
correction at N=4 and is unstable at N=8 (the margin swings by 120 ms across repeats, TTFT
30–35 s), because M0's error grows with KV depth within a run and a per-count quantile cannot
follow it. **Calibration corrects residual error; it cannot recover information discarded by the
state representation.** A thin margin from the multi-prefill trace fit (q95 4 ms) violates 4–12%
live — margins have to be learned on the engine.

## 6. Where geometry does not matter (negative result)

Under vLLM's default FCFS chunking, multi-request prefill steps arise only when the budget exceeds
a request's remaining tokens. Bursts of 16 × 2048 or 32 × 1024 fresh prompts into 8 decoders,
fixed budgets run natively, controllers pricing the FCFS partition, 100 ms deadline:

![FCFS bursts](../benchmarks/results/l20-prefill-cost-geometry/figures/fcfs_bursts.png)

*Figure 4 (appendix). Safe prefill throughput on default-FCFS short-prompt bursts.*

| burst | best | controllers (M2 static / M2 online / M0 online) |
| --- | --- | --- |
| 16 × 2048 | native fixed 1024: 11,042 tok/s, TTFT 1.66 s, 0% viol., p95 98 ms | 9,910 / 9,891 / 9,826; TTFT 1.79–1.80 |
| 32 × 1024 | native fixed 1024: 11,317, TTFT 1.57 s, 0%, p95 95 ms | 10,221 / 10,205 / 10,147; TTFT 1.68 |

M0 ≡ M2 within 1%, and the native fixed budget beats every controller by ~10%: at a 100 ms
deadline the admissible budget (768) is below the prompt length, so each step holds one fresh
prompt at depth <2k — the single-prefill distribution both models were fit on — and the 13 ms
static margin keeps the controllers one grid step short of the deadline. This is what makes the
claim precise: **the aliasing is a property of steps in which several partially-prefilled requests
with different KV depths share the budget** — threshold/fairness partitions, several long-request
tails, any scheduler that gives multiple partial prefills work in one iteration — and not of
short-prompt bursts under FCFS.

## 7. Killed hypotheses

| hypothesis | evidence | decision |
| --- | --- | --- |
| host result-ready gap ≠ GPU time at large chunks | join artifact; gap = CUDA under the sequence join | contract fixed |
| context shift breaks the aggregate predictor | P95 positive residual 11 ms, 0% misses | killed |
| decode-load shift breaks it | 5 ms | killed |
| decode-KV distribution matters at equal total | ≤1.5% | killed |
| CUDA-graph boundaries need a feature | ≤0.7 ms | killed |
| adding geometry statistics to the aggregate model suffices | M1 MAE 132 ms; unidentifiable on one-prefill data | killed |
| uncertainty margins alone fix the aggregate model | P99 table 34% misses; online margin unstable at N=8 | killed |
| a thin trace-fit margin transfers to the live engine | 4–12% violations | killed |
| trace replay predicts live magnitudes | 94–99% of 100 ms decisions extrapolated; over-priced small chunks | screening only |
| geometry control beats an oracle fixed budget by ≥15% | +8–10% static, +5–12% with online margin | not met |
| geometry matters for FCFS short-prompt bursts | M0 ≡ M2; native fixed wins by 10% | killed |
| a 50 ms deadline is attainable in this setting | smallest admissible chunk ≈ 45 ms | out of reach |

## 8. Limitations

- **One GPU, one model, one engine version.** L20 (bandwidth-bound decode, modest compute),
  Qwen3-4B, vLLM 0.29.0. The 6.4 ms per M attention-work units slope and the 1.9× ratio are
  hardware- and model-specific; the *form* of the aliasing (Σ qᵢkvᵢ vs (Σq)(Σkv)) is not.
- **Partition is produced by `--long-prefill-token-threshold`.** The multi-request geometry was
  created by capping per-request chunks so several long requests share a step; under default
  FCFS the same steps occur only at request tails and for prompts shorter than the budget (§6).
  The controller's equal split is one partition policy; FCFS pricing was implemented but only
  exercised in the burst workload.
- **Synthetic workloads.** Random-token prompts, fixed 8 decoders, prefills injected together.
  No arrival process, no prefix caching, no speculative decoding, no tensor parallelism.
- **The fixed baseline is an oracle.** `fixed 256/384/1024` are the best budgets chosen after
  seeing the results of each workload; a practitioner does not know them in advance and the safe
  value moves between workloads (512 is 4.7% at N=4, 10.7% at N=8; 1024 is best for bursts and
  unsafe for 8 × 16k). The adaptive controller's value is that it does not need this tuning.
- **Trace replay.** Reported only as a screening tool; its 100 ms decisions were extrapolated
  beyond the traced per-request chunk sizes and it over-priced them.
- **Residual noise floor.** The truth model's own residual q95 is 10 ms and live step variance is
  larger than trace-fit variance; controllers within a few percent of each other are not
  separable without more repeats than three.
- **Two one-off stalls.** The first two runs of campaign 22 contain a single 3.8 s step each (a
  warm-up/JIT-type event after a code reinstall); they are kept in the raw data and excluded from
  the reported medians; all 34 other runs have max ≤101 ms.
- **Not upstreamed.** The tracer and controller are experiment patches to an installed wheel;
  nothing here modifies or comments on any vLLM PR.

## 9. Conclusion

On L20 / Qwen3-4B / vLLM 0.29, aggregate prefill coordinates are sufficient for ordinary
context/load shifts and short-prompt FCFS bursts, but alias partially-prefilled multi-request
geometries whose measured step costs differ by up to 1.9×. Replacing the aggregate token × KV
interaction with per-request attention work reduces geometry-OOD prediction error from hundreds of
milliseconds to single-digit milliseconds and yields 49–180% more safe prefill progress than an
aggregate deadline controller at essentially zero 100 ms violations; against a hindsight-tuned
oracle fixed budget, the gain is a more modest 5–12%.

---

*Reproduction:* `scripts/step_trace_join.py`, `analyze_measurement_contract.py`,
`analyze_step_cost_v2.py`, `replay_prefill_controller.py`, `analyze_live_controller.py`,
`plot_prefill_geometry.py`; raw campaigns 17–22 and the exact server/harness commands under
`benchmarks/results/l20-prefill-cost-geometry/raw/`; engine patches under `patches/`.
