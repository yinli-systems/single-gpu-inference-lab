# Pre-registration: M2 without the aggregate prefill-KV term (2026-09-23)

Written and committed **before** any analysis of the Qwen2.5-1.5B-Instruct L20 campaign
(`campaign29`, started 17:27 CST, still running at time of writing) or of any later model/GPU.

## Finding that motivates it (post hoc, on already-published data)

The published M2 over-predicts multi-prefill steps, and the error grows with prefill count:
−7 / −15 / −30 ms mean signed error at 2 / 4 / 8 prefills on A100 Qwen3-4B, and −4 / −9 / −16 ms
on L20. Trained on train+test together, the same form fits the test rows to ~2 ms MAE, so the form
is not the problem. Extrapolation is. The linear aggregate prefill-KV feature `ctx_kv_sum` spans
0–32k in training (one prefill per step) and 0–128k in test, and on one-prefill data it is
correlated with the attention-work proxy (r = 0.65).

Dropping that one feature (primary geometry split, same Ridge, same filters):

| data | M2 MAE / P95 | M2 − ctx_kv_sum MAE / P95 |
| --- | --- | --- |
| L20 Qwen3-4B, unfiltered (published setting) | 8.3 / 17.6 | 3.2 / 10.5 |
| A100 Qwen3-4B, clean filters | 18.5 / 47.5 | 1.9 / 4.0 |
| A100 Qwen3-8B, clean filters | 18.0 / 50.8 | 1.7 / 3.3 |

Because this variant was chosen after seeing these test sets, the numbers above are **not**
evidence for it.

## Frozen prediction

Model `M2-noaggkv` = `analyze_step_cost_v2.features(r, 2)` with index 2 (`ctx_kv_sum / 1e4`)
removed; everything else unchanged (Ridge, lambda 1e-2, primary split definition, filters
`--exclude-over-median-x 10 --exclude-first-iteration`).

On every new (model, GPU) campaign run with the same cells, starting with Qwen2.5-1.5B on L20:

1. `M2-noaggkv` primary-split MAE is lower than the published `M2` MAE.
2. `M2-noaggkv` primary-split MAE is at most 5 ms, and its mean signed error is within ±5 ms.
3. On the reverse split, `M2-noaggkv` MAE is no more than 1.5× the published `M2` MAE (removing the
   term must not break the other direction).

A failure of any of these on new data is reported as a failure. The published `M2` stays the
reference in all existing artifacts either way.

Also predicted before looking: on Qwen2.5-1.5B (28 layers, 12 query heads of dim 128) the
attention-work slope in the partition fits is about (28·12)/(36·32) ≈ 0.29 of Qwen3-4B's L20
slope (6.4–6.6 ms/M), i.e. **1.9 ms/M, accepted range 1.3–2.5 ms/M** (kernel efficiency at fewer
heads per step may differ).

## Addendum (2026-09-23, before the A100 Qwen2.5-7B-Instruct campaign starts)

Outcome on Qwen2.5-1.5B (L20), for the record: predictions 1–3 held (primary MAE 1.75 vs M2
23.89 ms, signed −0.62 ms; reverse 2.16 vs 2.86 ms). The slope prediction held at budget 2048
(2.09–2.10 ms/M) and **missed** at budget 1024 (2.53–2.55 ms/M, above the 2.5 bound).

New, for Qwen2.5-7B-Instruct on the A100 (28 layers, 28 query / 4 KV heads, dim 128), same cells:

- Predictions 1–3 above apply unchanged.
- Attention-work slope ≈ (28·28)/(36·32) × 3.3 ms/M (A100 Qwen3-4B) = **2.25 ms/M**. Given the
  1.5B slope came out 10–35% above its head-count scaling, accepted range **2.0–3.0 ms/M** at both
  budgets.

## Addendum 2 (2026-09-23): live deadline controller on the A100 (Qwen3-4B), before launch

Protocol = L20 campaign20 (§5 of `l20-prefill-cost-geometry`): 8 decoders (4096 output tokens) +
N ∈ {4, 8} × 16k prefills injected together, `--max-num-batched-tokens 8192`, one fresh server per
run, conditions interleaved, 3 repeats; A100 GPU 0; controller models fit on A100 one-prefill
steps only (`replay_prefill_controller.py --export-models --exclude-over-median-x 10
--exclude-first-iteration`, q95 margins M0 10.69, M2 10.44, M2n 10.32 ms).

Arms: fixed-128, fixed-256, fixed-512, M0 (`m0-one`), M2 (`m2-one`, as published), M2n (`m2n-one`,
this pre-registration's variant; scheduler feature vector checked identical to the replay's on
6,000 random steps).

Deadline **65 ms**, chosen from the offline A100 replay (`--deadlines 50…100`) as the value
closest to the L20 regime (L20 at 100 ms: fixed-256 safe, fixed-512 4–10% violations). At 50 ms
every model controller is infeasible (fixed step cost ~38 ms + margin ~10 ms).

Predictions (the replay's truth model is itself a fit, so these can fail):

- P1. M2 (published) under-performs at N = 8: its aggregate prefill-KV term over-prices
  8 × 16k-deep prefills and the replay has it stalling. Live, the scheduler falls back to the
  smallest candidate (64), so expected: M2 progress well below fixed-256 at N = 8.
- P2. M2n ≥ M0 in safe prefill progress at both N, with violations ≤ 5%.
- P3. M2n ≥ 1.15 × the best fixed budget with violations ≤ 5%, at both N (the L20 gate; L20's
  M2 missed it with +7–8%).

Gates reported as on L20: aggregate → geometry (M0 vs M2 and M0 vs M2n); geometry → best safe
fixed budget (≥ 15%).

## Addendum 3 (2026-09-23, during the live run): stronger fixed baselines

The first live repeat showed fixed-512 at 0% violations of 65 ms (the replay had predicted 14%),
so the best safe fixed budget may lie above the pre-registered fixed arms. That would make P3
too easy. After the main 36-run block, a second block runs fixed-768 and fixed-1024 with fixed-512
and M2n as anchors: N ∈ {4, 8}, 3 repeats, the four arms interleaved, same server settings. P3 is
evaluated against the best fixed budget with violations ≤ 5% across **both** blocks. If an anchor
differs by more than 5% in safe prefill tok/s between blocks, cross-block comparisons are
reported as unreliable.

## Outcomes (recorded 2026-09-23 after all runs)

- Qwen2.5-1.5B (L20): predictions 1–3 held; slope held at budget 2048, missed at 1024.
- Qwen2.5-7B (A100): predictions 1–3 held (M2n MAE 1.23 vs M2 16.98 ms, signed +0.88; reverse
  1.60 vs 2.32); slope held (2.27–2.32 / 2.52–2.58 ms/M, range 2.0–3.0).
- Live A100 controller: P1 held (M2 at 64-token budgets at N=8, 2,806 tok/s); P2 held (M2n 2.38× /
  4.80× M0 at ≤ 0.6% violations); **P3 failed** (M2n vs best safe fixed-512: 0.98× / 1.02× main
  block, 1.07× / 1.04× follow-up block; fixed-768/1024 violate 23–59%). The N=4 M2n anchor moved
  +7.1% between blocks, above the 5% threshold, so N=4 cross-block comparisons are unreliable.

Evidence: `benchmarks/results/prefill-geometry-attention-shape/`,
`benchmarks/results/a100-prefill-live-controller/`.

## Addendum 4 (2026-09-24): L20 live M2n arm and live trace-replay goodput, before launch

### L20 live controller with M2n

Same protocol as L20 campaign20 (100 ms, 8 decoders + N ∈ {4, 8} × 16k, fresh server per run,
interleaved, 3 repeats). Arms: fixed-256, fixed-512, M0, M2, M2n. M2n was exported from the same
L20 traces (campaigns 18+19) that reproduce the published `m2-one.json` to 3e-11, q95 11.42 ms.

- L1: M2n ≥ M0 in safe prefill tok/s at both N, violations ≤ 5%.
- L2: M2n within ±10% of M2 at both N (on the L20, M2 did not collapse at N=8).
- L3: M2n ≥ 1.15 × the best fixed arm with ≤ 5% violations. **Predicted to fail** (L20 M2: +8–10%;
  A100 M2n: −2…+7%).

### Live trace replay (A100, Qwen3-4B)

`scripts/replay_trace_serving.py` at de36031: prefix caching on, `max-num-seqs` 256, 240 s arrival
window per run, 2 repeats, arms interleaved. GPU 0 replays Mooncake tool-agent (3 s slot spread) at
×0.1 and ×0.2. GPU 1 replays Azure 2023 code at ×0.25 and ×0.5.

Arms:
- `default`: `--max-num-batched-tokens` 2048 (the A100 API-server default)
- `agentx`: 2048 with `--long-prefill-token-threshold` 512
- `b8192`: 8192
- `ctl-m2n-fcfs`: controller D = 65 ms, M2n, fcfs partition, ceiling 8192
- `ctl-m0-fcfs`: the same with M0
- `ctl-m2n-equal`: the same as `ctl-m2n-fcfs` with equal partition

Primary SLO: TTFT ≤ 5 s and TPOT ≤ 100 ms; goodput = requests meeting both ÷ 240 s. Other SLO
pairs are reported.

- R1: goodput(`ctl-m2n-fcfs`) ≥ goodput(`ctl-m0-fcfs`) in every (trace, load) cell.
- R2: `ctl-m2n-equal` has a worse TTFT p90 than `ctl-m2n-fcfs` in every cell. Equal partition is
  processor sharing, and the whole-horizon analysis shows sharing raises mean TTFT.
- R3 (predicted null): no arm beats the best of {`default`, `agentx`, `b8192`} by ≥ 15% in primary
  goodput in any cell. Per-step allocation cannot change total work, so any controller gain must
  come from choosing the budget, which a good static setting already approximates.

Known limitation: the controller prices waiting requests at depth 0 before their prefix-cache
lookup, so it under-prices cache-hit first chunks.

## Addendum 5 (2026-09-24): BurstGPT control, simulator-vs-live check, workload shift; before launch

The decode-KV-skew mechanism is recorded as **measured but unexplained** and paused (it is not in
the FA2 decode kernel; see `prefill-geometry-attention-shape/` §3).

### BurstGPT negative control (live)
Same six arms and harness as addendum 4. BurstGPT (first 20k non-failed requests) at ×100, 240 s
window, 2 repeats.

### Simulator vs live
For the `default` arm of every live replay (Mooncake tool-agent ×0.1/×0.2, Azure code ×0.25/×0.5,
BurstGPT ×100), compute from the engine trace P(prefills per step > 1) and P(geometry error > 5 ms)
over prefill steps, with the same formula and slope as `trace-geometry-prevalence/` (3.3 ms/M).
Compare with the simulator run on the same window and scale.
- V1: P(geo > 5 ms) is higher for Mooncake tool-agent and Azure code than for BurstGPT in live data,
  as in the simulator.
- V2: each live P(geo > 5 ms) is within a factor of 2 of the simulator's for the same cell.

### Workload shift (live)
One 600 s run: chat (Azure conv ×1, 0–150 s) → code (Azure code ×0.5, 0–150 s) → agent (Mooncake
tool-agent, 3 s spread, ×0.2, 0–150 s) → chat (the same Azure conv segment again). Arms: stock
`--max-num-batched-tokens` 512 / 1024 / 2048 / 4096 / 8192, `agentx` (2048 + cap 512),
`ctl-m2n-fcfs`, `ctl-m0-fcfs` (D = 65 ms, ceiling 8192). 2 repeats: repeat 1 on A100 GPU 0,
repeat 2 on GPU 1 (same host, same model of GPU).

Per phase: primary goodput (TTFT ≤ 5 s, TPOT ≤ 100 ms) by arrival. The per-phase hindsight best is
the best of the six stock arms in that phase (mean of repeats). Regret(arm) = Σ_phases (best − arm)
/ Σ_phases best.
- W1: no single stock arm is within 5% of the per-phase hindsight best in every phase.
- W2 (the claim under test): regret(`ctl-m2n-fcfs`) ≤ the smallest regret among the six stock arms.
  Prior: uncertain. On fixed workloads M2n matched the best fixed budget but did not beat it, so W2
  can only hold if the best fixed budget changes between phases (W1).
- W3: regret(`ctl-m2n-fcfs`) < regret(`ctl-m0-fcfs`).

## Addendum 6 (2026-09-24): Qwen2.5-7B-Instruct on the L20, before the campaign starts

Disk was freed on the L20 so the 7B model fits. Same cells and filters as the other shape campaigns
(`campaign29.sh` with the 32k cell at 32,640 tokens).

- Predictions 1–3 (M2n) apply unchanged.
- Slope: L20 constant 5.6–6.2 ms/M per 1,000 layer·query-head units (Qwen3-4B, Qwen2.5-1.5B at
  budget 2048) × 0.784 = **4.4–4.9 ms/M at budget 2048, accepted range 4.0–5.6**. At budget 1024
  the L20 1.5B came out 34% high, so the accepted range there is 4.0–6.6.
- Partition gap at 12–16k, budget 2048 (1×2048 / 8×256): below the L20 Qwen3-4B's 1.88×,
  because the 7B has a larger fixed cost per step (A100: 7B 1.48× vs 4B 2.02×). **Accepted range
  1.3–1.7×.**

## Addendum 7 (2026-09-24): Azure-code window correction, before any shift or corrected run

The Azure 2023 code trace has 63 requests in its first 60 s, **none between 60 and 120 s**, then
~4 req/s. The addendum-4 cells ×0.25 and ×0.5 with a 240 s window starting at t = 0 therefore
replay the same 63 requests. The window was not checked at design time. Both cells are
unsaturated (every arm meets the SLO), so they carry no information about the arms; they are
kept and reported as such.

Corrections (same arms, SLOs, predictions R1–R3 and W1–W3):
- New cells `azure-code-t120` at ×0.25 and ×0.5: the trace from t = 120 s on, 240 s window,
  2 repeats (one per A100 GPU), run after the BurstGPT control.
- Workload-shift `code` phase: offset 120 s instead of 0 (Azure code ×0.5, 150 s). This is changed
  before any shift run starts.

### Addendum 7b (same day, still before any affected run): window audit

An audit of every replay window found that the Azure-code gap is longer than 60 s. Per minute, the
first 20 minutes hold 63, 0, 0, 531, 187, 130, … requests, so `t120 ×0.25` would contain 0
requests. The BurstGPT window at t = 0, ×100 holds only 86 requests (0.36 req/s; the trace is
sparse for its first 8 hours). Final windows, each checked for request count and simulated
stability (TTFT p50 < 1 s, clock in range) before launch:

| cell | offset | scale | window | requests | req/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| azure-code-t180 | 180 s | ×0.25 | 240 s | 531 | 2.21 |
| azure-code-t180 | 180 s | ×0.5 | 240 s | 718 | 2.99 |
| shift `code` phase | 180 s | ×0.5 | 150 s | 531 | 3.54 |
| burstgpt-t14h (replaces BurstGPT t = 0 ×100) | 14 h | ×60 | 240 s | 1,111 | 4.63 |

The Mooncake tool-agent windows (130 / 245 requests) and the shift chat and agent phases (613 /
156 requests) are unchanged. The two original Azure-code cells (t = 0) are kept and reported as
unsaturated.
