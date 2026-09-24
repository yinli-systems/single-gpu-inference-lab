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

## Addendum 8 (2026-09-24): pairing swap, learned baseline, P-PAS, cache-aware pricing, window rule; before any of it runs

### A. Pairing swap (live, L20 and A100, Qwen3-4B)
`scripts/measure_pairing_swap.py`: two partial prefills at depths (k_a, k_b) get chunks (q_a, q_b)
in state A and (q_b, q_a) in state B. Everything an aggregate or marginal-statistics model can see
is identical between the states: prefill count, Σq, Σk, both multisets and so their max/min/
variance, decode state, padded tokens. Only the pairing differs; ΔW = (q_a − q_b)(k_a − k_b).
The states are built exactly through vLLM's own scheduler (cached prefixes, budget = q_a + q_b);
every step is verified against the intended (chunks, depths). 20 trials per state, A/B
alternating, fresh suffix tokens each trial.

Configs (k_a/k_b, q_a/q_b): 4k/16k 256/768; 4k/16k 128/896; 4k/16k 384/640; 8k/24k 256/768;
0/16k 256/768; control 4k/16k 512/512 (ΔW = 0).
- PS1: for every config with |ΔW| ≥ 3M, the measured Δ = median(A) − median(B) has the predicted
  sign and lies within ±25% of slope × ΔW (slope: L20 6.5, A100 3.3 ms/M, the published fits).
- PS2: control |Δ| ≤ 1 ms.
- Identifiability: A and B map to the same aggregate vector, so any deterministic aggregate
  predictor has mean absolute error ≥ |Δ|/2 on the pair (|f − y_A| + |f − y_B| ≥ |y_A − y_B|).
  This is reported as the measured |Δ|/2.

### B. LPRS-style learned baseline (offline, existing steps.csv)
An MLP with 3 layers (2 hidden layers of 64, ReLU, scikit-learn, early stopping) on 16 aggregate
features: decode batch, Σ decode KV, max/mean decode KV, prefill count, Σq, max q, Σ prefill KV,
max/mean prefill KV, Σq·Σk, total tokens, padded tokens, eager flag, Σq², decode KV variance. It
sees every aggregate and marginal statistic but not the per-request pairing. Same primary and
reverse splits, same filters.
- LB1: MLP primary-split MAE > M2n MAE on all six datasets.
- LB2 (sample efficiency): train on N ∈ {32, 64, 128, 256, 512, 1024, 2048} one-prefill steps
  (5 random draws each) and test on the multi-prefill cells. Predicted: M2n reaches ≤ 5 ms MAE by
  N = 256 on every dataset; M0 and the MLP do not reach it at any N.
- LB3 (geometry exposure): add a fraction f ∈ {0, 1, 5, 10, 25}% of multi-prefill steps from the
  2×512 and 4×512 cells to training; test only on the 4×256 and 8×256 cells. This is exploratory;
  it reports how much representative geometry each model needs.

### C. Live arms added to the replay and workload-shift runs (A100)
- `ppas`: P-PAS as published (arXiv 2608.15171 Alg. 1: N_th 2, B_max 16384, B_cap 2048, prefill-token
  cap only, no per-request cap), `--max-num-batched-tokens` 16384.
- `ppas-8k`: the same with B_max = 8192 and MBT 8192, budget-matched to our 8192 ceiling. Labelled
  "P-PAS-style (budget-matched)", never "P-PAS".
- `ctl-m2n-fcfs-cache`: `ctl-m2n-fcfs` pricing the first 16 waiting requests at their prefix-cache
  hit depth.

Where they run:
- Workload shift: these three arms are added, and every arm goes to **4 repeats** (2 per GPU;
  GPU is a block). Report means with 95% CIs.
- Mooncake tool-agent ×0.2: the three arms, 2 repeats.
- R4: goodput(`ctl-m2n-fcfs-cache`) ≥ goodput(`ctl-m2n-fcfs`) on Mooncake tool-agent.
- W4: regret(`ctl-m2n-fcfs`) ≤ regret(`ppas`) and ≤ regret(`ppas-8k`) in the workload shift.
  This is the claim under test.

### D. Window-selection rule (frozen here)
A replay window is the earliest contiguous 240 s (after rate scaling) that has ≥ 100 requests, no
arrival gap > 20 s, simulated `default` TTFT p50 < 1 s, and < 10% of simulated steps outside the
clock range. No geometry statistic or controller outcome is used. The primary windows already
chosen are checked against this rule and any deviation is reported. For robustness, two more
non-overlapping windows per trace (Mooncake tool-agent ×0.2, Azure code ×0.5, BurstGPT ×60) are
picked by the same rule and run with {`default`, `ppas`, `ctl-m2n-fcfs`} for one repeat each.

### E. Mooncake within-slot spread
Simulator: 10 jitter seeds. Live: a second jitter seed for Mooncake tool-agent ×0.2 with
{`default`, `ppas`, `ctl-m2n-fcfs`}. Report the range.

### F. SLO sensitivity
The primary SLO stays TTFT ≤ 5 s and TPOT ≤ 100 ms. A sensitivity grid TTFT ∈ {2, 5, 10} s ×
TPOT ∈ {50, 100, 200} ms is computed from the per-request records.

### G. Scheduler overhead
The controller decision time is reported vs active requests (offline on the controller code:
p50 15 µs at 1 active and 133 µs at 256 on an M-series CPU; to be repeated on the A100 host's CPU).

### H. vLLM 0.30.0 minimal replication (A100, Qwen3-4B)
Partition cells (1×2048 / 4×512 / 8×256) and the pairing swap on vLLM 0.30.0, with the tracer
re-anchored. Pass criterion: 12–16k ratio 1×2048/8×256 within ±15% of the 0.29 value (2.02×),
and PS1 holds.

## Addendum 9 (2026-09-24): outcomes so far and the vLLM 0.30 replication moved to the L20; before 0.30 runs

- **A. Pairing swap, L20: held.** PS1: measured / predicted = 0.89, 0.88, 0.91, 0.86 and 1.01 for the
  five non-control configs, all with the right sign and within ±25%. PS2: the control Δ was −0.09 ms.
  (Three earlier attempts produced no usable data: the in-process engine skips the iteration tracer,
  the LLM class turns stats logging off, and sequential submission split the two prefills across
  steps. The measurement ran with AsyncLLM and one background decoder. The failed directories are
  kept.)
- **B. Learned baseline: LB1 and LB2 held.** MLP primary MAE 40–232 ms vs M2n 1.2–3.1 ms. M2n
  reaches ≤ 5 ms at N = 64; M0 and the MLP never do.
- **D. Window rule.** The Mooncake primary windows meet it. The BurstGPT primary window meets it but
  is not the earliest (that one is at 7.15 h). No Azure-code window meets the 20 s gap limit (the
  trace arrives in minute-long bursts). With the gap relaxed to 60 s for Azure code only (a reported
  deviation), the primary window t = 180 s is the earliest rule window.
- **E (simulation).** Across 10 jitter seeds, Mooncake prevalence moves by ≤ 0.7 percentage points
  in every stable cell.

**H moved to the L20** (the A100 queue runs for many more hours). vLLM 0.30.0 in its own venv,
tracer v2 re-anchored (0.30 adds `cudagraph_stats = None` before the common case; the installer
accepts both). Every cell of the shape campaigns runs on Qwen3-4B, plus the pairing swap, with
`--enable-scale-out` (0.30 makes `/inference/v1/generate` opt-in).
- H1: 12–16k ratio 1×2048 / 8×256 within ±15% of the L20 0.29 value (1.88×), i.e. 1.60–2.16×.
- H2: pairing swap PS1 and PS2 hold on 0.30.
- H3: M2n primary-split MAE ≤ 5 ms and below M2's, as in addendum 1.

## Addendum 10 (2026-09-24): stage-1 replay outcome and SLO-aligned deadline arms; before any D = 100 run

Stage-1 live replay outcome (addendum 4): on Mooncake tool-agent ×0.2 both deadline controllers
collapse. `ctl-m2n-fcfs` and `ctl-m0-fcfs` reach 0.06 req/s primary goodput against 0.98 for
`default` and `b8192`, with TTFT p50 of 42–45 s. They fill every step to the 65 ms deadline (TPOT
p50 52–53 ms) and so admit less prefill per second than the stock 2048 budget; the queue grows.
- R1: held at ×0.1 (0.51 vs 0.49); a tie at ×0.2 (0.06 vs 0.06).
- R2: held at ×0.2 (TTFT p90 188 s vs 64 s); **failed** at ×0.1 (2.40 s vs 3.76 s).
- R3: held. No arm beats the best stock arm; the controllers lose.
- V2: held for the four cells measured so far.
- The Azure-code t = 0 cells are unsaturated, as expected; every arm reaches 0.26 req/s.

The 65 ms deadline was chosen on the synthetic workload to match the L20's relative tightness. It is
stricter than the serving SLO needs (TPOT ≤ 100 ms). **Exploratory arms with the step deadline set
equal to the TPOT SLO (D = 100 ms)**, added after stage 3 on each GPU:
`ctl-m2n-fcfs-D100`, `ctl-m0-fcfs-D100`, `ctl-m2n-fcfs-cache-D100`. They run on Mooncake tool-agent
×0.2 (1 repeat per GPU) and in the workload shift (2 repeats per GPU).
- R5: goodput(`ctl-m2n-fcfs-D100`) ≥ 0.9 × the best stock arm on Mooncake ×0.2. If it holds, the
  collapse came from deadline tightness; if it fails, per-step deadline control is the wrong
  objective for SLO goodput.
- R6: goodput(`ctl-m2n-fcfs-D100`) ≥ goodput(`ctl-m0-fcfs-D100`) on Mooncake ×0.2.
- W5: regret(`ctl-m2n-fcfs-D100`) ≤ the smallest regret among the stock arms (the claim under test).

## Addendum 11 (2026-09-24): outcomes of addenda 4–10, recorded after all runs

| item | outcome |
| --- | --- |
| Pairing swap, L20 0.29 / L20 0.30 (PS1, PS2) | held / held (measured / predicted 0.86–1.01; control ≤ 0.09 ms) |
| Pairing swap, A100 (PS1, PS2) | **PS1 failed** (0.65–0.89; 2 of 5 within ±25%, sign always right); PS2 held (+0.01 ms) |
| Learned baseline LB1, LB2 | held (MLP 40–232 ms vs M2n 1.2–3.1 ms; M2n ≤ 5 ms at N = 64, MLP and M0 never) |
| L20 Qwen2.5-7B (addendum 6) | all held (M2n 2.31 ms; slope 4.21–4.30; gap 1.37×) |
| L20 live M2n (L1, L2, L3) | L1 held; L2 failed (+24%, better than predicted); L3 held against the pre-registered arms, but only +3…+11% over fixed-384 from the earlier block |
| vLLM 0.30 (H1, H2, H3) | H1, H2 held; **H3 failed** (M2n 6.30 ms). 0.30 has 2.2–4 s mid-run stalls (17 steps, 8 of 19 cells) under the 10× exclusion; a post-hoc 5× diagnostic gives 2.80 ms |
| Live replay R1–R6 | R1 held / tie; R2 held at ×0.2, failed at ×0.1; R3 held; R4 tie / not better; **R5 failed** (0.24 vs 0.98); R6 held |
| Simulator vs live V1, V2 | V1 held; V2 held in 10 of 13 cells, failed in 1 (BurstGPT 0.1% vs 0.5%), 2 too close to zero |
| Workload shift W1–W5 | **W1 failed** (b1024 best in every phase); W2, W4, W5 failed; W3 held (M2n 31% < M0 39% regret) |
| E (jitter) | robust: ≤ 0.7 points across 10 simulated seeds; live seed 1 matches seed 0 |
| G (overhead) | p50 28–214 µs, p99 ≤ 231 µs at 1–256 active requests (EPYC 7513) |
| Decode-skew mechanism | not the FA2 decode kernel (five tests); recorded as measured but unexplained |

Evidence: `benchmarks/results/{prefill-pairing-swap, prefill-geometry-learned-baseline,
prefill-geometry-attention-shape, l20-prefill-live-controller-m2n, vllm030-replication,
live-trace-replay, trace-geometry-prevalence}/`.
