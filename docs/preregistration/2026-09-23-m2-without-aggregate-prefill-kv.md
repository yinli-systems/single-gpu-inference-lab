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
