# Does the startup-static adaptive-verification cost profile mis-price real serving? — measured, and it does not matter (L20, Qwen3-4B, vLLM 0.29, DSpark)

**Hypothesis (pre-registered).** vLLM's DSpark adaptive verification prices verification budgets from
cost tables profiled once at startup with a synthetic context (`VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN`,
default 8192; tracker #51303, docs). If real context differs, the budget decision should be wrong
and cost goodput. Gate: kill if goodput regret across realistic mismatches is <3% and budgets barely
move, or the cost is wrong but the decision is not; alive at ≥5% regret or budgets wrong by a tier
with consequence; strong at ≥10%.

**Verdict: killed.** The cost prediction is wrong everywhere — by +6…+55 ms per step, systematically
optimistic, up to 2.3× at 6k context *even with a matched profile context* — but the budget
`argmax_b accepted(b)/cost(b)` is insensitive to it: budgets move by one tier only for
low-acceptance content at ≤2k context, not at all for high-acceptance content or at 6k, and the
measured goodput-versus-budget curve is flat within ±30% of its optimum, so the true hindsight
regret of the default profile is 0–3% in every cell. An online context-aware cost calibrator would
correct a 2× prediction error and change serving throughput by ≤3%.

## Setup

As in [`../l20-spec-decode-geometry/`](../l20-spec-decode-geometry/README.md) (Qwen3-4B, L20,
vLLM 0.29.0, `TRITON_ATTN`, DSpark block-7 drafter, `enable_adaptive_verification=true`), plus
tracer v4 ([`patches/apply_tracer_v4_adaptive.py`](patches/apply_tracer_v4_adaptive.py)): every
step row carries the controller's decision — chosen draft-slot budget, maximum budget, predicted
step cost (draft-table + verify-table terms), estimated accepted tokens — stashed at decision time
and picked up by the step timer. Forced budgets for the hindsight curves via
[`patches/patch_force_budget.py`](patches/patch_force_budget.py) (`VLLM_EXP_FORCE_AV_BUDGET`).
Actual context classes `code-ctx512 / 2048 / 6000`, `prose-ctx…` (prompt tokens measured);
analysis on the co-decoding window ([`scripts/analyze_adaptive_profile.py`](../../../scripts/analyze_adaptive_profile.py)).

- Campaign25: profile 512 / 2048 / 8192 × actual ~0.5k / ~2k × {code, prose} × B 32 / 64, 512-token
  completions, 2 repeats ([`campaign25-tables.md`](campaign25-tables.md)).
- Campaign25b: hindsight goodput-vs-budget curves with forced budgets on the decisive cells, and
  the matched profile 6144 ([`hindsight-budget-curves.json`](hindsight-budget-curves.json)).
- Campaign25c: the 6k regime with a real co-decoding window (B=24, 2048-token completions),
  profile 512 / 6144 / 8192, 2 repeats ([`campaign25c-6k-tables.md`](campaign25c-6k-tables.md)).
  (At B=32 × 6k the 25–33 s Triton prefill outlasts the completions, so those cells have no window.)

## 1. Cost prediction error (actual CUDA step − predicted, ms; p50 over the window)

| actual | B | profile 512 | profile 2048 | profile 6144 | profile 8192 (default) |
| --- | ---: | ---: | ---: | ---: | ---: |
| code ~0.5k | 32 | +7.8 (of 31.0) | +5.2 | — | −6.4 |
| prose ~0.5k | 64 | +12.0 (of 40.5) | +5.7 | — | −18.0 |
| code ~2k | 32 | +21.9 (of 45.0) | +19.5 | — | +7.7 |
| prose ~2k | 64 | +38.7 (of 67.2) | +33.2 | — | +9.6 |
| code ~6k | 24 | +50.5 (of 71.6) | — | **+42.6** | +40.0 |
| prose ~6k | 24 | +41.8 (of 60.8) | — | **+34.0** | +30.9 |

The profile is optimistic at every context except when an 8192 profile serves ~0.5k context; the
matched-context profile still under-predicts a 6k step by ~2.3×. The synthetic profiling batch is
not the real step (cf. #54046 on profiled shapes): context mismatch is a second-order shift on a
first-order fidelity gap.

## 2. Does the decision move? (chosen draft-slot budget p50 / max)

| actual | B | profile 512 | 2048 | 6144 | 8192 |
| --- | ---: | ---: | ---: | ---: | ---: |
| code ~0.5k | 32 | 160 / 224 | 160 | — | 160 |
| prose ~0.5k | 32 | 96 / 224 | 96 | — | **152** |
| prose ~0.5k | 64 | 128 / 448 | 128 | — | **192** |
| prose ~2k | 32 | 96 / 224 | 96 | — | **152** |
| code ~6k | 24 | 160 / 168 | — | 160 | 160 |
| prose ~6k | 24 | 104 / 168 | — | 104 | 104 |

Only low-acceptance content at ≤2k context sees a one-tier shift (the default profile's larger
fixed cost pushes the ratio's argmax up); high-acceptance content and the 6k regime pick the same
budget under every profile.

## 3. Does the moved decision cost anything? (hindsight curves, forced budgets)

| cell | budget → goodput (tok/s) | hindsight best | default profile chose → true regret |
| --- | --- | ---: | ---: |
| code ~0.5k, B=32 | 64: 3,459 · 96: 4,416 · 128: 4,685 · **160: 5,107** · 192: 4,661 · 224: 4,724 | 160 | 160 → 0% |
| prose ~0.5k, B=32 | 64: 2,921 · **96: 3,242** · 128: 3,181 · 160: 3,161 · 192: 2,886 · 224: 2,666 | 96 | 152 → ≈2.5% |
| prose ~2k, B=32 | 64: 1,889 · **96: 2,058** · 128: 2,006 · 160: 1,996 · 192: 1,826 · 224: 1,915 | 96 | 152 → ≈3% |
| prose ~0.5k, B=64 | 128: 3,750 · **192: 3,828** · 256: 3,557 · 320: 3,307 · 384: 3,057 · 448: 2,778 | 192 | 192 → 0% (the *matched* 512 profile's 128 is 2% worse) |
| code/prose ~6k, B=24 | all profiles identical budget | — | 0% (goodput within ±3% noise across profiles) |

Regret of every profile against the matched profile, all cells (campaign25 + 25c): −5.0% … +5.3%
with sign changing between repeats and cells; against the hindsight optimum: 0–3%.

## 4. Killed hypotheses

| hypothesis | evidence | decision |
| --- | --- | --- |
| a deployment-wide startup context scalar cannot fit real context | true for *cost prediction* (+6…+55 ms) — but the matched scalar is just as wrong at 6k | prediction error is a fidelity gap, not a context-selection gap |
| mispricing changes the verification budget | one tier, only for low-acceptance content at ≤2k | partly |
| the changed budget costs serving performance | ≤3% true regret; at B=64 the "wrong" larger budget is the optimum | killed |
| online context-aware cost calibration is worth building | it would fix a 2× cost error to gain ≤3% | killed |

## What survives

The ratio objective is what makes the shipped controller robust: a common additive error on all
candidate budgets barely moves `argmax_b accepted(b)/cost(b)`. The real profiler defect is
fidelity (synthetic steps ~2× cheaper than real ones at depth), which would matter for any
*absolute* cost consumer (deadline-aware scheduling, SLO admission) but not for this controller.
Not pursued further here.

## Limitations

One GPU/model/drafter; Triton attention (the only adaptive-capable backend on SM89); synthetic
prompts; the 6k regime measured at B=24 rather than 32/64 (KV capacity and prefill stagger);
forced-budget curves at 6 points, single repeat; run-to-run goodput noise ≈3%.
