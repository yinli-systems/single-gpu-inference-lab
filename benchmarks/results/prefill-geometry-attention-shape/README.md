# Prefill cost geometry across attention shapes, and a pre-registered fix to M2

**Questions.**
1. Does the prefill-geometry result hold for models with a different attention shape, and
   does the slope scale with the attention work per token as the physics says?
2. The published geometry model M2 over-predicts multi-prefill steps. Its P95 error on the A100
   is ~50 ms. Why, and does the fix hold on data it was not tuned on?

**Answers.**
1. Yes. On two new shapes (Qwen2.5-1.5B on the L20, Qwen2.5-7B on the A100) the same-aggregate
   partitions differ by up to **1.83×** and **1.48×**, and each budget's partitions collapse onto
   one line (slopes within 1–3%). The slope follows layers × query heads within 4–11% at budget
   2048. At budget 1024 the smaller models come out 11–34% above it; that part of the slope
   prediction **missed** for 1.5B (2.53 vs a 2.5 bound).
2. M2's error is extrapolation of one term, the linear aggregate prefill KV (`ctx_kv_sum`). It
   is nearly collinear with attention work on one-prefill training data and runs 4× past its
   training range on multi-prefill steps. The model form itself fits (in-sample MAE 1–2.6 ms).
   Dropping that term (**M2n**) was pre-registered on 2026-09-23, before the two new campaigns
   were analyzed. It then passed every pre-registered criterion on both: primary-split MAE
   **1.75 ms** (1.5B, vs M2 23.9) and **1.23 ms** (7B, vs M2 17.0).

Pre-registration with timestamps:
[`docs/preregistration/2026-09-23-m2-without-aggregate-prefill-kv.md`](../../../docs/preregistration/2026-09-23-m2-without-aggregate-prefill-kv.md)
(b9050c3, then 358c450 before the 7B campaign started).

## Provenance

| item | L20 Qwen2.5-1.5B-Instruct | A100 Qwen2.5-7B-Instruct |
| --- | --- | --- |
| hardware | NVIDIA L20 48 GB (the L20 of all earlier artifacts) | A100-SXM4-80GB, GPU 1 of the RunPod pod |
| model | 28 layers, 12 query / 2 KV heads, dim 128; sha256 of `model.safetensors` checked against the hub | 28 layers, 28 query / 4 KV heads, dim 128 |
| software | vLLM 0.29.0 + tracer v2 | vLLM 0.29.0 + tracer v2 (the venv of `a100-prefill-cost-geometry/`) |
| lab commit | `19903b5`, clean tree | `19903b5`, clean tree |
| campaign | [`campaign/L20/`](campaign/L20/) (`campaign29.sh`, then `campaign29b.sh` for the cells after the failure below) | [`campaign/A100/campaign_a100_shape.sh`](campaign/A100/campaign_a100_shape.sh) |

Cells are the same as in the A100 Qwen3 replication: 6 partition cells, 4 context, 3 load and
2 skew, each with 3 repeats.

**One deviation.** These models have 32,768 positions. Every cell keeps `max-model-len` 40960
(`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`); prompts are random tokens and only step timing is measured.
But the 32k context cell with a 32,768-token prompt plus 32 output tokens ran past the rotary
position table and crashed the engine (device-side assert). It is kept in
`raw/L20-Qwen2.5-1.5B-Instruct/failed-L32768-40960/`, outside the analysis. For both models the
32k context cell uses a **32,640-token** prompt instead. It keeps its name, so the context split
classifies it as the 32k cell.

## 1. Partition at equal aggregate coordinate

Median step CUDA ms at 12–16k aggregate KV (full tables in [`partition.md`](partition.md)):

| | budget 1024: 1×1024 / 2×512 / 4×256 | ratio | budget 2048: 1×2048 / 4×512 / 8×256 | ratio |
| --- | --- | ---: | --- | ---: |
| L20 Qwen3-4B (published) | 163.1 / 120.5 / 100.4 | 1.63× | 336.1 / 200.4 / 178.9 | 1.88× |
| **L20 Qwen2.5-1.5B** | 62.1 / 45.0 / 36.4 | **1.70×** | 116.8 / 70.3 / 63.8 | **1.83×** |
| A100 Qwen3-4B | 81.6 / 60.7 / 48.9 | 1.67× | 162.5 / 91.5 / 80.2 | 2.02× |
| A100 Qwen3-8B | 111.2 / 90.0 / 78.7 | 1.41× | 216.0 / 144.7 / 133.4 | 1.62× |
| **A100 Qwen2.5-7B** | 94.7 / 77.8 / 69.6 | **1.36×** | 179.9 / 130.3 / 121.7 | **1.48×** |
| **A100 Qwen2.5-1.5B** | 29.4 / 22.6 / 19.2 | **1.53×** | 56.9 / 34.8 / 31.3 | **1.82×** |
| **L20 Qwen2.5-7B** | 190.3 / 164.5 / 150.6 | **1.26×** | 381.2 / 292.5 / 277.5 | **1.37×** (pre-registered 1.3–1.7: held) |

Attention-work slope (ms per million units of Σqᵢ(kvᵢ+(qᵢ+1)/2)), from fits on the proxy alone,
one fit per partition:

| GPU | model | layers × query heads | budget 1024 | budget 2048 | pre-registered |
| --- | --- | ---: | --- | --- | --- |
| L20 | Qwen3-4B | 1152 | 6.44–6.57 | 6.37–6.65 | (reference) |
| L20 | Qwen2.5-1.5B | 336 | 2.53–2.55 | 2.09–2.10 | 1.9, range 1.3–2.5: **held at 2048, missed at 1024** |
| A100 | Qwen3-4B | 1152 | 3.29–3.47 | 3.16–3.32 | (reference) |
| A100 | Qwen3-8B | 1152 | 3.29–3.43 | 3.20–3.33 | — |
| A100 | Qwen2.5-7B | 784 | 2.52–2.58 | 2.27–2.32 | 2.25, range 2.0–3.0: **held** |
| A100 | Qwen2.5-1.5B | 336 | 0.98–1.02 | 0.98–1.00 | — (run after the addendum; no prediction) |
| L20 | Qwen2.5-7B | 784 | 4.21–4.29 | 4.22–4.30 | 4.4–4.9, range 4.0–5.6 (2048) / 4.0–6.6 (1024): **held** |

Normalized per 1,000 layer·query-head units at budget 2048: the L20 gives 5.6 (4B) and 6.2
(1.5B), the A100 gives 2.8 (4B/8B), 2.9 (7B) and 2.9 (1.5B). So one GPU-specific constant times the attention
work per token predicts the slope to within 4–11% at budget 2048. At budget 1024 the smaller
models cost 11–34% more per unit on the L20 and 7B's +11% on the A100; the A100 1.5B shows no
budget-1024 excess (0.98–1.02 at both budgets), so the excess is not simply "fewer heads". That is
what the L20 1.5B prediction missed.

## 2. Predictors under geometry shift, with the pre-registered M2n

[`scripts/analyze_m2_variants.py`](../../../scripts/analyze_m2_variants.py) →
[`m2-variants.md`](m2-variants.md). Filters as in the A100 artifact (first iteration and > 10×
median dropped). Primary split = train on one-prefill steps, test on multi-prefill steps.

| data | M0 MAE | M2 MAE / P95 / signed | **M2n** MAE / P95 / signed | M0 in-sample | M2 in-sample | M2n in-sample |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| L20 Qwen3-4B | 339.8 | 10.4 / 24.1 / −10.1 | 3.1 / 10.6 / +1.8 | 30.7 | 2.1 | 2.4 |
| A100 Qwen3-4B | 171.5 | 18.6 / 47.5 / −18.3 | 1.9 / 4.0 / −0.3 | 16.3 | 2.2 | 1.7 |
| A100 Qwen3-8B | 170.0 | 18.0 / 50.8 / −17.4 | 1.7 / 3.3 / +1.5 | 15.7 | 1.6 | 1.0 |
| **L20 Qwen2.5-1.5B** (unseen) | 116.4 | 23.9 / 67.3 / −23.7 | **1.75 / 3.4 / −0.6** | 11.5 | 2.3 | 1.6 |
| **A100 Qwen2.5-7B** (unseen) | 122.7 | 17.0 / 48.4 / −16.6 | **1.23 / 2.3 / +0.9** | 11.4 | 1.4 | 1.0 |
| A100 Qwen2.5-1.5B (after the addendum) | 55.9 | 17.4 / 45.9 / −17.2 | 2.13 / 3.7 / −1.7 | 6.2 | 2.1 | 1.7 |
| **L20 Qwen2.5-7B** (unseen, addendum 6) | 235.7 | 11.4 / 32.5 / −10.9 | **2.31 / 6.5 / +1.7** | 20.0 | 2.0 | 1.8 |

The first three rows are where the diagnosis was made, so they are not evidence for M2n. The
last two are the test.

Pre-registered criteria on the two unseen campaigns:

| criterion | L20 1.5B | A100 7B |
| --- | --- | --- |
| 1. M2n MAE < M2 MAE (primary) | 1.75 < 23.89 ✓ | 1.23 < 16.98 ✓ |
| 2. M2n MAE ≤ 5 ms and \|signed\| ≤ 5 ms | 1.75, −0.62 ✓ | 1.23, +0.88 ✓ |
| 3. reverse split: M2n ≤ 1.5 × M2 | 2.16 ≤ 4.29 ✓ | 1.60 ≤ 3.48 ✓ |

The same three criteria on the L20 Qwen2.5-7B campaign (addendum 6, run after the disk was freed):
primary MAE 2.31 < 11.43 ✓; 2.31 ≤ 5 ms with signed +1.68 ✓; reverse 2.55 ≤ 4.67 ✓. The
pre-registered slope and partition-gap ranges held as well (table above).

Two further readings:

- **M2's over-prediction grows with the number of prefills.** Mean signed error at 2 / 4 / 8
  prefills: A100 4B −7.5 / −14.9 / −30.6 ms, 1.5B −7.4 / −18.8 / −41.0 ms. With M2n it is flat
  within ±3 ms (see `m2-variants.md`).
- **M0's failure is the model form, not extrapolation.** Fitted on train and test together, M0
  is still off by 11–31 ms MAE, while M2 and M2n fit to 1–2.6 ms. No aggregate coordinate can be
  calibrated into pricing these steps.

The published M2 stays as the reference in the published artifacts. M2n is the model to use
going forward. It is also one feature simpler.

## 3. Decode-KV skew at equal aggregate: one pattern per GPU

[`decode-kv-skew.md`](decode-kv-skew.md): decode-only step p50, skewed (4×7936 + 4×256) over
balanced (8×4096) at equal aggregate:

| GPU | model | skewed / balanced |
| --- | --- | ---: |
| L20 | Qwen3-4B | 0.995 |
| L20 | Qwen2.5-1.5B | 1.007 |
| A100 | Qwen3-4B | 1.086 |
| A100 | Qwen3-8B | 1.054 |
| A100 | Qwen2.5-7B | 1.034 |
| A100 | Qwen2.5-1.5B | 1.066 |
| L20 | Qwen2.5-7B | 1.002 |

All three L20 models: no effect. All four A100 models: 3–9% slower. The decode-skew sensitivity
belongs to the GPU, not to one model.

### Mechanism: not the attention kernel (open)

The A100 artifact guessed that the longest sequence's split-KV work becomes the critical path.
That guess is **wrong**. Tests in [`skew-mechanism/`](skew-mechanism/), Qwen3-4B shapes, 36 layers:

| test | A100 skewed / balanced | L20 skewed / balanced |
| --- | ---: | ---: |
| FA2 paged decode kernel as vLLM calls it (FA2 takes no explicit `num_splits`) | 0.994 | 0.974 |
| same, in a CUDA graph captured at the actual lengths | 0.990 | 0.963 |
| same, graph captured once at max_seqlen_k 8192 / 40960, lengths swapped in | 0.990 / 0.990 | 0.963 / 0.963 |
| same, KV blocks shuffled (non-contiguous layout) | 1.010–1.011 | — |
| offline vLLM engine, torch profiler, decode kernel (`Split=true`) CUDA time | 0.988 (graph) / 1.024 (eager) | — |

None reproduces the +0.8–0.9 ms per step measured in the live server cells. The per-kernel
profile does show a +76% instance, but it is the causal no-split *prefill* kernel: the skewed
prompts' own quadratic prefill (4 × 7,936² vs 8 × 4,096²), not decode. So the penalty is real in
the live measurements (stable across repeats and four models) but it is not in the attention
kernel, and an offline engine run does not show it. The mechanism is open; it would need an
Nsight profile of the live server.

## Reproduce

```bash
A=benchmarks/results/prefill-geometry-attention-shape
python scripts/analyze_partition_geometry.py --steps L20-Qwen2.5-1.5B=$A/raw/L20-Qwen2.5-1.5B-Instruct/steps.csv A100-Qwen2.5-7B=$A/raw/A100-Qwen2.5-7B-Instruct/steps.csv
python scripts/analyze_m2_variants.py --steps L20-Qwen2.5-1.5B=$A/raw/L20-Qwen2.5-1.5B-Instruct/steps.csv A100-Qwen2.5-7B=$A/raw/A100-Qwen2.5-7B-Instruct/steps.csv
```

`steps.csv` regenerates from `raw/*/trace/*.jsonl.gz` (gunzip, then `analyze_step_cost_v2.py
--csv … --exclude-over-median-x 10 --exclude-first-iteration`).
