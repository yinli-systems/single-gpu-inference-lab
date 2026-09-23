# Prefill cost geometry on A100: replication across hardware and model size

**Question.** [`l20-prefill-cost-geometry/`](../l20-prefill-cost-geometry/README.md) found, on one
L20 with Qwen3-4B, that the aggregate step coordinate (decode batch, aggregate KV, prefill tokens)
aliases partitions of the same prefill budget whose step cost differs up to 1.9×, and that one
per-request attention-work term Σqᵢ(kvᵢ+(qᵢ+1)/2) fixes it. Does that hold on different hardware
and a larger model, with the same harness and no retuning?

**Answer.** Yes, and the gap gets larger on the faster GPU. On an A100-SXM4-80GB the
same-aggregate partition gap reaches **2.02×** (Qwen3-4B, 1×2048 vs 8×256; L20: 1.88×) and
**1.62×** on Qwen3-8B. The attention-work term alone again puts all partitions of a budget on one
line (slopes within 4–6%). The slope is the same for 4B and 8B, which share the attention shape.
Geometry-OOD prediction MAE drops **171 → 18 ms** (4B) and **170 → 18 ms** (8B). The
context-shift and load-shift negative controls replicate. **One L20 negative result does not
replicate:** at equal aggregate KV, a skewed decode batch is **5–9% slower** on A100
(L20: −0.5%). So decode-KV skew is a real feature on this GPU for decode-only steps, though a
small one (+0.8–0.9 ms per step).

## Provenance

| item | value |
| --- | --- |
| hardware | RunPod, 2× A100-SXM4-80GB (driver 595.91.07, CUDA 13.2); **GPU 1 only**, and the campaign refused to start if GPU 1 held >500 MiB ([`raw/*/nvidia-smi.txt`](raw/Qwen3-4B/nvidia-smi.txt)) |
| software | vLLM 0.29.0, torch 2.13.0+cu130 (same versions as the L20 venv); attention backend FLASH_ATTN (server logs) |
| models | Qwen3-4B, Qwen3-8B (bf16) |
| lab commit | `350612d`, clean tree (both campaign scripts refuse a dirty tree); tracer v2 installed into the venv with [`apply_tracer_v2.py`](../l20-prefill-cost-geometry/patches/apply_tracer_v2.py) (this PR makes it tolerate 0.29.0 installs without `BatchExecutionDescriptor.uniform_decode`) |
| harness | [`scripts/measure_prefill_interference.py`](../../../scripts/measure_prefill_interference.py), same arguments as L20 campaigns 18/19: [`campaign/campaign_a100_part.sh`](campaign/campaign_a100_part.sh) (partition cells), [`campaign/campaign_a100_pred.sh`](campaign/campaign_a100_pred.sh) (context, load, skew cells); 3 repeats each |
| not re-run | graph-boundary cells, the live controller, the FCFS-burst runs (§5 of the L20 artifact); the GPU was taken by another job afterwards |

`raw/<model>/` holds per-cell JSON and logs, server logs, `trace/*.jsonl.gz` (engine iteration and
runner step traces, gzipped), `steps.csv` (joined per-step rows) and the predictor reports.

## 1. Partition of the same prefill budget (L20 §2.4)

Decode batch 8, full prefill budget per step, binned by aggregate KV depth; median step CUDA ms
(n). Full tables in [`partition.md`](partition.md) / [`partition.json`](partition.json), with the
L20 table regenerated from its own `steps.csv` by the same script.

| aggregate KV | L20 4B 1×1024/4×256 | A100 4B | A100 8B | L20 4B 1×2048/8×256 | A100 4B | A100 8B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–4k | 1.10× | 1.11× | 1.05× | 1.13× | 1.17× | 1.10× |
| 4–8k | 1.29× | 1.35× | 1.20× | 1.42× | 1.47× | 1.29× |
| 8–12k | 1.51× | 1.55× | 1.33× | 1.68× | 1.77× | 1.46× |
| 12–16k | 1.63× | 1.67× | 1.41× | **1.88×** | **2.02×** | **1.62×** |
| 16–20k | 1.70× | 1.70× | 1.45× | — | — | — |

Absolute medians at 12–16k, budget 2048 (1×2048 / 4×512 / 8×256): L20 4B 336.1 / 200.4 / 178.9 ms,
A100 4B 162.5 / 91.5 / 80.2 ms, A100 8B 216.0 / 144.7 / 133.4 ms.

Fits on the attention-work proxy alone (ms = a + b·M, M = proxy in millions):

| | budget 1024: 1×1024, 2×512, 4×256 | budget 2048: 1×2048, 4×512, 8×256 |
| --- | --- | --- |
| L20 4B | 77.2+6.44M, 77.2+6.47M, 77.3+6.57M | 151.1+6.37M, 152.1+6.48M, 152.5+6.65M |
| A100 4B | 38.0+3.29M, 37.5+3.31M, 37.1+3.47M | 70.3+3.16M, 69.6+3.24M, 69.5+3.32M |
| A100 8B | 67.7+3.29M, 67.1+3.41M, 67.0+3.43M | 122.1+3.20M, 122.3+3.23M, 122.1+3.33M |

Two readings:

- **The A100 halves the per-unit attention cost (6.4 → 3.3 ms/M). It cuts the fixed per-step cost
  by about the same factor for 4B** (77 → 38 ms at budget 1024). Ratios therefore stay about the
  same on 4B. They grow at budget 2048 because the attention term takes a larger share of the step.
- **4B and 8B have the same slope** (3.2–3.5 ms/M). Both have 36 layers of 32 query / 8 KV heads
  of dim 128, so the same attention work per token; 8B differs only in hidden size (4096 vs 2560)
  and MLP width. The geometry penalty in absolute ms is the same
  (1×1024 minus 4×256 at 12–16k: 32.7 ms on 4B, 32.5 ms on 8B). The larger model only adds fixed
  cost, which is why its *ratios* are smaller. For a deadline scheduler the absolute error is what
  matters, and that does not shrink with model size.

## 2. Out-of-distribution predictors (L20 §3)

`analyze_step_cost_v2.py`, unchanged model definitions (M0 aggregate coordinate, M2 with the
attention-work term), same four splits. MAE in ms, P95 |error|, P99 of under-prediction:

| split | L20 4B M0 → M2 | A100 4B M0 → M2 | A100 8B M0 → M2 |
| --- | --- | --- | --- |
| primary: train 1-prefill, test multi-prefill | 341.1 → **8.3** (P95 1208 → 17.6) | 171.5 → **18.5** (P95 589 → 47.5) | 170.0 → **18.0** (P95 592 → 50.8) |
| reverse: train multi, test 1-prefill | 37.0 → **5.5** (under-P99 150 → 23) | 18.3 → **3.3** (under-P99 75 → 14) | 18.6 → **2.6** (under-P99 76 → 9.5) |
| context ≤16k → 32k (control) | 3.3 → 2.6 | 4.6 → 4.6 | 4.2 → 4.2 |
| decode batch ≤16 → 32 (control) | 3.2 → 2.7 | 2.7 → 2.7 | 2.7 → 2.7 |

A100 columns are [`predictors-clean.json`](raw/Qwen3-4B/predictors-clean.json)
(`--exclude-over-median-x 10 --exclude-first-iteration`). The L20 column is the published report,
which needed neither filter. On the primary split M2 reduces error about 10× on A100, against 40×
on L20. The remaining M2 error on A100 (P95 ~50 ms) is larger than on L20 even though A100 steps
are shorter, so the linear M2 form fits this GPU less tightly. No model was changed to recover it.

### Step exclusions (and why they are needed here but were not on L20)

The pod was new, with empty kernel caches. Two kinds of one-off steps appear that the L20 (warm
caches, many earlier campaigns) never showed:

1. **~2.9 s steps**, one per server on a first-seen shape: 6 on 4B, 4 on 8B, each 20–60× its
   cell median. These are kernel compiles. They are removed by `--exclude-over-median-x 10`. On
   8B none of them falls inside a split's row filter, so the 8B numbers do not change with it.
2. **The first engine iteration of each server** (325–333 ms for a 128-token prefill against a
   ~60 ms median). This is usually under 10× the median, so filter 1 keeps it. It was the entire cause of
   an apparent reverse-split failure: with only filter 1, A100 4B reverse under-P99 was 276 ms
   for M0 and 262 ms for M2. `--exclude-first-iteration` removes one step per cell (19 per model) and
   the split behaves as on L20. On 4B, one of these (326.9 ms in a cell with a small median) had
   also been caught by filter 1.

All three variants are kept per model: `predictors.json` (unfiltered), `predictors-excl10x.json`,
`predictors-clean.json`. Every excluded step is listed with its reason in `excluded_steps`. With
**no** filtering the primary-split conclusion is unchanged (4B 184.4 → 24.6, 8B 172.4 → 14.5). Both
flags are opt-in and default off; without them the script behaves exactly as before.

## 3. Decode-KV skew at equal aggregate (L20 §2.2): does not replicate

8 decoders, prompt KV summing to 32,768 either way: balanced 8×4096 vs skewed 4×7936 + 4×256.
p50 / p95 CUDA ms ([`decode-kv-skew.md`](decode-kv-skew.md); same script and selection for all three):

| | L20 4B bal / skew | A100 4B bal / skew | A100 8B bal / skew |
| --- | --- | --- | --- |
| decode-only step, B=8, agg. KV ~35k | 19.58 / 19.48 (**0.995×**) | 10.38 / 11.27 (**1.086×**) | 14.32 / 15.09 (**1.054×**) |
| decode p50 per third of the cell | 19.58 19.58 19.58 / 19.47 19.47 19.49 | 10.37 10.38 10.38 / 11.26 11.26 11.27 | 14.31 14.32 14.32 / 15.08 15.09 15.10 |
| 512-chunk prefill step at 8k depth | 90.8 / 91.0 | 49.9 / 47.1 | 64.2 / 61.2 |
| at 12k depth | 104.0 / 104.5 | 56.1 / 55.0 | 70.1 / 68.8 |

The decode difference is stable across all three repeats. Per-step spread is ±0.05 ms, well
below the 0.8–0.9 ms gap. Aggregate decode KV is matched (median 35.2k vs 35.1k). The steps run
under the same full CUDA graph (padded to 8). A plausible cause, **not measured here**: on the
A100's faster, shorter decode step, the longest sequence's split-KV work becomes the critical path
(8.2k max KV vs 4.4k). On the slower L20 it is hidden. Prefill steps show no skew penalty
(differences are inside the p95 spread, n=12).

Consequence for the L20 conclusion: *"Killed: decode KV max/variance as a scheduling feature"* is
**hardware-specific**. On A100 an aggregate decode-KV term under-prices a skewed decode batch by
5–9%. That is small next to the 1.6–2× prefill geometry gap, but it is a systematic bias, not noise.

## Reproduce

```bash
A=benchmarks/results/a100-prefill-cost-geometry
mkdir -p /tmp/a100-trace-4b && cp $A/raw/Qwen3-4B/trace/*.gz /tmp/a100-trace-4b/ && gunzip /tmp/a100-trace-4b/*.gz
PYTHONPATH=scripts python scripts/analyze_step_cost_v2.py --trace-dir /tmp/a100-trace-4b --csv /tmp/steps-4b.csv --output /tmp/pred-4b.json --exclude-over-median-x 10 --exclude-first-iteration
python scripts/analyze_partition_geometry.py --steps A100-Qwen3-4B=$A/raw/Qwen3-4B/steps.csv A100-Qwen3-8B=$A/raw/Qwen3-8B/steps.csv L20-Qwen3-4B=benchmarks/results/l20-prefill-cost-geometry/steps.csv
python scripts/analyze_decode_kv_skew.py --steps A100-Qwen3-4B=$A/raw/Qwen3-4B/steps.csv A100-Qwen3-8B=$A/raw/Qwen3-8B/steps.csv L20-Qwen3-4B=benchmarks/results/l20-prefill-cost-geometry/steps.csv
```

Checked before commit: the regenerated `steps.csv` is byte-identical to the committed one, and the
regenerated predictor MAEs match `predictors-clean.json` for every split and model, for both models.

## Not known

- Whether the live-controller gain (L20 §5: +49–180% over the aggregate controller, +5–12% over a
  hindsight fixed budget) holds on A100. It was not re-run.
- H100 / Hopper attention kernels (FA3) and other attention shapes (different KV-head counts would
  change the slope; here 4B and 8B share it).
- The mechanism of the A100 decode-skew penalty (it would need a kernel-level profile).
