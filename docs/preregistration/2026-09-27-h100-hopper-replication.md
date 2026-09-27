# Pre-registration: Hopper (H100, FA3) replication and the split cross/self cost term (2026-09-27)

Written and committed **before** any vLLM model step, pairing swap or attention kernel was timed on
the H100 used here. Before this commit the pod had only been used to install software, download
weights and import the patched vLLM. Nothing below has been measured on Hopper by this lab.

## Why

1. The report's open hardware item is Hopper. vLLM 0.29 on sm90 runs FlashAttention 3, a different
   kernel family from the FA2 used on the L20 and A100 (warp-specialized, asynchronous TMA/WGMMA
   pipelines, different tile shapes). If the pairing effect is a property of attention work and
   not of one kernel, it must survive this change.
2. **PS1 failed on the A100** (measured pairing Δ was 65–89% of slope × ΔW). The slope came from
   partition fits, where the cross term X = Σ qᵢkᵢ and the self term S = Σ qᵢ(qᵢ+1)/2 move together.
   The pairing swap moves only X. A mechanism that explains the failure, formed **after** seeing
   it: a causal chunk computes whole diagonal tiles of which about half is masked, so a unit of S
   costs more than a unit of X. A single slope fitted on partitions then overstates the price of X.
   This is tested prospectively on the H100 and only **post hoc** on the L20 and A100 data, and is
   labelled that way everywhere.

## Setup (fixed)

- RunPod, 1× H100 80GB HBM3 (SXM), driver 580.126.09. vLLM 0.29.0, torch 2.13.0+cu130, tracer v2
  applied with `apply_tracer_v2.py` (unchanged). Attention backend: whatever vLLM 0.29 selects on
  sm90 (expected FLASH_ATTN with FA3); the selected backend and FA version are read from server
  logs and recorded.
- Models: Qwen3-4B, Qwen3-8B, Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct (bf16), from the Hub.
- Shape cells: exactly the A100 cells (`campaign_a100_shape.sh`: 6 partition cells, 4 context,
  3 load, 2 skew; 3 repeats, fresh server per cell). Script:
  `benchmarks/results/h100-prefill-cost-geometry/campaign/campaign_h100_shape.sh`.
- Pairing swap: `scripts/measure_pairing_swap.py`, unchanged, the same six configurations, 20 trials
  per state, Qwen3-4B and Qwen3-8B.
- Filters and splits: `--exclude-over-median-x 10 --exclude-first-iteration`; primary split
  trains on one-prefill steps and tests on multi-prefill steps; reverse split the other way.

## Definitions

- **M2s** = M2n with the attention-work feature replaced by two features, X/1e6 and S/1e6
  (`attn_cross`, `attn_self`), same Ridge (λ = 0.01), same filters and splits. M2s is new here.
- **Pairing prediction from M2s**: fit M2s on *all* filtered prefill-containing steps of the H100
  shape cells of the same model (never on pairing-swap steps) and convert the X coefficient to raw
  units c_X (ms per million units). Predicted Δ = c_X · ΔX, with ΔX = (q_a − q_b)(k_a − k_b).
- **Single-slope prediction** (the published PS1 method): slope = mean of the six partition fits
  ms = a + b·W on the H100 partition cells of that model.

## Frozen predictions

**Pairing swap (Qwen3-4B and Qwen3-8B, H100):**
- **HP1 (existence).** Every configuration with |ΔW| ≥ 3M has Δ = median(A) − median(B) > 0.
- **HP2 (control).** The ΔW = 0 control has |Δ| ≤ 1 ms.
- **HP3 (single slope, expected to fail).** Measured / single-slope predicted Δ lies in 0.50–0.95
  for at least 4 of 5 configurations. That is, the A100 failure of PS1 repeats on Hopper.
- **HP4 (split term).** The M2s prediction c_X · ΔX is within ±25% of the measured Δ for at least
  4 of 5 configurations, on each model.
- **HP5 (ratio).** c_S / c_X > 1.2 on each H100 model (a unit of self work costs more than a unit
  of cross work).

**Partition geometry (H100):**
- **HG1.** Qwen3-4B, 12–16k aggregate KV, 1×2048 / 8×256 median ratio in 1.6–2.4×; Qwen3-8B in
  1.3–2.0×.
- **HG2 (collapse).** For each model and budget, the three partition slopes b are within 8% of their
  mean.
- **HG3 (slope).** Qwen3-4B partition slope 1.0–2.2 ms/M (A100 3.3 ms/M; FA3 on H100 expected
  roughly 1.5–3× faster per unit of attention work). Qwen3-8B within 10% of Qwen3-4B (same
  attention shape).

**Out-of-distribution prediction (H100, four models):**
- **HM1.** M2n primary-split MAE ≤ 5 ms and mean signed error within ±5 ms on every model.
- **HM2.** M0 primary-split MAE ≥ 10× M2n on every model.
- **HM3.** LPRS-style MLP (5 seeds, `analyze_learned_baseline.py` unchanged) primary-split MAE >
  M2n on every model.
- **HM4.** M2s primary-split MAE ≤ M2n MAE + 1 ms on every model (splitting the term must not hurt
  OOD prediction).

**Isolated attention kernels (H100, `scripts/measure_kernel_pairing.py`, new):**
One attention layer's prefill call, run exactly as vLLM calls it (paged KV, block size 16, bf16,
causal, per-request depths), with the Qwen3-4B shape (32 query / 8 KV heads, d = 128) and the
Qwen2.5-7B shape (28 / 4, d = 128). Batch = the same two prefill requests as the pairing-swap
configuration plus one decode row. Kernels: vLLM FA2, vLLM FA3, and FlashInfer if a working build
installs against this torch (if it does not, that is recorded and it is dropped). CUDA events,
interleaved A/B, 200 timed calls per state after warm-up.
- **HK1.** For every kernel and every configuration with |ΔW| ≥ 3M, median(A) > median(B); the
  control has |Δ| ≤ 2% of its median.
- **HK2 (closure).** For FA3 with the Qwen3-4B shape, 36 × (kernel Δ) is 70–130% of the model-step
  Δ measured by HP1, for at least 4 of 5 configurations.
- **HK3 (split term at kernel level).** On a grid of single requests, q ∈ {128, 256, 512, 1024,
  2048} × k ∈ {0, 2k, 4k, 8k, 16k, 32k}, fitting kernel time = a + c_X·(qk) + c_S·q(q+1)/2 gives
  c_S / c_X > 1.2 for FA2 and FA3.

## What is and is not claimed ahead of time

- HP3 is a prediction that the old single-slope method fails again. If it *holds* on the A100
  data retrospectively (it did not) or fails on the H100 in the other direction, the split-term
  explanation is weakened, and that will be written down.
- If HP4 or HK3 fails, the mechanism above is wrong or incomplete, and the report will keep the
  single-slope proxy described as an approximation.
- The L20 and A100 M2s numbers will be computed from the existing `steps.csv` and pairing-swap
  traces and reported as **post hoc** in every table.
- Harness smoke runs (one config, few trials) are allowed to validate that scripts run; their
  numbers are discarded and not reported.
