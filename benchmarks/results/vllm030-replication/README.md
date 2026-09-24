# Prefill cost geometry on vLLM 0.30.0 (L20, Qwen3-4B)

**Question.** Every other geometry artifact uses vLLM 0.29.0, and 0.30.0 was released on 2026-09-22.
Is the result specific to one release?

**Answer.** No, the result is not release-specific. Pre-registered in addendum 9 H (moved from the
A100 to the L20 before any 0.30 run):

| prediction | 0.29 (L20) | 0.30 (L20) | verdict |
| --- | --- | --- | --- |
| H1: partition gap 1×2048 / 8×256 at 12–16k within ±15% of 1.88× | 1.88× (336.1 / 178.9 ms) | **1.87×** (335.5 / 179.1 ms) | held |
| H2: pairing swap PS1 and PS2 | 0.86–1.01, control −0.09 ms | **0.86–1.01, control −0.05 ms** | held |
| H3: M2n primary MAE ≤ 5 ms and below M2 | 3.1 ms (M2 10.4) | **6.30 ms** (M2 14.9) | **failed** |

Every partition median agrees with 0.29 to within 1%, and the pairing swap agrees to within 0.7 ms
per config.

**Why H3 failed: 0.30 stalls, not geometry.**
- 0.30 adds **2.2–4.0 s stalls** to prefill steps mid-run: 17 steps in 8 of 19 cells, sometimes
  twice in one cell (steps 68 and 144 of `part2-1x2048`).
- 0.29 had none in the same 28 cells.
- Several of these stalls are under the pre-registered 10× median exclusion, so they stay in the
  fits (for example 8.6× in `part2-4x512`, whose proxy-only fit becomes 237 + 2.6·M instead of
  152 + 6.5·M).
- A post-hoc diagnostic that drops steps > 5× their cell median gives M2n **2.80 ms** (reverse
  2.36 ms), the 0.29 values. This is labelled post hoc; H3 stays failed.
- The stalls look like on-demand work at first sight of a new shape. 0.30's release notes
  advertise much faster CUDA-graph initialization, and deferring capture would produce exactly this
  pattern. That is a hypothesis; it was not verified.

## Setup

| item | value |
| --- | --- |
| engine | vLLM 0.30.0 from the Tencent PyPI mirror in its own venv ([`campaign/`](campaign/)); `pip-freeze.txt` in `raw/` |
| toolchain fix | 0.30's wheel set pulled `nvidia-cuda-nvcc` 13.4, `nvidia-nvvm` 13.4, `nvidia-cuda-crt` 13.4 and `nvidia-cuda-cccl` 13.3 against 13.0 CUDA headers, so FlashInfer's JIT of its sampler failed (compiler/header mismatch, then PTX 9.4 vs ptxas 9.0). These four compile-time packages were pinned to the 13.0 versions of the working 0.29 venv. vLLM and torch binaries are unchanged. |
| tracer | tracer v2, re-anchored: 0.30 adds `cudagraph_stats = None` before the common case in `execute_model`, and `apply_tracer_v2.py` now accepts both versions |
| API | `--enable-scale-out` (0.30 makes `/inference/v1/generate` opt-in) |
| cells | every cell of the shape campaigns, on Qwen3-4B, with the same arguments as 0.29, plus the pairing swap ([`../prefill-pairing-swap/`](../prefill-pairing-swap/README.md)) |

Files: [`partition.md`](partition.md), [`m2-variants.md`](m2-variants.md), `raw/L20-Qwen3-4B/`
(gzipped traces, `steps.csv`, predictors).
