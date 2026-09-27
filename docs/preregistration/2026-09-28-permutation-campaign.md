# Pre-registration: controlled permutation campaign on the A100 (2026-09-28)

Committed before any campaign data exists. The only runs on this pod so far were harness smokes (two
blocks per config) used to check that every state's requests land in one engine step; their timing
was not looked at and the files were deleted unread except for capture counts.

## Why

The pairing swap (two requests) and live replay (observational) are the evidence that marginal-only
representations miss cost. This campaign is the controlled version at scale: for fixed chunk and depth
multisets, it runs many different pairings of the same marginals and measures the step time of each.

## Setup (fixed)

- RunPod A100-SXM4-80GB (driver 580.126.16, CUDA default 12.8), vLLM 0.29.0, torch 2.13.0+cu130, tracer
  v2 (`apply_tracer_v2.py`), Qwen3-4B bf16, FLASH_ATTN backend as selected by vLLM (expected FA2 on sm80).
- Harness `scripts/measure_pairing_permutations.py`, analysis `scripts/analyze_permutation_campaign.py`,
  campaign `benchmarks/results/a100-pairing-permutations/campaign/campaign_a100_permutations.sh`.
- Configs (depths k; chunk multiset q):
  - n2: k = {4k, 16k}; q = {256, 768}
  - n4: k = {0, 4k, 12k, 24k}; q = {128, 256, 512, 1024}
  - n8: k = {0, 2k, 4k, 6k, 8k, 12k, 16k, 24k}; q = {64, 128, …, 512}
  - n16: k = {0, 1k, …, 28k} (16 values); q = {32, 64, …, 512}
- States per config: `same` (C_max), `opposite` (C_min), 6 random pairings, `eqC-a` / `eqC-b` (two
  different pairings with equal C, n ≥ 4), and `order-null` (the `same` pairing submitted in reverse
  order: an identical state).
- Each state's n requests are enqueued while scheduling is paused (`pause_generation(mode="keep",
  clear_cache=False)`) and one background request decodes; scheduling then resumes with
  `max_num_batched_tokens = Σq + 1`, so the next step is exactly the n chunks plus one decode row.
- Design: 12 randomized complete blocks per (config, layout seed), 2 warm-up rounds dropped, fresh suffix
  tokens every trial. Layout seeds 0 and 1 create the prefixes in different orders, which changes their
  physical KV block placement. Cells run interleaved: n2:0, n4:1, n8:0, n16:1, n2:1, n4:0, n8:1, n16:0.
  GPU clocks, power and temperature are logged every 0.5 s.
- Exclusions: only trials whose requests did not land in one step (reported as `missing`). No
  outlier removal.

## Primary outcome

**P1.** In every (config, layout) cell with n ≥ 4 (six cells), the Spearman correlation between the
coupling C and the median step time over the distinct pairings is ≥ 0.8.

## Secondary predictions

- **P2 (existence).** In every cell, T(same) − T(opposite) > 0: the 95% bootstrap CI over blocks of the
  paired median difference excludes 0.
- **P3 (null).** In every cell, `order-null` vs `same` is equivalent: the 90% bootstrap CI of the paired
  median difference lies inside ±ε, ε = max(0.5 ms, 2% of median T(same)).
- **P4 (layout).** For every config, each state's median under layout seed 0 and 1 is equivalent within
  ±ε (90% bootstrap CI), and the Spearman correlation of the state medians across seeds is ≥ 0.9.
- **P5 (C sufficiency).** For n4, n8, n16 and both seeds, `eqC-a` vs `eqC-b` is equivalent within ±ε.
  This asks whether the sum C, not the individual pairs, sets the cost on this GPU; it is expected to
  hold on FA2, and a failure would mean per-request structure beyond C matters (as on FA3, §5.1).
- **P6 (magnitude).** T(same) − T(opposite) is 0.6–1.0 × 3.23 ms/M × (C_max − C_min) in every cell (the
  A100 pairing swap came out at 0.65–0.89 of the partition slope).
- **Reported, not predicted:** the minimax floor (max − min state median)/2 per cell, the number of
  missing trials, and GPU clock/temperature ranges.
