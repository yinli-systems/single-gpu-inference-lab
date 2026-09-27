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

## Addendum 1 (2026-09-27): post hoc split-term check on L20/A100, and a tile-schedule hypothesis; before any H100 data is analyzed

The H100 shape queue had started when this was written; no H100 output file has been opened.

**Post hoc, on already-published L20 and A100 data** (`scripts/analyze_split_term.py`):

| data | c_X (ms/M) | c_S (ms/M) | c_S / c_X | single slope | pairing measured / (c_X·ΔX) | M2n / M2s primary MAE |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| L20 Qwen3-4B | 6.33 | 8.81 | 1.39 | 6.50 | 0.89–1.03 | 3.07 / 19.86 |
| A100 Qwen3-4B | 3.23 | 5.70 | 1.76 | 3.30 | 0.67–0.91 | 1.92 / 9.14 |
| A100 Qwen3-8B | 3.26 | 6.15 | 1.89 | 3.32 | — | 1.66 / 11.84 |

- A unit of self work does cost more than a unit of cross work (HP5's direction), but the fitted
  c_X is within 3% of the single slope, because X dominates the shape-cell steps. **The split term
  does not explain the A100 PS1 failure.** On these data HP4 would fail exactly like PS1, and HM4
  would fail (M2s extrapolates badly: on one-prefill training data S is nearly a function of the
  token count). HP4 and HM4 stay registered as written and will be reported as they come out.

**New hypothesis (tile schedule).** The pairing-swap steps are small (budget q_a + q_b + 1, one
decode row). A varlen prefill kernel launches one CTA per (request, query tile, query head); a
CTA's work is its number of key tiles, n = ceil((k + end of its query tile) / B_k). With few CTAs
per SM the step is bounded by how the CTAs pack onto SMs, not by total work: state B (large chunk
shallow, small chunk deep) has fewer but longer deep CTAs. Total work then over-predicts Δ, more
so on a GPU with more SMs per unit of work.

Tile-schedule model **TS**: kernel time = a + c · makespan, where makespan is the longest-first
(LPT) packing of all CTA works (in key tiles) onto `num_SMs` slots; (B_q, B_k) is chosen from
{64, 128} × {64, 128, 176} and (a, c) fitted, on the single-request grid of HK3 only. The work
model **W** is kernel time = a + b · W fitted on the same grid.

- **HK4.** On the isolated-kernel pairing configurations, TS predicts Δ with lower MAE than W, for
  FA2 and for FA3.
- **HK5.** The W model over-predicts the kernel pairing Δ (measured / predicted < 0.9) for at least
  3 of 5 configurations on FA3; the TS model is within ±25% for at least 4 of 5.

## Addendum 2 (2026-09-27): a second engine (SGLang) and two published predictors; before any of it runs

**Prior art found in a literature check today** (it narrows claim C3; the report is updated either way):
- KernelSight-LM (arXiv 2606.28565) already prices chunked-prefill attention *per request* ("the
  prefill chunk over its cached history plus its own causal triangle"). Per-request attention work
  as a feature is therefore not new; what is tested here is that the *aggregate* states used by
  schedulers and simulators cannot see it.
- Vidur (MLSys 2024; microsoft/vidur @ 8383d29, `sklearn_execution_time_predictor.py` l. 852–870)
  looks up batch prefill-attention time at (Σ kv_cache_size, round(√Σ chunk²)²).
- LLMVisor (arXiv 2608.08382): T = β + a₁Σpᵢ + a₂Σcᵢ + a₃Σpᵢ² + a₄|B|.
Both are exactly invariant under a pairing swap.

**SGLang pairing swap** (`scripts/measure_pairing_swap_sglang.py`, tracer
`integrations/sglang_step_tracer`; SGLang 0.5.20, torch 2.13.0+cu130, default attention backend,
H100, Qwen3-4B, same six configurations, 20 trials per state). SGLang runs extend batches without
decode rows, so its pair step is the two prefills alone.
- **SG1.** Δ = median(A) − median(B) > 0 for every configuration with |ΔW| ≥ 3M.
- **SG2.** Control |Δ| ≤ 1 ms.
- **SG3.** For at least 4 of 5 configurations, the SGLang Δ is within ±30% of the vLLM H100 Δ of the
  same configuration.

**Published predictors** (`scripts/analyze_published_predictors.py`):
- **VL1.** The Vidur prefill-attention key is identical for A and B in all six configurations, so
  its error floor on each measured pair is |Δ|/2. Reported for L20, A100, H100 (vLLM) and SGLang.
- **VL2.** The LLMVisor formula, fitted by least squares on the primary-split training steps, has
  primary-split MAE ≥ 5× the M2n MAE on every dataset with shape cells (L20, A100, H100). The
  L20 and A100 values are computed after this addendum is committed.

## Addendum 3 (2026-09-27): request ordering with geometry pricing (live, H100); before any run

The report's negative result says per-step budgeting cannot change total work and that TTFT +
TPOT goodput is set by admission and ordering. This tests ordering directly, using vLLM 0.29's
own `--scheduling-policy priority` and a client-side priority (`scripts/replay_trace_serving.py
--priority`); no engine change.

**Arms** (H100, Qwen3-4B, vLLM 0.29.0, prefix caching on, MBT 2048, `max-num-seqs` 256, 240 s window):
- `fcfs`: default policy.
- `prompt`: priority = prompt tokens (cache-unaware SJF).
- `uncached`: priority = prompt tokens minus the estimated prefix-cache depth k₀ (cache-aware SJF).
- `cost`: priority = a·L + b·(L·k₀ + L(L+1)/2) with L = uncached tokens: the geometry price of the
  remaining prefill at its cached depth. (a, b) = OLS of step `cuda_ms` on (ctx_tokens/1e3, W/1e6)
  over the H100 Qwen3-4B one-prefill shape steps (filtered), fixed before the first run.
- k₀ is estimated client-side as the leading run of 512-token blocks already sent by an earlier
  arrival (no eviction model).

**Cells:** Mooncake tool-agent (@3 s jitter, seed 0) at rate scale 0.2 and 0.4, 2 repeats each;
Azure 2023 code at rate scale 0.5 (no shared prefixes, so k₀ = 0 and `uncached` = `prompt`),
1 repeat. Order of arms is rotated per repeat.

**Offline, before any run:** on the Mooncake windows, `cost` and `uncached` orders are discordant
on only 4.4–4.6% of pairs of requests arriving within 5 s (Kendall τ 0.91–0.92), while `prompt`
vs `cost` has τ 0.64. The geometry term therefore changes few ordering decisions beyond
cache-awareness.

**Predictions** (goodput = requests with TTFT ≤ 5 s and TPOT ≤ 100 ms per second of window;
repeats averaged):
- **O1.** On both Mooncake cells, `uncached` and `cost` each have mean TTFT ≤ 0.8× `fcfs`.
- **O2.** On both Mooncake cells, `cost` mean TTFT is within ±5% of `uncached` (geometry adds
  little to ordering beyond cache-awareness, as the offline discordance suggests).
- **O3.** On both Mooncake cells, `prompt` mean TTFT is higher than `uncached` (cache-unaware
  ordering is worse).
- **O4.** Goodput of the best priority arm ≥ 1.05× `fcfs` on at least one Mooncake cell.
- **O5 (descriptive).** p99 TTFT of every priority arm against `fcfs` is reported (SJF can
  starve long requests).

## Addendum 4 (2026-09-27): wave staircase of a deep chunk (engine level, H100); before it runs

**Post hoc observation that motivates it** (H100 Qwen3-4B pairing swap, already analyzed): the
state-A step time is 30.9–31.2 ms for deep chunks of 640, 768 and 896 tokens at 16k (configs 2, 0,
1). The pairing Δ does not scale with ΔW (7.5 ms at ΔW 6.3M, 7.6 ms at 9.4M, 5.3 ms at 3.1M), and
the single-slope ratio scatters on both sides of 1 (0.76–1.58). This is what the tile-schedule
hypothesis of addendum 1 implies: a query tile of a request at depth K is one CTA per query head
whose length grows with K, and a chunk of q tokens launches ceil(q/128) × 32 of them on 132 SMs.

**Test** (`scripts/measure_chunk_staircase.py`; vLLM 0.29 FA3, Qwen3-4B, one decode row, one
prefill of q tokens at cached depth K, q ∈ {128, 256, …, 1280}, 20 trials per q in shuffled
rounds). Depths K = 16384 and 32768 (long CTAs), and K = 4096 as control. Predicted with
B_q = 128, 32 query heads, 132 SMs: 1 wave of deep CTAs up to q = 512, 2 waves for 640–1024,
3 waves from 1152.
- **ST1.** At K = 16384 and at K = 32768, the increments 512→640 and 1024→1152 are each ≥ 3× the
  median of the other seven increments.
- **ST2.** At K = 16384 and at K = 32768, the total rise over the plateau 640→1024 is smaller than
  the single jump 512→640.
- **ST3 (control).** At K = 4096 no increment is ≥ 3× the median of the others.
If the staircase sits elsewhere (e.g. because FA3 packs GQA heads or uses a different B_q), ST1
fails as registered; the observed positions are then reported as exploratory.

## Addendum 5 (2026-09-27): heavier ordering cells chosen by a frozen load rule; before any priority arm is seen

The only ordering run finished so far is `mooncake-x0.2-fcfs-r0`: 245 requests, TTFT p50/p90/p99
0.07/0.48/0.86 s, goodput equal to the offered rate. The H100 is not queueing at the A100's rate
scale, so ordering cannot matter there; ×0.4 is only twice that. The addendum-3 cells run as
registered and are reported as registered. Added, after them:

- **Load probe (fcfs only):** Mooncake tool-agent at rate scale 1, 2 and 4, one run each.
- **Frozen rule:** S* = the smallest probed scale whose fcfs TTFT p50 ≥ 1 s; if none, the largest.
  Only the fcfs arm is used to choose the load.
- **Heavy cells:** Mooncake at S* and S*/2, all four arms, 2 repeats, arms rotated per repeat.
- O1–O5 are evaluated on the heavy cells exactly as written in addendum 3, and reported
  separately from the registered light cells.

## Addendum 6 (2026-09-27): light ordering cells stopped after repeat 0; before any heavy-cell run

Deviation, decided with the repository owner after the first repeat of the addendum-3 cells:
Mooncake ×0.2 and ×0.4 completed repeat 0 for all four arms (plus `uncached` and `cost` of
repeat 1 at ×0.2); the remaining repeat-1 runs and the Azure-code control were not run. At these
loads the H100 does not queue (fcfs TTFT p50 0.07–0.08 s, goodput equal to the offered rate in every
arm, 24/245 and ~56/490 requests above 0.5 s TTFT in every arm), so further repeats cannot change
the verdict and the GPU time goes to the addendum-5 heavy cells instead. O1–O5 on the light cells
are reported from the runs that exist, labelled "repeat 0 only". One request per affected run
(the same request index in every arm, including fcfs; ~20k-token prompts) ends with a client-side
`ServerDisconnectedError` with nothing in the server log; it is kept in the counts and reported.

## Addendum 7 (2026-09-27): fix the cache-depth estimate and bound the tail (aging); before any run

**Diagnosis of O3, from the addendum-5 runs (post hoc).** `uncached` and `cost` estimated each
request's cached depth k₀ assuming nothing is evicted. At rate scale 1 that estimate gives a
token-weighted prefix hit of 51.4%, but vLLM's own log reports **40.6%** (KV cache 455,008 tokens,
93–97% used at peak). A block-level LRU of the logged capacity, with no tuning, gives **42.6%**
(×0.5: 40.4% vs 40.3% measured). The cache-aware arms were ranking by an over-optimistic k₀, which
can explain why cache-unaware `prompt` did not lose (O3 failed).

**New arms** (H100, Qwen3-4B, Mooncake tool-agent @3 s, rate scale 1, same server settings and
cost coefficients as addendum 3; `--cache-capacity-tokens 455008`):
- `uncached-lru`, `cost-lru`: as `uncached` and `cost`, with k₀ from the LRU.
- `cost-lru-age(β)`: priority = cost-lru + β · arrival time (both µs), which at every instant orders
  requests by cost − β · time already waited (aging). No engine change.

Offline, before any run: with the LRU, `cost-lru` and `uncached-lru` disagree on 2.2% of pairs
arriving within 5 s; the median cost is 13.7 ms and the P90 259 ms.

**β is chosen on a separate window.** Tuning window: the same trace, trace time 240–480 s, rate
scale 1 (1,308 requests vs 1,361 in the primary window). One run each of `fcfs` and
`cost-lru-age(β)` for β ∈ {0.01, 0.05, 0.25}. Frozen rule: β* = the β with the highest goodput among
those whose TTFT p99 ≤ the tuning-window `fcfs` p99; if none, the β with the lowest p99. The primary
window is then run with `uncached-lru`, `cost-lru` and `cost-lru-age(β*)`, 2 repeats each, and
compared with the existing `fcfs` and `prompt` runs (2 repeats each) of addendum 5.

**Predictions** (primary window, means of 2 repeats; goodput at TTFT ≤ 5 s, TPOT ≤ 100 ms):
- **I1.** `uncached-lru` mean TTFT ≤ `prompt` mean TTFT (a correct k₀ makes cache-awareness help).
- **I2.** `cost-lru` mean TTFT within ±5% of `uncached-lru` (geometry still adds little to ordering).
- **I3.** `cost-lru-age(β*)`: TTFT p99 ≤ `fcfs` p99 **and** goodput ≥ 1.8× `fcfs`.
- **I4 (descriptive).** Goodput and p99 of every arm, including the addendum-5 arms.

## Addendum 8 (2026-09-27): engine-side SLO-aware ordering (online Moore–Hodgson); before any run

Client-side static priorities cannot use the queue state, and aging traded almost all of the goodput
gain for the tail (addendum 7 tuning window: β = 0.25 gives p99 13.6 s vs fcfs 14.6 s but goodput
1.06× fcfs). Goodput counts requests whose TTFT meets a deadline, which is the objective of
1‖ΣUⱼ, for which Moore–Hodgson is optimal on one machine with known processing times.

**Arm `slo-mh`** (engine patch, `benchmarks/results/h100-request-ordering/patches/exp_slo_scheduler.py`,
appended to vLLM 0.29's `scheduler.py` by `apply_slo_scheduler.py`, inert unless
`VLLM_EXP_SLO_ORDER=1`). At every `schedule()` call:
- each waiting request gets deadline = engine arrival + 5 s − 0.1 s and processing time
  κ · (a·L + b·(L·k₀ + L(L+1)/2)) with a, b as in addendum 3 and k₀ its **exact** current prefix-cache
  hit from the KV cache manager (read-only lookup, refreshed at most once per second per request);
- requests are taken in deadline order (= arrival order) after the remaining prefill of running
  requests; whenever the running total misses a deadline, the most expensive accepted request (latest
  arrival on ties) moves to the late class; requests past their deadline are late;
- on-time requests get priority 0 and late ones 1; the queue's tie-break is arrival time, so each
  class is FCFS; running requests get priority 0, so preemption picks the latest arrival as usual;
- κ converts predicted prefill ms into wall time and is re-estimated online (wall time over
  predicted prefill ms of the last ~2 s of backlogged steps, EWMA 0.2, start 1.5). No parameter is
  tuned on either trace window.

**Cells:** Mooncake tool-agent @3 s at rate scale 1 (the addendum-5 primary window): `slo-mh`, 2
repeats, compared with every existing arm there. Rate scale 1.5 (robustness, 1 repeat each):
`fcfs`, `prompt`, `uncached-lru`, `slo-mh`.

**Predictions:**
- **M1.** At scale 1, `slo-mh` goodput is the highest of all arms (fcfs, prompt, uncached, cost,
  uncached-lru, cost-lru, cost-lru-age).
- **M2.** `slo-mh` goodput ≥ 1.8× `fcfs` at scale 1 and at scale 1.5.
- **M3.** Reorder overhead: the median over logged seconds of the per-second maximum is ≤ 2 ms.
- **M4 (descriptive).** Mean, p50 and p99 TTFT of `slo-mh`; the late class is expected to wait, so
  p99 is expected to be worse than `fcfs`.
