# Per-request heterogeneous pre-draft K for speculative decoding — screened and killed (L20, Qwen3-4B, vLLM 0.29)

**Hypothesis (pre-registered).** vLLM's shipped adaptive verification (DSpark) decides *after* drafting
how many draft slots to verify, so 100% of draft work is always executed. If per-request utility is
heterogeneous, choosing K per request *before* drafting should recover the wasted draft compute.
Gate: oracle heterogeneous pre-draft K must beat fixed-K + adaptive verification by ≥5% goodput
(≥10% to justify a runtime prototype).

**Verdict: killed.** Composed oracle bound C/B = 0.94–1.00 on every measured mixed cell. Two
measured facts close it: (1) the DSpark draft pass is per-row-fixed, nearly independent of K
(K=2→7 costs +5–20%), so shrinking K buys almost nothing on the draft side; (2) DSpark proposals are
not prefix-invariant — a smaller block accepts fewer tokens than the truncated K=7 proposal in
14–16% of positions (K≤3) — so small K also loses reward. No row in any batch is better off at K=0.
Not run on H100.

## Setup

Qwen3-4B bf16, one L20, vLLM 0.29.0 + tracer v3 (`patches/`: per-step `draft_ms` around
`speculator.propose`, per-request `[req, depth, drafted, accepted]` from the scheduler), `TRITON_ATTN`
(the only backend on SM89 that adaptive verification accepts), `--max-model-len 16384 --max-num-seqs 64
--no-enable-prefix-caching`. Drafters: `deepseek-ai/dspark_qwen3_4b_block7` (DSpark, parallel block
of 7) and `AngelSlim/Qwen3-4B_eagle3` (autoregressive; weak here: 0.6–1.4 accepted/row/step, kept as a
drafter-dependence control). Harness [`scripts/measure_spec_geometry.py`](../../../scripts/measure_spec_geometry.py):
closed batches of B prompts of a class, greedy, fixed output length; mixed batches release the short
prompts once every long prompt has its first token so both classes co-decode. Analysis on the
co-decoding window only ([`scripts/analyze_spec_geometry.py`](../../../scripts/analyze_spec_geometry.py)).
Raw JSON/JSONL under `raw/` (commands in `raw/campaign23.sh`, `raw/campaign24.sh`; lab commits
`fa0d79a` / `dd492a1`).

## Campaign23 — screening and mechanism (no-spec, DSpark K=7, DSpark K=7 + adaptive, EAGLE3 K=5; 6 classes; B 8–64; 2 repeats)

Full T1–T4 tables: [`campaign23-tables.md`](campaign23-tables.md).

- **T1 draft share**: DSpark 20–24% of the step across class and B (4.0/18.3 ms at B=8 short →
  15.8/68.6 at B=32/6k); 25–31% under adaptive verification because only the non-draft part shrinks.
  EAGLE3 12–19%.
- **T2 accepted prefix per row per step**: code P25/50/75/90 = 1/3/6/7 (mean 3.0–4.4, 10–18% zero);
  prose 0/1/3/4 (mean 1.6–1.8, 27–31% zero).
- **T3 mixed batches**: utility is heterogeneous by content class, not depth (mix-ls: 6k-code rows
  3.4–4.4 vs 0k-prose 1.7; mix-sl: 0k-code 3.0–3.3 vs 4k-prose 1.5–1.6).
- **T4 what adaptive verification saves**: nothing at B ≤ 16 (identical step time, 0–8% fewer
  tokens → slightly lower goodput); at B=32/64 it cuts non-draft time only (code-short B=64 44.9 →
  34.4 ms, +17% goodput; prose-short B=64 44.8 → 24.8 ms, +39% — fixed K7 is below no-spec there,
  3116 vs 4008 tok/s, adaptive lifts it to 4348). Draft time unchanged in every cell.
- **{0,7} oracle**: rows whose per-row token rate is higher without speculation: 0/B in every cell —
  the smallest measured utility (prose, 1.6 accepted) still yields 2.6 tokens per step at 1.4–2× the
  no-spec step time.

## Campaign24 — K-prefix invariance and the draft cost surface (DSpark K ∈ {1,2,3,5,7}, one server each; code/prose × short/3k/6k; B 4–32; 512-token completions)

**A. Invariance** (greedy; committed sequences drift across batch shapes without batch-invariant
kernels at median position ~115/512, so only pre-divergence positions are compared; test:
`accepted_K(p) == min(accepted_7(p), K)` at positions where both runs start a step):

| K | coincident positions | equal | K-block accepts fewer | accepts more |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8,376 | 80.8% | 16.5% | 2.8% |
| 2 | 8,271 | 80.3% | 16.0% | 3.7% |
| 3 | 8,433 | 81.2% | 15.2% | 4.6% |
| 5 | 9,253 | 87.7% | 8.3% | 4.5% |

Not invariant: the parallel block's noise positions help the early positions; a K=1 block loses
≈0.14 accepted tokens per step net. A K=7 trace therefore *over*-estimates small-K reward.

**B. Cost surface** T_draft(B, L, K), median ms over co-decode steps (draft | step | non-draft):

| class | B | K=1 | K=2 | K=3 | K=5 | K=7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| code-short | 32 | 3.7 / 19.6 | 4.3 / 20.6 | 4.7 / 21.0 | 5.6 / 25.4 | 7.0 / 31.6 |
| code-long (6k) | 16 | 6.1 / 40.6 | 9.9 / 45.3 | 10.0 / 45.5 | 10.6 / 46.8 | 10.9 / 47.0 |
| prose-long (6k) | 32 | 7.8 / 51.6 | 13.1 / 57.4 | 13.6 / 57.7 | 14.5 / 63.0 | 16.1 / 70.0 |

Draft cost is dominated by per-row context work (the drafter's KV over the context); K=2→7 adds
5–20%. What grows with K is the verify side (B·(K+1) target tokens crossing graph capture sizes) —
the part adaptive verification already trims. Full surface in `campaign24-analysis.json`.

**C. Composed oracle bound** (per-class K chosen to maximise per-row reward/cost with the measured
surface and the invariance penalty; draft attributed per row; verify side taken from the measured
adaptive cell; "measured surface" = every (class, B, K) used exists in the trace set):

| mixed batch | B | adaptive: step / draft / goodput | oracle K per class | draft hetero vs K7 | C upper | C/B |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| code-short + prose-long | 8 | 26.7 / 5.3 ms / 1,001 | K7 / K5 | 4.9 vs 5.0 | 991 | 0.990 |
| code-short + prose-long | 16 | 30.3 / 6.8 / 1,812 | K7 / K7 | 6.9 vs 6.9 | 1,812 | 1.000 |
| code-short + prose-long | 32 | 43.4 / 10.9 / 2,594 | K5 / K5 | 10.0 vs 11.6 | 2,444 | 0.942 |
| code-long + prose-short | 8 | 30.6 / 5.9 / 895 | K7 / K7 | 5.5 vs 5.5 | 895 | 1.000 |
| code-long + prose-short | 16 | 34.8 / 7.6 / 1,796 | K7 / K7 | 7.9 vs 7.9 | 1,796 | 1.000 |

## What survives

- Fixed K7 is net-negative at large B for low-acceptance content (prose-short B=64: below no-spec);
  the shipped batch-size → K table and adaptive verification both fix that — no new mechanism needed.
- Adaptive verification's benefit starts at B ≥ 32; at B ≤ 16 it is a small loss (0–8%) — a fixed
  control overhead worth a causal decomposition, not a research line.
- The startup cost profile is keyed by `num_reqs` / `num_target_tokens` at a fixed synthetic context
  (8192); the measured draft and verify costs depend strongly on actual context (draft 4.0 → 7.0 →
  16.1 ms at B=32 for 0.1k / 3k / 6k). Whether that mis-prices the verification budget is the next
  question (campaign25).

## Limitations

One GPU (L20, Triton attention), one model, one DSpark checkpoint; synthetic prompts; the oracle
bound is a composition of measured homogeneous cells (labelled), not a heterogeneous execution; the
invariance test is on accepted counts, not proposal ids (draft ids are placeholders in the scheduler
under async scheduling); EAGLE3 was not composed (acceptance too low to matter).
