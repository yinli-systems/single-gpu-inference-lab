# Task 27 — speculative-decoding acceptance skew across synchronized DP/EP ranks

Question: under lockstep DP2/EP2, does a low-acceptance rank pay the high-acceptance rank's verification width
(padded execution tokens, step period), and does the high-acceptance rank keep its rate next to a low-acceptance peer?

Setup: 2×RTX 4090, vLLM 0.29.0, Qwen1.5-MoE-A2.7B-Chat, DP2/EP2 (`allgather_reducescatter`), multi-port external LB
(rank r = port 8300+r), graph mode, q=512, n-gram speculation server-wide (`{"method":"ngram","num_speculative_tokens":4,
"prompt_lookup_max":4,"prompt_lookup_min":2}`), B ∈ {8, 32} temperature-0 decode streams per rank, 10-s windows, 3 repeats.
Prompt *kind* per rank sets the acceptance regime: `rep` = repeated 8-token pattern (drafts accepted), `rnd` = random ids.
Harness `scripts/dp_ep/measure_spec.py` (generation rate from the `/metrics` generation counter — SSE chunks carry every
accepted token of a step, so chunk counts undercount), analysis `scripts/dp_ep/analyze_spec.py`, job script `sbatch_c27.sh`.

## Round 1 (job 1602607, node wqd10nba07g5, GPUs NUMA 2/0 same socket; `raw/c27-1602607.json`) — null, weak test

Pooled medians over 3 repeats, per rank (rate = tok/s generated in the window; accept = accepted / drafted):

| cell | B | rank kind | gen tok/s | accept | drafts per req-step | ITL p50 / p95 | period p50 | width p50 (tokens/step) | est. padded |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| rep\|rep | 8 | rep / rep | 1140 / 1332 | 0.79 / 0.93 | 0.95 / 0.97 | 26.4/36.2 · 26.3/36.8 | 26.2 | 40 / 40 | 40 |
| rnd\|rnd | 8 | rnd / rnd | 1298 / 1162 | 0.96 / 0.91 | 0.91 / 0.93 | 26.1/36.7 · 26.0/35.4 | 26.0 | 36 / 40 | 40 |
| rep\|rnd | 8 | rep / rnd | 1083 / 1221 | 0.87 / 0.90 | 0.80 / 0.89 | 25.9/35.5 · 26.0/36.6 | 25.8 | 32 / 36 | 40 |
| rnd\|rep | 8 | rnd / rep | 1080 / 1090 | 0.83 / 0.82 | 0.82 / 0.87 | 26.0/37.2 · 26.0/37.9 | 25.9 | 36 / 36 | 40 |
| rep\|rep | 32 | rep / rep | 2799 / 2886 | 0.89 / 0.88 | 0.64 / 0.68 | 34.0/40.4 · 34.0/41.2 | 33.5 | 105 / 111 | 112 |
| rnd\|rnd | 32 | rnd / rnd | 2839 / 2765 | 0.90 / 0.86 | 0.64 / 0.63 | 34.3/40.2 · 34.2/40.6 | 33.9 | 101 / 100 | 104 |
| rep\|rnd | 32 | rep / rnd | 2826 / 2759 | 0.90 / 0.88 | 0.65 / 0.62 | 34.3/44.8 · 34.1/41.4 | 34.0 | 100 / 107 | 112 |
| rnd\|rep | 32 | rnd / rep | 2652 / 2812 | 0.79 / 0.90 | 0.66 / 0.63 | 34.5/40.5 · 34.3/41.5 | 34.2 | 108 / 103 | 112 |

Skew effect (a kind's rank in a skewed cell vs the same kind's homogeneous cell): ITL p50 ×0.98–1.01, period ×0.98–1.02,
p95 ×0.97–1.10, generation rate ×0.88–0.99 (rate differences track the per-cell acceptance draw, not the period).

Reading: the `rnd` regime is **not** low-acceptance under greedy decoding — the continuation of random ids loops and the
n-gram lookup finds it (accept 0.79–0.96, drafts on 0.6–0.9 of request-steps, same width as `rep`), so every cell has the
same execution shape (width 36–40 at B=8, 100–111 at B=32) and there is nothing to skew. Round 1 is a null but a weak test.

## Round 2 (job 1603006, submitted 2026-09-19 ≈ 14:00 cluster clock; cells `rep|rep, rndT|rndT, rep|rndT, rndT|rep`)

`rndT` = random ids with seeded temperature-1.0 sampling: the sampled continuation has no 2-gram repeats, so no draft on most
steps → width ≈ 1 per request (genuinely low acceptance; seeded temperature sampling itself costs ~+1 ms/step, task 21).
Gate (pre-registered, unchanged): rank generation rate / ITL change < 5 % kill; > 10 % continue; > 10 % for group-aware
over independent per-rank control = strong. Predictions before the data: in `rep|rndT` the `rndT` rank's step period and
ITL follow the `rep` rank's (lockstep): ≈ the rep|rep period (26 / 34 ms at B=8/32) instead of the rndT|rndT period (≈ the
plain temperature-sampled decode period, ~15 / 23 ms) → ITL ×1.5–1.8 — that is the user-visible cost and what the gate scores.
Whether the rndT rank is *padded* to the peer's verification width (36–40 / 100–112 tokens per step vs its own B) or merely
waits is read from its own step CUDA time vs the peer's (padded → equal; waiting → its CUDA time stays at the B-token cost
and the gap is idle). The `rep` rank in `rep|rndT` is predicted unchanged vs rep|rep (it already sets the wave).
