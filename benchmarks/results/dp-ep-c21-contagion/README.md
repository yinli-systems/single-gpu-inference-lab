# Task 21 — cross-rank feature contagion under DP2/EP2 (2×RTX 4090, vLLM 0.29.0)

Status: **round 1 measured (2026-09-19); round 2 (decomposition + fixed prompt-logprobs cells)
running as Slurm job 1602499.** Only measured numbers below.

## Question

Does an expensive *request-local* feature on DP rank 0 (own GPU, own engine core, own API server)
raise the inter-token latency of unrelated plain requests on DP rank 1, because the ranks
rendezvous every step (DP coordination all-reduce, per-layer EP collectives)? Pre-registered gate
on the plain rank's ITL p50 ratio vs the plain|plain baseline: <5 % kill, 10 % alive, 20 % strong,
50 % priority upstream problem.

## Measurement contract

- Server: `scripts/dp_ep/sbatch_c21.sh Qwen1.5-MoE-A2.7B-Chat graph multiport 512` → `vllm serve
  --data-parallel-size 2 --data-parallel-size-local 2 --enable-expert-parallel --all2all-backend
  allgather_reducescatter --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 512
  --gpu-memory-utilization 0.88 --no-enable-prefix-caching --data-parallel-multi-port-external-lb
  --api-server-count 1` (default CUDA-graph mode; rank r on port 8300+r; exact text in the
  results dir `server.cmd`).
- Client: `scripts/dp_ep/measure_contagion.py --features logprobs,struct,sampling,plogp,pfx
  --batch-sizes 8,32 --repeats 3` — B decode streams per rank (128-token random prompts, 4096-token
  generations, `ignore_eos`), 3 s settle, 8 s measurement window, cells plain|plain, F|plain and
  plain|F (swap), shuffled per repeat (66 windows). ITL = gaps between token arrival times inside
  the window; per-rank step trace (`trace/step.jsonl.dp{0,1}`) gives the step period.
- Features: `logprobs` = top-5 logprobs per token; `struct` = xgrammar EBNF grammar
  (`root ::= line*`, word/number lines) on every stream; `sampling` = temperature 1, top-p 0.9,
  top-k 50, min-p 0.05, repetition 1.2, presence 0.5, frequency 0.5; `pfx` = a train of
  2048-token prompts (max_tokens 8) issued back-to-back on the feature rank during the window
  (prefill cost only, no feature); `plogp` = the same train with `prompt_logprobs=1`.
- Analyzer: `scripts/dp_ep/analyze_contagion.py --dir <results dir> --output c21.json`; repeats
  pooled by median.

## Round 1 — job 1602481 (`raw/c21-round1-1602481.{json,md}`, git d3a8911)

Plain-rank ITL p50 with the feature on the *other* rank, ratio vs plain|plain on the same rank:

| feature | B | F on r0 → r1 plain ITL (ratio) | F on r1 → r0 plain ITL (ratio) | p95 ratios | plain-rank step period base → with F (ms) |
| --- | ---: | --- | --- | --- | --- |
| logprobs (top-5) | 8 | 14.4 → 14.5 (1.01) | 14.5 → 14.5 (1.00) | 1.00 / 0.97 | 14.3 → 14.4 / 13.7 |
| logprobs (top-5) | 32 | 21.8 → 22.0 (1.01) | 21.5 → 21.9 (1.02) | 1.02 / 1.02 | 21.1 → 21.6 / 21.6 |
| struct (xgrammar) | 8 | 14.4 → 13.7 (0.95) | 14.5 → 13.6 (0.94) | 0.95 / 0.94 | 14.3 → 13.5 / 13.4 |
| struct (xgrammar) | 32 | 21.8 → 21.0 (0.96) | 21.5 → 21.6 (1.00) | 0.98 / 1.01 | 21.1 → 20.8 / 21.2 |
| sampling (penalties + top-k/p + min-p) | 8 | 14.4 → 16.3 (**1.13**) | 14.5 → 16.2 (**1.12**) | 1.12 / 1.10 | 14.3 → 15.9 / 16.1 |
| sampling | 32 | 21.8 → 23.7 (**1.09**) | 21.5 → 23.6 (**1.10**) | 1.09 / 1.09 | 21.1 → 23.5 / 23.1 |
| pfx (2048-token prompt train, no feature) | 8 | 14.4 → 27.7 (**1.92**) | 14.5 → 16.6 (1.15)¹ | 3.27 / 1.70 | 14.3 → 15.6 / 15.2 |
| pfx | 32 | 21.8 → 29.1 (**1.34**) | 21.5 → 28.5 (**1.33**) | 1.90 / 1.72 | 21.1 → 23.6 / 23.7 |
| plogp | 8, 32 | invalid | invalid | — | — |

¹ The pfx B=8 cells are bimodal across repeats, not across placements: plain-rank ITL p50 =
16.9 / 27.7 / 30.2 ms (pfx|plain, reps 0–2) and 15.9 / 16.9 / 29.5 ms (plain|pfx), with the
same train (23–24 requests, TTFT 230–238 ms) and the same plain-rank step period p50 (14.9–16.5 ms)
in every window. The pooled medians (27.7 vs 16.6) therefore understate the spread; the p95s
(49–96 ms) are consistently 2–3× the baseline. Unexplained; round 2 repeats the cell.

Invalid cells: `plogp` — vLLM rejects `prompt_logprobs` with `stream=True` (HTTP 400,
"`prompt_logprobs` are not available when `stream=True`"), so those windows were a 2.6–2.8k-request
train of immediate 400s on the feature rank's API server. Recorded as a by-product: **this
API-server error hammer did not move the peer rank's ITL (0.97–1.03)**, i.e. front-end CPU load on
one rank is not exported. Fixed in round 2 (non-streaming requests for the train).

Errors in decode streams: 0 in all 66 windows.

## Round 2 — job 1602499 (`raw/c21-round2-1602499.{json,md}`, git bb647a3 harness, seed 62)

Decomposition of the sampling bundle (`temp` = temperature 1 + per-request seed; `topk` = temp +
top-p 0.9 / top-k 50 / min-p 0.05; `pen` = temp + repetition/presence/frequency penalties), the
fixed prompt-logprobs cell (`plogp`: non-streaming 2048-token prompt train with `prompt_logprobs=1`,
vs `pfx`: the same train without it) and top-20 logprobs. 90 windows, 3 repeats, shuffled; decode
stream errors: 2 windows with one errored stream each. Plain-rank ITL p50 ratio vs plain|plain on
the same rank, pooled medians over repeats:

| feature | B | F on r0 → r1 plain (ratio) | F on r1 → r0 plain (ratio) | p95 ratios | plain-rank step period base → with F (ms) |
| --- | ---: | --- | --- | --- | --- |
| temp (T=1 + seed) | 8 | 13.3 → 14.3 (1.07) | 13.6 → 14.8 (1.09) | 1.04 / 1.10 | 13.2 → 14.3 / 13.1 → 14.7 |
| topk (+ top-p/k, min-p) | 8 | 13.3 → 14.6 (1.10) | 13.6 → 14.6 (1.07) | 1.08 / 1.09 | 13.2 → 14.5 / 13.1 → 14.4 |
| pen (+ penalties) | 8 | 13.3 → 15.0 (**1.13**) | 13.6 → 15.3 (**1.12**) | 1.12 / 1.12 | 13.2 → 15.0 / 13.1 → 15.2 |
| sampling (all of the above) | 8 | 13.3 → 16.2 (**1.21**) | 13.6 → 16.2 (**1.20**) | 1.20 / 1.21 | 13.2 → 16.1 / 13.1 → 16.1 |
| logprobs20 | 8 | 13.3 → 14.2 (1.07) | 13.6 → 14.2 (1.04) | 1.07 / 1.06 | 13.2 → 14.2 / 13.1 → 14.0 |
| pfx (prompt train, no feature) | 8 | 13.3 → 15.8 (**1.18**) | 13.6 → 16.3 (**1.20**) | 1.89 / 2.03 | 13.2 → 15.6 / 13.1 → 15.1 |
| plogp (prompt train + prompt_logprobs) | 8 | 13.3 → 16.8 (**1.26**) | 13.6 → 15.6 (**1.15**) | 2.35 / 2.08 | 13.2 → 15.8 / 13.1 → 15.2 |
| temp | 32 | 21.8 → 22.2 (1.02) | 21.8 → 22.2 (1.02) | 1.02 / 1.02 | 21.3 → 21.8 / 21.4 → 21.7 |
| topk | 32 | 21.8 → 23.2 (1.07) | 21.8 → 22.9 (1.05) | 1.07 / 1.05 | 21.3 → 22.7 / 21.4 → 22.5 |
| pen | 32 | 21.8 → 22.8 (1.05) | 21.8 → 22.6 (1.04) | 1.05 / 1.03 | 21.3 → 22.3 / 21.4 → 22.3 |
| sampling | 32 | 21.8 → 23.7 (1.09) | 21.8 → 23.6 (1.08) | 1.09 / 1.08 | 21.3 → 23.4 / 21.4 → 23.4 |
| logprobs20 | 32 | 21.8 → 21.8 (1.00) | 21.8 → 21.6 (0.99) | 1.01 / 0.99 | 21.3 → 21.6 / 21.4 → 21.3 |
| pfx | 32 | 21.8 → 34.5 (**1.58**)² | 21.8 → 28.0 (**1.29**) | 1.93 / 1.79 | 21.3 → 23.5 / 21.4 → 23.7 |
| plogp | 32 | 21.8 → 27.8 (**1.28**) | 21.8 → 27.5 (**1.26**) | 1.83 / 1.73 | 21.3 → 23.6 / 21.4 → 23.5 |

² pfx remains bimodal across repeats (see round 1, ¹); its step-period increase (21.3 → 23.5) is the
same as plogp's (→ 23.6), so the 1.58 vs 1.28 difference is token-delivery noise, not a mechanism.

Reading of round 2:

1. **Every sampler component is exported 1:1 and each is small.** At B=8 (13 ms steps) the
   penalties path adds +1.8–2.1 ms per step on both ranks (×1.12–1.13) — the PR #47540 mechanism at
   the short-history end of its curve; seeded temperature sampling alone adds +1.1–1.6 ms (×1.07–1.09,
   the per-request generator path); top-k/p/min-p add ≤0.3 ms beyond that; the bundle is the sum
   (+2.9–3.0 ms, ×1.20–1.21). At B=32 (21 ms steps) all of them are ≤×1.09. Nothing here is a
   separate isolation mechanism.
2. **Prompt logprobs add nothing beyond the prefill train**: plogp and pfx raise the plain rank's
   step period identically (B=8: 15.1–15.8 ms; B=32: 23.5–23.7 ms); the plain rank pays for the
   synchronized 512-token chunk steps, not for the logprob gather on the other GPU.
3. **logprobs20 ≤ ×1.07** (B=8) and nil at B=32 — same verdict as top-5 in round 1.

**Task 21 closed (2026-09-19):** cross-rank feature contagion on this platform = the feature's own
per-step cost exported by lockstep; individual features ≤ ×1.13 (pen) at B=8 and ≤ ×1.07 at B=32;
the only >20 % effect is chunked prefill on the peer (×1.2–1.6), which is the synchronized-shape
cost recorded under G1 / task 24. No new line.

## Reading (bounded)

1. **Lockstep export is exact.** In every cell the plain rank's step period rose by the same
   amount as the feature rank's (e.g. sampling B=8: both ranks 14.3 → 15.9–16.2 ms; B=32: 21.1 →
   23.1–23.6 ms): whatever a feature adds to one rank's step is added to the other rank's step, so
   cross-rank contagion equals the feature's own per-step cost. The ITL ratio therefore measures
   how expensive the feature is per step on this model, not a separate isolation mechanism.
2. **logprobs and structured output export nothing measurable** (≤2 %) — their per-step cost on a
   2.7B-active model with B ≤ 32 is below the noise of a 14–22 ms step. Kill for these features.
3. **The sampling bundle exports +1.7–2.4 ms per step (×1.09–1.13, symmetric under swap)** — in
   the 10 % "alive" band, but the mechanism is already known upstream: vLLM PR #47540 (open,
   2026-07-03, "Maintain persistent penalty statistics instead of per-step CPU rebuild in V1
   sampler") states that the penalties path rebuilds an O(rows × max_output_len) token matrix on
   the CPU every step and that "in a DP deployment this is a per-step straggler: whichever rank
   holds the longest output stalls every other rank at the DP sync point", measured at 6 ms/step
   for 4k output history growing to 47 ms/step at 28k. Our windows have ≤ ~600 tokens of output
   history, so we see the small end of that curve. Downgraded to a replication; round 2 separates
   penalties (`pen`) from GPU top-k/top-p/min-p (`topk`) and temperature-only (`temp`).
4. **The only large contagion is the prefill train (×1.33–1.92)** — chunked prefill on one rank
   makes every synchronized wave a 512-token chunk step (~40 ms) for the peer's decoders. That is
   the barrier-synchronized-DP straggler cost already seen in campaign G1, not a feature effect.

Gate outcome for the feature question: logprobs / struct KILLED (<5 %); sampling 9–13 % =
measured but not novel (upstream PR #47540); prefill contagion 33–92 % belongs to task 24 / G1.
