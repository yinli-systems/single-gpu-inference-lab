# Campaign G1 — cross-rank DP padding / graph-mode synchronization tax (2×RTX 4090, vLLM 0.29.0 DP2/EP2)

Status: **KILLED 2026-09-19 (rounds 1–3 complete): the pre-registered oracle gain is 0.0 % in all three regimes (arm B never wins a cell). Only measured numbers below.**

## Question

In vLLM 0.29.0 `vllm/v1/worker/dp_utils.py`, DP ranks synchronize `cudagraph_mode` by taking the
minimum across ranks; when the synced mode is not NONE every rank is padded to the maximum token
count across ranks (`_post_process_dp_padding`); the runner then re-dispatches with
`valid_modes={synced_mode}` (`gpu_model_runner.py` ~L4088–4110). Two regimes follow:

1. **Padding regime** — the heavy rank's batch fits a captured graph size: light ranks execute
   `max_r n_r` tokens (garbage rows through QKV/MoE/LM-head). `A_pad = R·max_r n_r / Σ_r n_r`.
2. **Graph-downgrade regime** — the heavy rank exceeds `max_cudagraph_capture_size`
   (= `min(2·max_num_seqs, 512)` by default, so **128** with `--max-num-seqs 64`): it reports
   NONE, the synced min is NONE, and every rank runs that step eager and unpadded.

Oracle: per cell, arm **A** (default: graph + max-DP padding), arm **B** (`--enforce-eager`:
synchronized eager, true per-rank sizes), **C** = min(A, B) per cell. Pre-registered gate on the
oracle gain over A: <5% kill, 5–10% engineering-only, >10% alive, >20% strong.

Theoretical expectation recorded before the graph arm finished (so it can be falsified): a
synchronized wave costs `max_r T_r`; padding a light rank to the heavy rank's shape converts idle
wait into wasted GPU work but does not lengthen the wave unless the light rank becomes the
straggler; arm B pays the eager launch floor (~45 ms/step on this model, measured in M1a) on every
step. Hence A ≥ B is expected only where the padded light rank's own work exceeds the heavy rank's.

## Measurement contract

- Server: `scripts/dp_ep/sbatch_g1.sh <model> <graph|eager> multiport <q> [extra]` — Qwen1.5-MoE-A2.7B-Chat,
  `--data-parallel-size 2 --data-parallel-size-local 2 --enable-expert-parallel
  --all2all-backend allgather_reducescatter --max-model-len 16384 --max-num-seqs 64
  --max-num-batched-tokens q --gpu-memory-utilization 0.88 --no-enable-prefix-caching
  --data-parallel-multi-port-external-lb --api-server-count 1` (rank r on port 8300+r).
- Client: `scripts/dp_ep/measure_dp_ep_waves.py` — cells dd / pp / pd, B ∈ {8,16,32} decode
  streams per rank (128-token prompts, 4096-token generations, ITL from token arrival times),
  prefill L ∈ {4096, 8192} tokens chunked at q; 3 shuffled repeats (45 windows per job).
- Server-side step trace per rank (`trace/step.jsonl.dp{0,1}`): `num_tokens` (true), `padded_tokens`
  (executed after graph-bucket + DP padding), `cg_mode`, `cuda_ms` (this rank's step GPU time incl.
  waiting inside collectives), `t` (host monotonic s). Same node clock for both ranks.
- Analyzer: `scripts/dp_ep/analyze_g1.py --arm-a <graph dir> --arm-b <eager dir> --output g1.json`.
- Metrics per cell: decode-rank ITL p50/p95 (user-visible), prefill TTFT, wave period, per-rank
  true/executed tokens, A_pad, graph mode. Oracle gain = 1 − min(A,B)/A on ITL p50 and TTFT.

## Runs (Slurm job → arm → results dir on the cluster)

| job | arm | q | capture cap | regime | state |
| --- | --- | ---: | ---: | --- | --- |
| 1602456 | B eager | 512 | — | true shapes | done (M1a run reused) |
| 1602471 | A graph | 512 | 128 (default) | graph-downgrade for chunk steps | done |
| 1602484 | A graph | 512 | 512 (`--max-cudagraph-capture-size 512`) | padding (light rank padded to 512) | done |
| 1602472 | A graph | 128 | 128 | padding (light rank padded to 128) | done |
| 1602473 | B eager | 128 | — | true shapes | done |

## Results

### Round 1 — q=512, default capture cap 128 (graph-downgrade regime): oracle gain 0.0 %

`raw/g1-oracle-graphcap128-1602471-vs-eager-1602456.json` (analyzer output, both arms, all 45
windows each; git d3a8911; commands in `raw/server-1602471.cmd` and the M1 README).
Repeats pooled by median; decode rank = rank 1 (ports 8301); ratio = A/B.

| cell | B | L | ITL r1 p50 A / B (ratio) | ITL r1 p95 A / B | TTFT A / B (ratio) | wave period A / B (ratio) |
| --- | ---: | ---: | --- | --- | --- | --- |
| dd | 8 | 0 | 13.7 / 45.6 (0.30) | 27.4 / 92.5 | — | 13.4 / 45.4 (0.30) |
| dd | 16 | 0 | 16.7 / 46.0 (0.36) | 34.0 / 93.2 | — | 16.4 / 45.6 (0.36) |
| dd | 32 | 0 | 21.6 / 47.1 (0.46) | 43.7 / 94.8 | — | 21.3 / 46.4 (0.46) |
| pd | 8 | 4096 | 17.7 / 47.2 (0.38) | 54.0 / 94.6 | 452 / 518 (0.87) | 17.3 / 46.7 (0.37) |
| pd | 8 | 8192 | 38.9 / 48.0 (0.81) | 96.8 / 95.9 | 861 / 865 (1.00) | 34.0 / 47.1 (0.72) |
| pd | 16 | 4096 | 22.0 / 47.7 (0.46) | 56.8 / 96.0 | 470 / 516 (0.91) | 20.3 / 46.9 (0.43) |
| pd | 16 | 8192 | 46.6 / 48.3 (0.97) | 97.4 / 97.6 | 868 / 900 (0.96) | 35.6 / 47.9 (0.74) |
| pd | 32 | 4096 | 27.5 / 49.9 (0.55) | 76.7 / 96.7 | 476 / 526 (0.90) | 24.4 / 48.5 (0.50) |
| pd | 32 | 8192 | 47.2 / 48.0 (0.98) | 97.4 / 96.6 | 911 / 948 (0.96) | 27.8 / 47.4 (0.59) |
| pp | 8 | 4096 | 18.1 / 45.6 (0.40) | 53.0 / 93.7 | 417 / 497 (0.84) | 17.1 / 45.4 (0.38) |
| pp | 8 | 8192 | 40.4 / 46.0 (0.88) | 90.5 / 91.6 | 798 / 846 (0.94) | 20.6 / 45.0 (0.46) |
| pp | 16 | 4096 | 21.6 / 46.7 (0.46) | 47.8 / 92.6 | 445 / 524 (0.85) | 20.5 / 46.3 (0.44) |
| pp | 16 | 8192 | 41.0 / 45.8 (0.90) | 85.9 / 92.7 | 819 / 865 (0.95) | 24.0 / 45.5 (0.53) |
| pp | 32 | 4096 | 28.4 / 47.6 (0.60) | 78.1 / 94.5 | 446 / 507 (0.88) | 24.4 / 47.0 (0.52) |
| pp | 32 | 8192 | 44.0 / 48.6 (0.91) | 93.9 / 97.3 | 900 / 952 (0.94) | 34.8 / 47.2 (0.74) |

Arm B (eager) never wins a cell: oracle gain over the default = **0.0 % on decode-rank ITL p50
(0/15 cells) and 0.0 % on TTFT (0/12)** → below the 5 % kill line for this regime.

What the step traces show (arm A, rep 0): decode-only waves run FULL graphs at the true size
(exec = true = 8/16/32 tokens; 13.2 / 16.5 / 21.3 ms). Every 512-token chunk step is `NONE` on
**both** ranks (the heavy rank exceeds the 128-token cap; rank 1 with 8 true tokens executes 8
tokens, eager, and waits): chunk-step GPU time 48–52 ms on both ranks, identical per rank because
the light rank's `cuda_ms` includes the wait inside the collectives — the wave is the heavy
rank's eager chunk, and it costs the same 46–50 ms in arm B. Real DP padding occurs only on the
decode steps of pd cells (rank 0 has 9 = 8 decoders + 1 prefill token → bucket 16; rank 1 is
padded 8 → 16; exec_amp up to 1.5) and is free at these sizes (weight-bound decode).

Eager floor: in arm B every step costs 44–48 ms regardless of shape ((8,8) to (512,512)), so the
"true-shape" arm can only win where padding costs more than ~30 ms per step on the light rank.

The peer-rank ITL cost of a chunked prefill is a *synchronization* cost, not a padding cost:
rank 1's ITL rises from 13.7 ms (dd) to 38.9 ms (pd, L=8192) in arm A, because every chunk step
of rank 0 is a ~50 ms wave. Whether the 512-cap (padded, piecewise-graphed chunk steps) shortens
that wave is the round-2 question (job 1602484).

### Round 2 — q=128, capture cap 128 (pure padding regime): oracle gain 0.0 %

`raw/g1-analysis/g1-oracle-q128-graph-1602472-vs-eager-1602473.json` (+ `g1-oracle-q128.md`, the
full analyzer tables; git d3a8911; server commands in each results dir's `server.cmd`).
Here every chunk step fits the cap, so DP padding really happens: in pd cells the decode rank
executes **128 tokens per step for 8/16/32 true tokens** (`exec r1 = 128`, A_pad 1.88/1.78/1.60,
exec_amp 1.83/1.66/1.49, graph mode PIECEWISE on both ranks).

| cell | B | L | A_pad | exec_amp | r1 exec/true | ITL r1 p50 A / B (ratio) | TTFT A / B (ratio) | wave period A / B |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| dd | 8 | 0 | 1.00 | 1.00 | 8/8 | 13.7 / 46.9 (0.29) | — | 13.6 / 46.3 |
| dd | 16 | 0 | 1.00 | 1.00 | 16/16 | 17.0 / 58.7 (0.29) | — | 16.9 / 48.8 |
| dd | 32 | 0 | 1.00 | 1.00 | 32/32 | 21.1 / 50.3 (0.42) | — | 21.0 / 49.5 |
| pd | 8 | 4096 | 1.88 | 1.83 | 128/8 | 23.5 / 49.3 (0.48) | 861 / 1874 (0.46) | 23.3 / 48.9 |
| pd | 8 | 8192 | 1.88 | 1.85 | 128/8 | 24.9 / 50.3 (0.49) | 1777 / 4229 (0.42) | 24.5 / 49.1 |
| pd | 16 | 4096 | 1.78 | 1.66 | 128/16 | 24.4 / 50.6 (0.48) | 938 / 2389 (0.39) | 24.0 / 49.5 |
| pd | 16 | 8192 | 1.78 | 1.69 | 128/16 | 25.7 / 48.6 (0.53) | 1966 / 3797 (0.52) | 25.3 / 48.0 |
| pd | 32 | 4096 | 1.60 | 1.49 | 128/32 | 25.5 / 51.1 (0.50) | 1154 / 2275 (0.51) | 25.1 / 50.4 |
| pd | 32 | 8192 | 1.60 | 1.53 | 128/32 | 27.8 / 52.1 (0.53) | 2398 / 4461 (0.54) | 27.2 / 50.2 |
| pp | 8 | 4096 | 1.00 | 1.22 | 128/128 | 24.3 / 47.4 (0.51) | 883 / 1799 (0.49) | 23.7 / 47.0 |
| pp | 8 | 8192 | 1.00 | 1.13 | 128/128 | 27.0 / 48.1 (0.56) | 1826 / 3488 (0.52) | 25.3 / 47.3 |
| pp | 16 | 4096 | 1.00 | 1.11 | 128/128 | 24.6 / 47.8 (0.51) | 967 / 1948 (0.50) | 24.4 / 47.7 |
| pp | 16 | 8192 | 1.00 | 1.07 | 128/128 | 26.4 / 47.9 (0.55) | 2013 / 3680 (0.55) | 25.8 / 47.4 |
| pp | 32 | 4096 | 1.00 | 1.05 | 128/128 | 25.9 / 50.4 (0.52) | 1165 / 2285 (0.51) | 25.5 / 48.5 |
| pp | 32 | 8192 | 1.00 | 1.03 | 128/128 | 28.2 / 54.6 (0.52) | 2428 / 4546 (0.53) | 27.4 / 49.2 |

Oracle gain 0.0 % on ITL (0/15 cells) and 0.0 % on TTFT (0/12). The padded decode rank's step
time equals the heavy rank's (cuda r0 = cuda r1 = 23–27 ms for 128 executed tokens on both), and
the pd wave period (23.3–27.2 ms) equals the pp period (23.7–27.4 ms) where rank 1 does 128 *real*
tokens — i.e. the wave is the heavy rank's chunk in both cases and the padded rank's wasted
128-token forward never extends it. The decoder's user-visible cost of the peer prefill
(13.7 → 23.5 ms ITL at B=8) is the same synchronization cost as in round 1, now paid as executed
padding instead of idle wait.

### Round 3 — q=512, capture cap raised to 512 (padding at 512): oracle gain 0.0 %

`raw/g1-analysis/g1-oracle-q512cap512-graph-1602484-vs-eager-1602456.json` (+ `.md`). With
`--max-cudagraph-capture-size 512` the 512-token chunk steps run PIECEWISE graphs on both ranks and
the decode rank is padded 8/16/32 → 512 on those steps (`exec max 512/512`, A_pad max 1.97,
exec_amp 1.81/1.56/1.42 in pd). Chunk-step GPU time (cuda p95, both ranks, all pd/pp windows):
**40.5–48.4 ms padded-graph vs 47.9–58.8 ms eager** in arm B — the graph saves ~15 % per chunk step.

| cell | B | L | A_pad | exec_amp | ITL r1 p50 A / B (ratio) | TTFT A / B (ratio) | period A / B |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| dd | 8 | 0 | 1.00 | 1.00 | 13.6 / 45.6 (0.30) | — | 13.4 / 45.4 |
| dd | 16 | 0 | 1.00 | 1.00 | 17.5 / 46.0 (0.38) | — | 16.4 / 45.6 |
| dd | 32 | 0 | 1.00 | 1.00 | 21.7 / 47.1 (0.46) | — | 21.4 / 46.4 |
| pd | 8 | 4096 | 1.06 | 1.81 | 17.7 / 47.2 (0.38) | 380 / 518 (0.73) | 17.5 / 46.7 |
| pd | 8 | 8192 | 1.06 | 1.85 | 37.4 / 48.0 (0.78) | 724 / 865 (0.84) | 24.8 / 47.1 |
| pd | 16 | 4096 | 1.03 | 1.56 | 22.5 / 47.7 (0.47) | 386 / 516 (0.75) | 20.7 / 46.9 |
| pd | 16 | 8192 | 1.03 | 1.65 | 38.5 / 48.3 (0.80) | 757 / 900 (0.84) | 24.7 / 47.9 |
| pd | 32 | 4096 | 1.02 | 1.42 | 27.7 / 49.9 (0.56) | 415 / 526 (0.79) | 24.5 / 48.5 |
| pd | 32 | 8192 | 1.17 | 1.52 | 40.8 / 48.0 (0.85) | 830 / 948 (0.88) | 29.0 / 47.4 |
| pp | 8 | 4096 | 1.00 | 1.43 | 17.6 / 45.6 (0.39) | 391 / 497 (0.79) | 17.0 / 45.4 |
| pp | 8 | 8192 | 1.00 | 1.32 | 40.2 / 46.0 (0.87) | 758 / 846 (0.90) | 21.8 / 45.0 |
| pp | 16 | 4096 | 1.00 | 1.23 | 21.8 / 46.7 (0.47) | 399 / 524 (0.76) | 20.8 / 46.3 |
| pp | 16 | 8192 | 1.00 | 1.17 | 39.5 / 45.8 (0.86) | 788 / 865 (0.91) | 24.5 / 45.5 |
| pp | 32 | 4096 | 1.00 | 1.12 | 28.2 / 47.6 (0.59) | 421 / 507 (0.83) | 24.7 / 47.0 |
| pp | 32 | 8192 | 1.00 | 1.09 | 41.9 / 48.6 (0.86) | 849 / 952 (0.89) | 29.3 / 47.2 |

Oracle gain 0.0 % (0/15, 0/12). Compared with round 1 (same q, default cap 128 → chunk steps
eager on every rank), the 512 cap lowers TTFT in every pd/pp cell (e.g. pd 8/4096 452 → 380 ms,
pd 8/8192 861 → 724 ms, pp 32/8192 900 → 849 ms; 6–16 %) with decode-rank ITL unchanged within
noise (37.4 vs 38.9 ms at pd 8/8192; 17.7 vs 17.7 at pd 8/4096) — the padded decode rank does
not become the straggler even at 512 executed tokens for 8 true ones.

## Verdict (2026-09-19): KILLED under the pre-registered gate, on mechanism grounds

1. Arm B (`--enforce-eager`) never wins a cell in any regime: its launch floor is 44–50 ms per
   step for every shape from (8,8) to (512,512) on this platform (2.7B-active MoE, 24 layers, 60
   experts, CPU-launch-bound), against 13–29 ms graph waves. A graph-vs-eager admission controller
   has nothing to choose.
2. The padding itself is user-invisible in DP2: a padded rank executes at most the heavy rank's
   token count with strictly less attention work, so its step is never longer than the heavy rank's
   and the synchronized wave is unchanged (round 2: pd period = pp period; round 3: padding to 512
   leaves ITL unchanged while the graphs cut TTFT). Padding converts idle wait into wasted GPU
   energy (exec_amp up to 1.88), not into latency or lost throughput — in lockstep DP the light rank
   could not have used that time anyway.
3. What *is* user-visible in these data is the synchronization cost of a peer rank's chunked
   prefill on unrelated decoders: rank 1's ITL p50 13.7 → 23.5 ms (q=128) and → 37–41 ms (q=512,
   L=8192) purely because every wave is the heavy rank's chunk. This is the barrier-synchronized
   DP straggler cost (BalanceRoute territory) and is the input to task 24 (group-aware chunk
   budget), not a padding effect. Engineering side-note (single-GPU knob, amplified by DP): the
   default `max_cudagraph_capture_size = min(2·max_num_seqs, 512)` = 128 forces every rank eager
   on 512-token chunk steps; `--max-cudagraph-capture-size 512` recovers 6–16 % TTFT at no ITL cost.

