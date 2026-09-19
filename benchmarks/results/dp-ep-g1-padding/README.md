# Campaign G1 — cross-rank DP padding / graph-mode synchronization tax (2×RTX 4090, vLLM 0.29.0 DP2/EP2)

Status: **round 1 (2026-09-19), arms running; this README is updated per round with only measured numbers.**

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
| 1602484 | A graph | 512 | 512 (`--max-cudagraph-capture-size 512`) | padding | running |
| 1602472 | A graph | 128 | 128 | mixed (128+B > 128 → downgrade) | running |
| 1602473 | B eager | 128 | — | true shapes | running |

## Results

(filled per round from `analyze_g1.py` output; raw JSON under `raw/`)
