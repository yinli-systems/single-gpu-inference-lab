# AUTONOMY_STATE — updated by each headless session (read this first)

## 2026-09-19 (initial state written by the interactive session that set up the environment)

- Repo: branch `dp-ep-waves`, SHA see `git log -1`. Local commits only; never push.
- Environment on ParaCloud validated end to end on 2×4090: vLLM 0.29.0 DP2/EP2, allgather_reducescatter,
  multi-port placement, tracers v2 (rank-suffixed) + v5 (EP). Job template `scripts/dp_ep/sbatch_m1.sh`.
- Slurm jobs of the previous line (R1 phase routing, kept only as comparison data):
  1602456 (`Qwen1.5-MoE eager multiport q512`) and 1602457 (`eager internal q512`) — check
  `squeue`/`sacct -j`; results under `$W/results/m1-<jobid>-*`. Analyze with
  `scripts/dp_ep/analyze_dp_ep_waves.py --dir <dir> --output <json>` (it expects rank-suffixed
  step traces `step.jsonl.dp0/dp1`; older result dirs from jobs ≤1602445 have unsuffixed traces).
- Known gotchas: home quota 1 GB (all caches are redirected in the sbatch script); compute nodes offline;
  q < 64 needs `--max-num-seqs` ≤ q (handled by the script); startup ≈ 4 min; venv untar ≈ 1 min per node.

## Lines
| line | status | next |
| --- | --- | --- |
| G1 padding amplification oracle | **not started** | write `scripts/dp_ep/campaign_g1.sh`: for q in 32 64 128 256 512: sbatch graph+multiport and eager+multiport with harness `--cells dd,pd --batch-sizes 8,16,32 --prefix-lens 4096,8192`; then analyze padded vs true tokens per rank, step CUDA ms, ITL; compute A_pad, T_A/T_B, oracle |
| G2 cost boundary | blocked on G1 | — |
| P1 low-precision EP transport bound | not started | from the eager EP traces of 1602456: bytes per wave = Σ sizes × hidden × 2 B; share of step; halve → bound |
| C1 DP control-plane sync cost | not started | instrument `coordinate_batch_across_dp` all-reduce time (host clock) in eager runs |
| R1 phase-aware routing | comparison only (BalanceRoute exists) | analyze 1602456/1602457 when done |

## Unresolved risks
- Claude auth on the login node must be done by the user before the loop can run.
- Result dirs are large (EP traces ~40 MB/rank/run); keep under `$W/results`, copy only summaries into the repo.
