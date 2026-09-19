# Autonomous Research State

## Timestamp
2026-09-19 (initial, written by the interactive setup session)

## Git
branch dp-ep-waves; see `git log -1` for SHA; working tree clean at handoff.

## Running Slurm jobs
None at handoff. Finished: 1602456 (`m1 eager multiport q512`, 45 cells, results under
/data/run01/scxi253/inference/results/m1-1602456-*) and 1602457 (`eager internal q512`) which
**died** — triage its server.log before reusing the internal-LB arm.

## Completed this round
Environment built and validated end to end: vLLM 0.29.0 + torch 2.13cu130 venv (tarball
`$W/venv-vllm.tar`, node-local untar), models staged, tracers v2 (rank-suffixed iter/step) and v5
(EP collectives) installed, Slurm template `sbatch_m1.sh`, harness `measure_dp_ep_waves.py`,
analyzer `analyze_dp_ep_waves.py`, SSH helpers `pc.sh` / `pcput.sh` / `pcget.sh`.

## Measured results
Smoke (1602445, eager, multiport, Qwen1.5-MoE): DP2/EP2 healthy; GPU KV 29,520 tokens/rank;
EP trace 308,975 collective rows/rank, seq-aligned across ranks; host arrival skew (r0−r1)
p5/p50/p95 = −2.22 / +0.29 / +4.50 ms; dispatch CUDA p50/p95 = 0.134/0.800 ms, combine 0.034/0.508;
wave sizes observed (8,8), (16,16), (32,32), (512,512), (9,8). Prefill TTFT 527–1007 ms at 4k–8k.

## Alive hypotheses
G1 cross-rank DP padding amplification (flagship, not started — start here).
P1 generic low-precision PCIe EP transport (upper bound only). C1 DP control-plane tax.
Frontier pool 21–30 per the prompt's priority order.

## Killed hypotheses
None yet. R1 phase-aware routing downgraded to comparison only (BalanceRoute, arXiv:2605.06113).

## Blocked hypotheses
None.

## Current strongest result
None yet (environment only).

## Next exact action
Write `scripts/dp_ep/campaign_g1.sh` and submit the first G1 pair: for q in 32,64,128,256,512 submit
`sbatch_m1.sh Qwen1.5-MoE-A2.7B-Chat graph multiport <q>` and `... eager multiport <q>` with harness
args `--cells dd,pd --batch-sizes 8,16,32 --prefix-lens 4096,8192`; while they queue, extend
`analyze_dp_ep_waves.py` to report per rank true vs padded tokens, A_pad, graph mode and step CUDA
time, and triage why 1602457 died.

## Files/artifacts created
scripts/dp_ep/{sbatch_m1.sh,measure_dp_ep_waves.py,analyze_dp_ep_waves.py,apply_tracer_v5_ep.py,
pc.sh,pcput.sh,pcget.sh,run_claude_watchdog.sh}, AUTONOMY_PROMPT.md, AUTONOMY_STATE.md,
docs/multigpu-opportunity-ledger.md.

## Risks / unresolved methodological issues
- EP tracer hooks do not execute inside captured CUDA graphs: in graph mode rely on the step trace
  (`num_tokens` vs `padded_tokens`, `cg_mode`, `cuda_ms`), not on ep.*.jsonl.
- Login-node SSH resets under load; retry once then continue with other work.
- Result dirs are large (~40 MB EP trace per rank per run); keep raw data on the cluster.
- Home quota is 1 GB and full; every cache is redirected in the sbatch script — keep it that way.
