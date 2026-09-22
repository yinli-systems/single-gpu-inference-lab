# Autonomous Research State

## Timestamp
2026-09-23 ~00:55 (Mac, UTC+4) = 04:55 cluster clock. Written by an interactive session (the 8-hour
autonomous sprint ended 2026-09-19 ~12:16 Mac; the six jobs it had queued started *after* that, so no
autonomous round ever saw their output — this session scored them).

## Git (branch, SHA, dirty files)
branch `dp-ep-waves`, 38 commits ahead of `origin/dp-ep-waves` (nothing pushed since the sprint).
This session's commit adds the 1603076 / 1603006 results, the ledger rows and this file.

## Running Slurm jobs (id, purpose, expected output path)
None. Queue holds only the user's unrelated training jobs (dense2b / ccl2 / moe7b).

## Completed this round
Scored every job the sprint left behind:
- **1603076** (`c26kv5-dp2pin`, DP2/EP2 placement, 4 arms × 5 repeats, node wqd10naf05g2): full data,
  scored against the pre-registered Q1–Q4 + per-arm CPU sidecar.
- **1603006** (`c27-dpep`, spec-decode acceptance skew round 2): full data, scored against the
  pre-registered gate.
- **1603014 / 1603015 / 1603052 / 1603071**: void — every 1-GPU arm died at engine init with
  `OOM on device 0 … free: 720896 of 25250627584` (the GPU handed to them was already occupied).
Artifacts pulled into `benchmarks/results/dp-ep-c26-pcie/raw/` and `dp-ep-c27-spec-skew/raw/`.

## Measured results (actual numbers only)
1603076, ratios against arm C (connector off, same node and cpuset), rank-0 chunk step = eager step ≥ 256 tokens:
| arm | chunk p50 | slow frac | sustained slow runs | peer store p95 | vs C |
| --- | --- | --- | --- | --- | --- |
| A-on-shared (on, unpinned) | 82.7 ms | 0.83 | 11 of 15 (35 s) | 164.8 ms | ×1.72 |
| M-on-mainiso | 54.7 ms | 0.25 | 0 of 10 | 116.3 ms | ×1.21 |
| B-on-isolated | 58.4 ms | 0.39 | 4 of 11 (10 s) | 115.0 ms | ×1.20 |
| C-off-shared | 47.0 ms | 0.00 | 0 of 10 | 96.0 ms | ×1.00 |
Sidecar, arm A, EngineCore main threads, slow vs fast samples: sibling busy 16/17 % vs 36/18 %,
run-queue wait ≈ 0.02 % both, migrations < 1.1/s, util 96–97 % vs 83–84 %, node PSI 0; on-GPU-node
21 % vs 0 % (rank 0) and 82 % vs 17 % (rank 1). Arm C: one main thread 87 % far-node, no slow state.
1603006: `rndT|rndT` period 25.8 ms (B=8) / 35.1 ms (B=32) vs `rep|rep` 26.0 / 34.1; skew effect on the
low-acceptance rank ITL p50 ×1.02, period ×1.01, generation rate ×1.24–1.34 (B=8) with padded width
32 → 40; the high-acceptance rank −12–13 %; everything ≤ ×1.03 at B=32.

## Alive hypotheses (evidence + gate)
Row 26 residual: with per-rank core reservation the connector still costs the peer ×1.16–1.21 and the
slow state is still *entered* (0.25–0.39 of chunk steps). Target for the next round: explain that
residual inside the connector/memory family (H1), gate unchanged (peer p95 +10 % strong, <5 % kill).
`--numa-bind` as a remedy is untested (its job died before serving).

## Killed hypotheses (reason + evidence)
- **SMT-sibling / run-queue sharing of the launch thread** (round-61 candidate): no slow-vs-fast
  contrast in sibling busy, run-queue wait, migrations or PSI within arm A; core reservation changes
  how often the state is entered without changing any per-sample contention reading.
- **NUMA placement of the EngineCore main thread**: points the wrong way (fast samples are *less*
  often on the GPU node; the clean arm runs 87 % far-node; the fully GPU-node-pinned arm still flips).
- **Row 27, spec-decode acceptance skew across DP ranks**: ITL effect ×1.02 vs a 5 % gate; the
  premise (a faster low-acceptance period to be dragged up) is false. Throughput redistribution at
  B=8 is symmetric, so no group-aware policy wins.

## Blocked hypotheses (specific blocker)
`--numa-bind` arm: needs a rerun; vLLM refuses auto-detection in a constrained cpuset and wants
`--numa-bind-nodes` explicitly. Any rerun needs a free-GPU-memory assertion before `vllm serve`
(four jobs were wasted on an occupied GPU).

## Current strongest result (bounded wording)
On 2×RTX 4090 DP2/EP2 (vLLM 0.29.0, Qwen1.5-MoE-A2.7B) inside a 6-core Slurm cpuset, turning the
native CPU-offload connector on puts the storing rank's worker into a sustained CPU-side slow state:
its eager 512-token chunk steps run ×1.75 the connector-off cost (82.7 vs 47.0 ms, 5 repeats) and the
peer rank's store-cell ITL p95 is ×1.72, with 11 of 15 sustained runs slow. Reserving cores per rank
(either the launch thread alone or the whole EngineCore) removes the sustained state and cuts the
peer cost to ×1.20–1.21, but does not eliminate it, and no per-sample CPU contention reading (SMT
sibling, run-queue wait, migrations, NUMA node, PSI) separates slow from fast samples — so core
reservation is a ~75 % mitigation, not the mechanism.

## Next exact action (the exact first command/file/experiment for the next session)
No GPU work should be submitted until the site has free GPUs (`sinfo`/`squeue` show the partitions
full). When one is available, the single highest-value experiment is the residual: rerun
`sbatch_c26kv5.sh` with arms {A-on-shared, M-on-mainiso} × {`--kv-offloading-size 4`, `16`} plus a
free-GPU-memory assertion at the top of the script (`nvidia-smi --query-gpu=memory.used` must be
< 500 MiB per assigned GPU before `vllm serve`), to test whether the residual scales with tier size
(connector-internal) or not (memory path). CPU-side meanwhile: nothing in this line is blocked on it.

## Files/artifacts created (paths)
benchmarks/results/dp-ep-c26-pcie/README.md ("Result: DP2 placement follow-up"), its
raw/{c26kv5-1603076.txt,c26kv5score-1603076.json,sidecar-1603076-*.json};
benchmarks/results/dp-ep-c27-spec-skew/README.md ("Round 2 result"), raw/c27-1603006.json;
docs/multigpu-opportunity-ledger.md rows 26 (round 63) and 27 (killed).

## Risks / unresolved methodological issues
- 1603076 is one node and one allocation; the arms are sequential within it, so a drift in node state
  across the ~26 min run is not excluded (arm order A, M, B, C; the two pinned arms sit in the middle).
- "Slow" is an absolute 65 ms threshold on rank-0 chunk steps, chosen from the round-59 data; M and B
  sit just above it (54–58 ms p50) so their slow fractions are threshold-sensitive, while the sustained-run
  count (0 and 4 of 11 vs A's 11 of 15) is not.
- The node's GPUs were occupied by another tenant during the 16:07–16:27 window; whether that also
  perturbed the 19:14 run is not knowable from the data kept.
