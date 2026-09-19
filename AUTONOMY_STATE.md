# Autonomous Research State

## Timestamp
2026-09-19 06:05 (Mac, UTC+4) = 10:05 cluster clock (UTC+8). Written by autonomous round 57 (session started 05:43).

## Git (branch, SHA, dirty files)
branch `dp-ep-waves`; commits this round: `c535f14` (hog `--engine sm`, analyze_dmon, sbatch_c26r2/r3/hogtest), `fb2e0ce`
(dmon counters, real mover, SGLang #34805 prior art), `8c791dd` (NCCL transport confirmed, real-mover harness draft),
`14516cd` (pre-registered predictions), then the final state commit (see `git log -1`). Working tree clean after it.
Cluster copies in `/data/run01/scxi253/inference/lab-scripts/`: pcie_hog.py (04e97116…), measure_pcie.py (976cb9d0…),
analyze_dmon.py, sbatch_c26r2.sh, sbatch_c26r3.sh, sbatch_hogtest.sh, measure_kvoffload.py, sbatch_c26kv.sh (md5 = repo files).

## Running Slurm jobs (id, purpose, expected output path)
- **1602541** `c26r2-dpep` (started 09:44 cluster, cells from 10:02): task 26 round 2 = d2h duty/cap sweep + h2d:cap12 + 0.75 GiB
  bursts + NCCL transport log; 12 specs × B{8,32} × 3 repeats = 72 cells ≈ 20 min → `results/c26-1602541-Qwen1.5-MoE-A2.7B-Chat-graph-multiport-q512/{pcie.json,waves.log,hog/,dmon.log,nccl-*.log,trace/}`.
- **1602557** `c26r3-dpep` (started 09:55 cluster): task 26 round 3 = requester probe (copy engine vs Triton-SM D2H/H2D, full and
  cap 8), 7 specs × 2 × 3 = 42 cells → `results/c26-1602557-…/`.
- 1602548 `hogtest` COMPLETED (SM engine validation, `logs/hogtest-1602548.out`).
- Other jobs in `squeue -u $USER` (mbert149-glue, vft-*, moe2b-ds-*, fork1b, moe7b, f3b-prof) are the user's own unrelated work — never touch them.

## Completed this round
1. Round 2 + round 3 of task 26 designed, harness extended and submitted (see above). `pcie_hog.py --engine sm` = Triton kernel
   copying through the pinned host pointer (validated on an idle 4090, job 1602548: ce and sm both 26.2–26.3 GB/s each way,
   d2d ce 454 / sm 221 GB/s, 8 GB/s cap exact). `measure_pcie.py` accepts `d2h-sm:…` specs. Note: the new hog file replaced the
   cluster copy while job 1602541 was running; `--engine` defaults to `ce`, identical copy path (only an `engine` field added to the log).
2. **PCIe counters of job 1602508 analyzed** (`scripts/dp_ep/analyze_dmon.py`, `raw/dmon-1602508.json`): plain baseline already
   shows ~6 GB/s PCIe rx per GPU (NCCL host-memory transport polling; doubles to 12.3 GB/s on the waiting GPU in the d2d cell),
   EP payload ≈ 0.3 GB/s; the hog saturates one direction of GPU 0's link (rx 26.5–27.6 GB/s h2d, tx 17.6–20.4 GB/s d2h) while
   rank 1's own traffic is unchanged → the harm is delayed rank-0 *sends* on the GPU0→host direction.
3. **NCCL transport confirmed** (job 1602541 logs, `raw/nccl-1602541-rank{0,1}.log`): `isAllDirectP2p 0 / isAllCudaP2p 0`,
   `Channel 00/01 : 0[0] -> 1[1] via SHM/direct/direct` (both communicators; GPUs at busId 41000 / 61000).
4. **Real mover read in 0.29.0** (`vllm/v1/kv_offload/cpu/gpu_worker.py`): GPU→CPU = copy engine `swap_blocks_batch`
   (cudaMemcpyBatchAsync) on its own stream, serialised, unpaced; CPU→GPU = Triton SM kernel for pages < 28 KiB (ours are 128 KiB →
   copy engine too). Enabled by `--kv-offloading-size <GiB> --kv-offloading-backend native` (needs prefix caching on).
5. **Duplicate-work update**: SGLang PR #34805 (2026-08-14, open, unmerged, opt-in) "Avoid H2D/A2A contention during layerwise
   offload" = same mechanism class (bulk DMA vs NCCL collective on PCIe-only boxes; windowing remedy) for diffusion weight prefetch.
   Task 26 novelty narrowed (ledger + README). vLLM/TRT-LLM/Dynamo/arXiv: nothing on offload traffic vs collectives/peer ranks.
6. Real-mover harness drafted and pushed, untested: `measure_kvoffload.py` (plain / store train / load train on rank 0 with the native
   connector; rank 1 = B decoders) + `sbatch_c26kv.sh <model> graph multiport 512 <OFFLOAD_GIB>`.
7. Pre-registered predictions for rounds 2–3 and the real-mover projection (README) before any cell ran.

## Measured results (actual numbers only)
(unchanged from round 56 for job 1602508; new this round:)
- Standalone idle 4090 (job 1602548, node wqd10naf13g7, driver 580.105): h2d ce 26.2 / sm 26.3 GB/s; d2h ce 26.3 / sm 26.3 GB/s
  (4, 12, 24, 48 Triton programs all 26.3); d2d ce 454 / sm 221 GB/s; `--rate-gbs 8` → 8.0 GB/s.
- dmon medians (job 1602508, MB/s, GPU0 rx/tx | GPU1 rx/tx): off B=8 6399/889 | 6038/846; B=32 5853/1064 | 5524/1019;
  h2d:1.0 26513/3228 | 6594/852 (B=8), 27628/3546 | 5782/1015 (B=32); d2h:1.0 3713/20353 | 7565/907, 3431/17552 | 8042/1072;
  d2d:1.0 2039/358 | 12344/1258, 1753/459 | 6488/864; h2d:cap4 10474/1344 | 5996/832.
- Round 2 / round 3 numbers: **see the refreshed section below if present; otherwise not yet analyzed** (jobs running at write time).

## Alive hypotheses (evidence + gate)
- **Task 26 — ALIVE / strong on the mechanism** (job 1602508: rank-1 p95 ITL ×1.28–1.48 under line-rate d2h on GPU 0; gate +10 %),
  **novelty narrowed** (SGLang #34805 covers the mechanism class for weight prefetch/H2D/diffusion). Open before any policy claim:
  (a) round 2 duty/cap sweep (harmless d2h rate, p95 under bursty duty), (b) round 3 requester probe (Triton-SM D2H vs copy engine →
  decides one-line `_select_swap_blocks_fn` policy vs pacing), (c) real-mover run (`sbatch_c26kv.sh`, projected ≤ 10 %).
- Task 27 (spec-decode acceptance skew): drafted, untested, not submitted.
- P1 (FP8 EP transport bound), 30 (barrier-free replay): untouched.

## Killed hypotheses (reason + evidence)
- G1–G4: oracle 0.0 % in 45/45 cells (round 55).
- Task 21 (closed): contagion = feature's own per-step cost exported 1:1; ≤ ×1.13 at B=8, ≤ ×1.07 at B=32; penalties known upstream (#47540).
- Task 28: analytic ≤ 11 % byte saving at EP2 (< 1.3× gate).
- C1/29: provisionally < 1 % visible at B ≥ 8 (period = CUDA time); this round confirmed the DP sync uses Gloo not NCCL under async scheduling ("Disabling NCCL for DP synchronization").
- Task 24: downgraded to engineering-only. Tasks 22/23 deprioritised.

## Blocked hypotheses (specific blocker)
- None. WebSearch/curl from the Mac are not permitted headless; searches run on the login node (GitHub API + export.arxiv.org).

## Current strongest result (bounded wording)
On 2×RTX 4090 DP2/EP2 (vLLM 0.29.0, Qwen1.5-MoE-A2.7B, allgather_reducescatter EP over NCCL SHM/direct — no P2P, confirmed), a bulk
GPU→host copy-engine stream on rank 0's GPU (14.8–17.5 GB/s) raises rank 1's decode ITL ×1.20 (B=8) to ×1.47 (B=32) at p50 and
×1.28–1.48 at p95 although rank 1 moves no data; idle context ×1.00, host→GPU at 21 GB/s ×1.07–1.09, 4 GB/s cap ×1.00. PCIe counters
show the EP payload is ~0.3 GB/s and the baseline link already carries ~6 GB/s of NCCL polling reads; the harm is the delayed
rank-0 sends on the saturated GPU0→host direction. The mechanism class (DMA vs collective on PCIe-only nodes) has an open prior-art
PR in SGLang (#34805, weight prefetch); the KV/D2H/DP-EP-decode instance and the remedy choice are the remaining contribution,
and the realistic native-connector traffic is projected in the ≤ 10 % band (untested).

## Next exact action (the exact first command/file/experiment for the next session)
1. Analyze rounds 2 and 3 (both should be COMPLETED):
   `scripts/dp_ep/pc.sh 'cd /data/run01/scxi253/inference && for j in 1602541 1602557; do d=$(ls -d results/c26-$j-*); LD_LIBRARY_PATH=/tmp/scxi253/libfix /tmp/scxi253/venv-vllm/bin/python lab-scripts/analyze_pcie.py --dir $d --output results/c26-analysis/c26-$j.json | tail -40; python3 lab-scripts/analyze_dmon.py --dir $d --output results/c26-analysis/dmon-$j.json; done'`
   then `pcget.sh` the two `c26-analysis/*.json` + `waves.log` into `benchmarks/results/dp-ep-c26-pcie/raw/` and score the
   pre-registered predictions in the README (section "Pre-registered predictions"); apply the round-3 decision rule (H_a vs H_b).
2. If H_a (SM-issued D2H interferes much less): the minimal fix is env-gated Triton-for-GPU→CPU in
   `vllm/v1/kv_offload/cpu/gpu_worker.py::_select_swap_blocks_fn` (copy a patched file over the node-local venv in the sbatch);
   if H_b: pacing (bytes-per-step budget in `SingleDirectionOffloadingHandler.transfer_async`). Either way first run the real mover:
   `scripts/dp_ep/pc.sh 'cd /data/run01/scxi253/inference && sbatch lab-scripts/sbatch_c26kv.sh Qwen1.5-MoE-A2.7B-Chat graph multiport 512 16 && sbatch lab-scripts/sbatch_c26kv.sh Qwen1.5-MoE-A2.7B-Chat graph multiport 512 0'`
   (harness untested: watch the first `[r0]` line — `store` should report n ≈ 20–25 requests / 8 s ≈ 15–19 GiB KV; `load` needs
   `--kv-offloading-size 16` ≥ 10 × 0.75 GiB; check `metrics_rank0_after` for offload counters; if the server refuses
   `--kv-offloading-size` with prefix caching or DP, read `config/vllm.py:970-995`).
3. Gate for the policy line: real-mover store(ON)−store(OFF) or load(ON) rank-1 p95 > 10 % → build the fix + 3 interleaved A/B repeats;
   ≤ 5 % → record "mechanism real, realistic native-connector traffic below gate", keep the line as an upstream note, move to task 27
   (`sbatch lab-scripts/sbatch_c27.sh Qwen1.5-MoE-A2.7B-Chat graph multiport 512 "--speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":4,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}"`).

## Files/artifacts created (paths)
- `scripts/dp_ep/{pcie_hog.py (engine sm), measure_pcie.py (-sm specs), analyze_dmon.py, sbatch_c26r2.sh, sbatch_c26r3.sh, sbatch_hogtest.sh, measure_kvoffload.py, sbatch_c26kv.sh}`
- `benchmarks/results/dp-ep-c26-pcie/README.md` (round-2 groundwork, NCCL transport, real mover, prior art, pre-registered predictions),
  `raw/dmon-1602508.json`, `raw/hogtest-1602548.txt`, `raw/nccl-1602541-rank{0,1}.log`
- `docs/multigpu-opportunity-ledger.md` (task 26 row: prior art SGLang #34805, mechanism evidence, status wording)
- Cluster: `results/c26-1602541-*/`, `results/c26-1602557-*/` (running), `results/c26-analysis/dmon-1602508.json`, `logs/hogtest-1602548.out`.

## Risks / unresolved methodological issues
- Rounds 2/3 ran on different nodes (wqd10nba07g3 / g5) from job 1602508 (ratios are within-job vs `off`, so comparable; absolute GB/s differ by node/driver: 23–24.6 vs 26.3).
- The hog is still a separate process (not the worker's stream); the real-mover harness is untested and its store traffic is
  prefill-rate-bound (~2.4 GB/s) — a null result there is expected and would bound the practical impact of the native connector, not the mechanism.
- ITL p95 ≈ 2×p50 in temperature-0 streams (bimodal token delivery); p50 + step trace are the robust signals; p95 quoted because the gate names it.
- `nvidia-smi dmon` PCIe counters are 1 s samples of a ~20 ms hardware window — direction/magnitude are reliable, fine timing is not.
- Task 27 harness untested; home quota (1 GB) full — every cache is redirected in the sbatch scripts.
