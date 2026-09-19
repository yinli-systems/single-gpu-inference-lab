# Autonomous Research State

## Timestamp
2026-09-19 06:13 (Mac, UTC+4) = 10:13 cluster clock (UTC+8). Written by autonomous round 57 (session started 05:43).

## Git (branch, SHA, dirty files)
branch `dp-ep-waves`; commits this round: `c535f14` (hog `--engine sm`, analyze_dmon, sbatch_c26r2/r3/hogtest), `fb2e0ce`
(dmon counters, real mover, SGLang #34805), `8c791dd` (NCCL transport, real-mover harness draft), `14516cd` (pre-registered
predictions), `5a5a0bc` (draft state), then the final state commit (see `git log -1`). Working tree clean after it.
Cluster copies in `/data/run01/scxi253/inference/lab-scripts/` (md5 = repo files): pcie_hog.py 04e97116…, measure_pcie.py 976cb9d0…,
analyze_dmon.py, sbatch_c26r2.sh, sbatch_c26r3.sh, sbatch_c26topo.sh, sbatch_hogtest.sh, measure_kvoffload.py, sbatch_c26kv.sh.

## Running Slurm jobs (id, purpose, expected output path)
- **1602541** `c26r2-dpep` (node wqd10nba07g3, same-socket pair NUMA 1+0; cells from 10:02, 29/72 done at 10:11, ends ≈10:28):
  round 2 = d2h duty/cap sweep, h2d:cap12, 0.75 GiB bursts, both:cap8, NCCL log → `results/c26-1602541-Qwen1.5-MoE-A2.7B-Chat-graph-multiport-q512/{pcie.json,waves.log,hog/,dmon.log,nccl-*.log,trace/}`.
- **1602557** `c26r3-dpep` (node wqd10nba07g5, same-socket pair NUMA 2+0; 10/42 cells at 10:11, ends ≈10:24): round 3 = requester
  probe (copy engine vs Triton-SM, full rate and cap 8) → `results/c26-1602557-…/`.
- **1602580** `c26topo-dpep` (node wqd10nba07g4, 6 GPUs, started 10:08): topology round — picks a same-socket and a cross-socket GPU
  pair from the allocation (NUMA via sysfs; `pairs.txt`, `topo.txt`, `gpus.txt`), runs the DP2/EP2 server + hog per pair with
  `--specs off,d2h:1.0,h2d:1.0 --batch-sizes 8,32 --repeats 3 --seed 69` → `results/c26topo-1602580-…/{same,cross}/{pcie.json,waves.log,hog/,nccl-*.log,trace/}`, ≈ 2 × (4 min start + 7 min cells) after staging; `logs/c26topo-1602580.out`. Untested script: if `pairs.txt` is empty or a server dies, read the log.
- 1602548 `hogtest` COMPLETED (SM engine validation).
- Other jobs in `squeue -u $USER` (mbert149-glue, vft-*, moe2b-ds-*, fork1b, moe7b, f3b-prof) are the user's own unrelated work — never touch them.

## Completed this round
1. Rounds 2, 3 and the topology round of task 26 designed, harness extended, submitted (see above). `pcie_hog.py --engine sm` = Triton
   kernel through the pinned host pointer (validated on an idle 4090, job 1602548: ce and sm 26.2–26.3 GB/s each way; d2d ce 454 / sm 221 GB/s).
   `measure_pcie.py` accepts `d2h-sm:…`. The new hog file replaced the cluster copy during job 1602541 (`--engine` defaults to `ce`, same copy path).
2. **PCIe counters of job 1602508** (`analyze_dmon.py`, `raw/dmon-1602508.json`): plain baseline ~6 GB/s PCIe rx per GPU = NCCL host-memory
   transport polling (12.3 GB/s on the waiting GPU in the d2d cell); EP payload ≈ 0.3 GB/s; the hog saturates one direction of GPU 0's link,
   rank 1's own traffic unchanged.
3. **NCCL transport confirmed** (`raw/nccl-1602541-rank{0,1}.log`): `isAllDirectP2p 0`, `Channel 00/01 … via SHM/direct/direct` both ways.
4. **Real mover read in 0.29.0** (`vllm/v1/kv_offload/cpu/gpu_worker.py`): GPU→CPU = copy engine `swap_blocks_batch`, own stream, serialised,
   unpaced; CPU→GPU = Triton SM kernel for pages < 28 KiB (ours 128 KiB → copy engine). Enabled by `--kv-offloading-size <GiB>` (prefix caching on).
5. **Prior art**: SGLang PR #34805 (open, unmerged, opt-in) = same mechanism class (bulk H2D vs NCCL all-to-all on PCIe-only boxes, no-DMA window)
   for diffusion weight prefetch → task 26 novelty narrowed (README + ledger). vLLM/TRT-LLM/Dynamo/arXiv: nothing on offload traffic vs collectives.
6. **Node topology discovered** (srun --overlap on the running jobs): 8 × 4090 per node, one GPU per PCIe root port, 2-socket 8-NUMA EPYC
   (NPS4), no PCIe switches, `nvidia-smi topo -m` = SYS; Slurm pairs for rounds 2/3 were same-socket; round 1's pair (node wqd10nba06g6) unlogged.
7. Real-mover harness drafted and pushed, untested: `measure_kvoffload.py` + `sbatch_c26kv.sh <model> graph multiport 512 <OFFLOAD_GIB>`.
8. Pre-registered predictions for rounds 2–3 written before any cell ran (README), partially scored below.

## Measured results (actual numbers only)
Round 2 (job 1602541, same-socket pair, first repeat; rank-1 ITL p50 ms, ratio vs `off` at same B):
- off B=8 13.6, B=32 21.9–22.0. d2h:1.0 (hog 22.3 / 19.5 GB/s, not slowed) 14.8 (×1.09) / 25.3–25.5 (×1.16); d2h:1.0:0.75 15.0 (×1.10) / 22.6–22.8 (×1.04);
  d2h:0.5 ×1.06 / 22.4–23.3; d2h:0.25 ×1.13 (B=8); d2h cap2/4/8/12 at B=8 ×1.04/1.07/1.01/1.08, at B=32 22.4–22.6 / 22.3–23.0 / 22.3–22.5 / 23.2–23.8;
  h2d:cap12 ×1.05 (B=8); both:cap8 ×0.97 (B=8). p95 B=8: off 26.9, d2h:1.0 29.7 (×1.11), d2h:0.25 31.4 (×1.17), cap8 27.8 (×1.03).
- vs round 1 (job 1602508, other node/pair, 3 repeats): d2h:1.0 ×1.20 / ×1.47 with the hog slowed to 17.5 / 14.8 GB/s.
Round 3 (job 1602557, same-socket pair, first repeat): d2h-sm:1.0 (8.3 GB/s achieved) rank-1 p50 43.1–43.4 at B=32 (×1.96 vs 22.0), 28.2–29.4 at B=8 (×2.0);
  d2h-sm:cap8 (5.3 GB/s) 32.4–32.7 at B=32 (×1.47); d2h:1.0 (18.0 GB/s) 14.9–15.0 at B=8 (×1.09); d2h:cap8 B=32 22.6–22.8 (×1.03); h2d:1.0 B=8 15.2–15.5.
Standalone links: idle node (1602548) ce/sm 26.2–26.3 GB/s both ways; node 07g5 h2d ce 21.5 / sm 24.7, d2h ce 22.7 / sm 18.2 GB/s; node 07g3 d2d 450 GB/s.
dmon medians (job 1602508, MB/s, GPU0 rx/tx | GPU1 rx/tx): off B=8 6399/889 | 6038/846; h2d:1.0 26513/3228 | 6594/852; d2h:1.0 3713/20353 | 7565/907; d2d 2039/358 | 12344/1258.

## Alive hypotheses (evidence + gate)
- **Task 26 — ALIVE but re-scoped**: the mechanism (bulk copy-engine D2H on rank 0's GPU degrading rank 1 via NCCL SHM transport) is real on
  the round-1 pair (×1.47 p50 / ×1.48 p95 at B=32, 3 repeats) but only ×1.16 / ×1.09 on same-socket pairs (1 repeat) — **topology-dependent;
  cross-socket hypothesis under test (job 1602580)**. Gate (rank-1 p95 +10 %) passed on the round-1 pair, marginal on same-socket pairs.
  New sub-finding: SM-issued copies (vLLM's Triton swap path shape) cost the peer ×2 while resident (SM contention exported by lockstep).
  Policy candidates now: (i) NUMA-/socket-aware placement of offload buffers or GPU pairs (if 1602580 confirms), (ii) prefer the copy engine
  over the Triton path under DP/EP, (iii) pacing only if the cross-socket run shows a rate threshold. All need the real-connector run.
- Task 27 (spec-decode acceptance skew): drafted, untested, not submitted. P1, 30: untouched.

## Killed hypotheses (reason + evidence)
- H_a of round 3 ("SM-issued D2H interferes less"): rejected, ×1.96–2.0 vs ×1.09 for the copy engine (SM contention, like the d2d control).
- Round-2 predictions 1–2 (linear duty scaling / monotone rate caps) not supported on the same-socket pair (residual ×1.02–1.10, no rate order).
- G1–G4 (oracle 0.0 % in 45/45 cells); task 21 (closed, features ≤ ×1.13, penalties known upstream #47540); task 28 (analytic ≤ 11 % at EP2);
  C1/29 (< 1 % visible at B ≥ 8; DP sync is Gloo under async scheduling); task 24 downgraded; 22/23 deprioritised.

## Blocked hypotheses (specific blocker)
- Round-1 topology cannot be recovered (pair not logged; from now on every sbatch logs `nvidia-smi topo -m` + bus ids — c26topo does).
- 8-GPU single-node allocations unavailable in the partition right now (0 runnable 8-GPU jobs); 6 GPUs obtained.

## Current strongest result (bounded wording)
On 2×RTX 4090 DP2/EP2 (vLLM 0.29.0, Qwen1.5-MoE-A2.7B, EP over NCCL SHM/direct — no P2P, confirmed), a line-rate GPU→host copy-engine
stream on rank 0's GPU raised rank 1's decode ITL ×1.47 (p50) / ×1.48 (p95) at B=32 on one GPU pair (3 repeats), but only ×1.16 on two
same-socket pairs (1 repeat each) where the stream is not slowed by the server; the peer's cost is therefore set by the host-side path the
copy and NCCL's host-memory traffic share, not by GPU 0's link alone. SM-issued copies of the same bytes cost the peer ×2 regardless.
The mechanism class has an open prior-art PR (SGLang #34805); realistic native-connector traffic is projected ≤ 10 % (untested).

## Next exact action (the exact first command/file/experiment for the next session)
1. Analyze all three jobs (2/3 should be COMPLETED; topo may still run — check `squeue`):
   `scripts/dp_ep/pc.sh 'cd /data/run01/scxi253/inference && for d in results/c26-1602541-* results/c26-1602557-* results/c26topo-1602580-*/same results/c26topo-1602580-*/cross; do echo "## $d"; LD_LIBRARY_PATH=/tmp/scxi253/libfix /tmp/scxi253/venv-vllm/bin/python lab-scripts/analyze_pcie.py --dir $d --output results/c26-analysis/$(basename $(dirname $d))-$(basename $d).json | grep -A30 "rank-1 (no KV"; done; cat results/c26topo-1602580-*/pairs.txt; cat results/c26topo-1602580-*/topo.txt'`
   then `pcget.sh` the analysis JSONs + `waves.log` + `pairs.txt`/`topo.txt` into `benchmarks/results/dp-ep-c26-pcie/raw/`, replace the
   "partial" table in the README with 3-repeat numbers, and score: cross-socket d2h ×≥1.3 and same-socket ≤ ×1.16 → topology confirmed.
2. If topology confirmed: policy = socket-aware placement (offload buffer NUMA-binding / pair selection) — check where vLLM's connector
   allocates its pinned CPU tensors (`kv_offload/cpu/…` `torch.empty(pin_memory=True)`, no NUMA policy) and whether Slurm's CPU binding
   already lands them on the GPU's socket; then the real-mover run on a cross-socket pair (extend `sbatch_c26kv.sh` with the pair logic of
   `sbatch_c26topo.sh`): `sbatch lab-scripts/sbatch_c26kv.sh Qwen1.5-MoE-A2.7B-Chat graph multiport 512 16` and `… 0`.
   If not confirmed (both pairs ≤ ×1.2): task 26 = engineering band (5–10 %); record, and move to task 27
   (`sbatch lab-scripts/sbatch_c27.sh Qwen1.5-MoE-A2.7B-Chat graph multiport 512 "--speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":4,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}"`).
3. Either way record the SM-copy ×2 finding as a separate upstream-relevant note (vLLM `swap_blocks_triton` under DP/EP) with the real
   connector as the required next evidence (`measure_kvoffload.py` load cells use the copy engine for 128 KiB pages, so a Triton-path test
   needs a model with pages < 28 KiB or `THRESHOLD_BYTES` raised in the node-local venv copy).

## Files/artifacts created (paths)
- `scripts/dp_ep/{pcie_hog.py (engine sm), measure_pcie.py (-sm specs), analyze_dmon.py, sbatch_c26r2.sh, sbatch_c26r3.sh, sbatch_c26topo.sh, sbatch_hogtest.sh, measure_kvoffload.py, sbatch_c26kv.sh}`
- `benchmarks/results/dp-ep-c26-pcie/README.md` (round-2 groundwork, NCCL transport, real mover, prior art, pre-registered predictions,
  partial rounds 2–3 + topology), `raw/{dmon-1602508.json, hogtest-1602548.txt, nccl-1602541-rank0.log, nccl-1602541-rank1.log, c26-1602541-partial.json}`
- `docs/multigpu-opportunity-ledger.md` (task 26 row: prior art, mechanism evidence, topology dependence, requester result; C1/29 Gloo note)
- Cluster: `results/c26-1602541-*/`, `results/c26-1602557-*/`, `results/c26topo-1602580-*/`, `results/c26-analysis/{dmon-1602508.json,c26-1602541-partial.json}`, `logs/hogtest-1602548.out`.

## Risks / unresolved methodological issues
- Rounds 2/3 numbers are single-repeat at write time (jobs finish ≈10:24–10:28 cluster); the topology job is an untested script (pair
  selection in an inline python heredoc; `continue 2` on server death) — verify `pairs.txt` first.
- Round 1's GPU pair is unknown; the topology claim rests on 1602580. Even with a cross-socket pair, the pinned buffers' NUMA node
  (harness-spawned hog, first-touch on whatever core Slurm gives) is uncontrolled — log `numactl -s`/`taskset -p` in the hog next.
- The requester probe is confounded by SM occupancy (12 resident Triton programs); the "copy engine is benign" conclusion is still
  the relevant product comparison, but "PCIe requester" per se was not isolated.
- Real-mover harness untested; native-connector store traffic is prefill-rate-bound (~2.4 GB/s) → an expected null there bounds the
  practical impact of the native connector, not the mechanism.
- ITL p95 ≈ 2×p50 in temperature-0 streams (bimodal token delivery); p50 + step trace are the robust signals; p95 quoted because the gate names it.
- `nvidia-smi dmon` PCIe counters are 1 s samples of a ~20 ms window — direction/magnitude reliable, timing not.
- Task 27 harness untested; home quota (1 GB) full — every cache is redirected in the sbatch scripts.
