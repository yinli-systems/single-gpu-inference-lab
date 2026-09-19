# Autonomous Research State

## Timestamp
2026-09-19 11:15 (Mac, UTC+4) ≈ 15:02 cluster clock (CST; cluster ≈ Mac + 3 h 47 min). Written by autonomous round 61 (session started 10:36 Mac / 14:23 cluster). CPU-only session again: 0 free GPUs cluster-wide, all my jobs PENDING (Slurm estimates 2026-09-21T16:31 for the 1-GPU jobs and — after the time-limit cut — also for the 2-GPU c27 job).

## Git (branch, SHA, dirty files)
branch `dp-ep-waves`; base at session start `f7d5f68`; this round's commit = `git log -1` (analyze_sidecar.py, cpusidecar.py upgrade, pinner.py, cpuplan.py, sbatch_c26kv4.sh, README section "Slow state: the cpuset finding…", ledger row 26, state). Working tree clean after it.
Cluster copies in `/data/run01/scxi253/inference/lab-scripts/`: analyze_sidecar.py, cpusidecar.py (round-61 version: sibling map + per-CPU busy + Cpus_allowed), pinner.py, cpuplan.py, sbatch_c26kv4.sh (all = repo files) plus the round-59/60 set.

## Running Slurm jobs (id, purpose, expected output path)
All **PENDING (Priority)**; time limits cut this round (`scontrol update TimeLimit`) to help backfill — every earlier lab job finished in ≤ 39 min.
- **1603071** `c26kv4-place` (1 GPU, 6 threads, 1:30, new): placement diagnostic, DP=1, four arms in one allocation: `A-on-shared` (= 1603052 arm A), `D-on-squeezed` (tree + harness on 2 cores), `B-on-isolated` (EngineCore / API server / harness+sidecar each on their own core via `pinner.py`), `C-off-shared`; harness `--kinds store,plain --batch-sizes 32 --repeats 5` → `results/c26kv4-1603071-Qwen1.5-MoE-A2.7B-Chat-q512/{A-on-shared,D-on-squeezed,B-on-isolated,C-off-shared}/{cpusidecar.jsonl,kvoffload.json,waves.log,trace/step.jsonl,server.log,affinity-ready.txt,pinner.txt(B),ps-*.txt}` + `{cpuplan.txt,gpus.txt,lscpu-e.txt,numactl.txt,sysctl.txt}`; log `logs/c26kv4-1603071.out`. Pre-registered P1–P4 in the script header and README.
- **1603052** `c26kv3-mode` (1 GPU, 1:30): arms A on/unbound, B on/`--numa-bind`, C off. **Arm B is expected to die at engine start** (auto NUMA detection refuses the constrained affinity → RuntimeError; numactl probably absent) — the script continues with C; A and C run with the upgraded sidecar (read at run time) → `results/c26kv3-1603052-…/{A-on-unbound,B-on-numabind,C-off-unbound}/`, log `logs/c26kv3-1603052.out`.
- **1603014** / **1603015** `c26kv1-dp1` (1 GPU each, 1:00): DP=1 connector ON (16 GiB) / OFF controls → `results/c26kv1-16030{14,15}-…-off{16,0}/`.
- **1603006** `c27-dpep` (2 GPUs, 1:00): task 27 round 2 (`rndT`) → `results/c27-1603006-*/spec.json`; predictions in `benchmarks/results/dp-ep-c27-spec-skew/README.md`.
- Other jobs in `squeue -u $USER` (vft-*, moe2b-ds-*, moe7b, f3b-prof) are the user's own unrelated work — never touch them.

## Completed this round
1. Time limits of the four pending lab jobs cut from 3:00 to 1:00–1:30 (all earlier lab jobs ≤ 39 min) — c27's estimate moved from 09-22 to 09-21.
2. `scripts/dp_ep/analyze_sidecar.py` written and validated on a synthetic fixture (old and new sidecar formats): joins `cpusidecar.jsonl` with the step trace on the monotonic clock, labels samples slow/fast/nochunk/idle by the sustained-run rule, prints the per-run timeline and per-state table (util, schedstat wait %, nv/s, cpu:node, on-GPU-node %, migrations, **SMT-sibling busy %**, Cpus_allowed, PSI cpu/mem, MemAvailable, Shmem) and the S-place / S-smt / S-contend / S-mem readings.
3. **Premise corrected**: the Slurm cpuset is NOT unconfined — `raw/gpus-numa-kv.txt` shows `--cpus-per-task` threads allocated as whole cores + SMT siblings (1602619: `6-8,36-38,70-72,100-102`, nodes 0/4, GPUs on nodes 1/7; 1602620: nodes 1/2 = the GPU nodes). Site cap: 6 CPUs per GPU on gpu_4090 (`DefCpuPerGPU=6`, submit filter rejects more). Read `vllm/utils/numa_utils.py` (0.29.0): `--numa-bind` needs `numactl` on PATH, refuses a constrained affinity unless `--numa-bind-nodes` is given, and `--cpunodebind` to a node without allowed CPUs fails → not usable in a Slurm cpuset.
4. Sidecar upgraded (`cpusidecar.py`: static `siblings` + `node_of`, per-sample `cpu_busy` for every CPU from `/proc/stat`, per-thread `allowed`), `pinner.py` (numactl-free per-process affinity: EngineCore vs rest) and `cpuplan.py` (splits the cpuset into whole cores → ENGINE/API/HARN/SQUEEZE) written, smoke-tested on the login node (Linux) with a dummy tree.
5. `sbatch_c26kv4.sh` written (12-CPU version rejected by the site filter → 6 threads / 3 cores), submitted as job 1603071.
6. README section "Slow state: the cpuset finding and the placement diagnostic (round 61)", ledger row 26 updated.

## Measured results (actual numbers only)
- No new GPU data this round. Cluster facts: gpu_4090 partition `DefCpuPerGPU=6`, nodes 96 threads (48 cores) or 128 threads (64 cores) with 8 GPUs; node 07g6 (the slow-state node of 1602608) currently CPUAlloc 46/96 with all 8 GPUs allocated, CPULoad 43.5.
- Job cpusets (from `gpus.txt`): 1602619 `6-8,36-38,70-72,100-102` (6 cores on NUMA 0 and 4; GPUs NUMA 1 and 7); 1602620 `6-8,12-14,54-56,60-62` (6 cores on NUMA 1 and 2; GPUs NUMA 2 and 1). 1602608/1602609 did not log the cpuset (same `--cpus-per-task 12`).
- Round-60 numbers (slow state: chunk steps 85 vs 48 ms, graph steps unchanged, host slack 2.9–3.7 vs 1.4–1.6 ms, slow fraction ON 0.60 / OFF 0.13 / connector-free ≤ 0.05) unchanged — see the c26 README.

## Alive hypotheses (evidence + gate)
- **Task 26 — ALIVE (hog: strong on cross-socket pairs; real connector: reframed)**. Hog evidence unchanged (buffer socket +×1.14–1.16, pair span +×1.15, worst corner ×1.48 at B=32, 3 repeats). Real connector: loads/stores benign for the peer's decode steps (×0.97–1.02); the peer's p95 ×1.80 = lockstep export of a CPU slow state of the storing rank. **New concrete candidate: SMT-sibling co-scheduling / run-queue sharing of the EngineCore launch thread inside the 6-core cpuset** (S-smt / S-contend), vs S-place (far NUMA node: constant per job, cannot explain flips) vs S-mem. Decided by 1603071 (P1–P4) and 1603052 arms A/C. Gate unchanged (rank-1 p95 +10 %); if isolation cures it the remedy is a per-rank core budget / pinning (engineering; `--numa-bind` not applicable in cpusets — an upstream usability gap worth recording, not a mechanism).
- **Task 27 — undecided**: round 2 (1603006) pending with pre-registered predictions (README). Gate: <5 % kill, >10 % continue, >10 % group-aware over per-rank control = strong.

## Killed hypotheses (reason + evidence)
- "Slurm leaves the cpuset unconfined" (rounds 58–60 premise): `taskset -cp $$` logs show 12 confined threads = 6 cores + siblings.
- "`--numa-bind` is the remedy candidate" in its literal form: not usable inside a constrained cpuset (numa_utils.py: constrained affinity → auto-detect skipped → RuntimeError; needs numactl; cpunodebind to a node with no allowed CPUs fails). Kept only as "upstream flag cannot address this deployment shape".
- H1 per-copy form; tier fill / eviction; "the GPU is slower in the slow state"; within-socket buffer node; H2D loads interfere; D2H stores hit ordinary decode steps; round-2 predictions 1–2; H_a round 3; "GPU 0's PCIe link is the contended resource"; G1–G4 (0.0 %); task 21 (≤ ×1.13, #47540); task 28 (≤ 11 % at EP2); C1/29 (< 1 % at B ≥ 8); task 24 downgraded; 22/23 deprioritised — all unchanged.

## Blocked hypotheses (specific blocker)
- Everything GPU-side: 0 free GPUs cluster-wide; five lab jobs pending (estimates 2026-09-21T16:31). Nothing can be forced.
- A 12-CPU 1-GPU job is impossible on gpu_4090 (site filter, 6 CPUs/GPU); a DP2 isolation run (2 GPUs, 12 threads: each EngineCore on 2 own cores) is the production-shaped follow-up — needs `pinner.py` to take per-EngineCore CPU sets (not yet written) and a `sbatch_c26kv.sh` copy.
- H2 (cross-socket collective slowdown by store bursts) moot unless the DP=1 arms show no slow state and the DP2 chunk step is still slow on a same-socket pair.

## Current strongest result (bounded wording)
On 2×RTX 4090 DP2/EP2 (vLLM 0.29.0, Qwen1.5-MoE-A2.7B, NCCL SHM/direct), a line-rate GPU→host copy-engine stream on rank 0's GPU raises rank 1's decode ITL through two additive host-side placement factors at B=32 (3 repeats, `move_pages`-verified): far-socket pinned buffer +×1.14–1.16 and a DP/EP pair spanning sockets +×1.15; worst corner ×1.48 (vLLM's NUMA-agnostic offload tier can hit it); same-socket pair with a local buffer ×1.13. vLLM's real CPU-offload connector moves KV at 10–12.7 GB/s (store) / 21.5 GB/s (load) without touching the peer's decode steps (×1.01–1.02; loads help: peer ITL ×0.5, TTFT 2.5× better). The peer's p95 ×1.80 with the connector on is the lockstep export of a sustained CPU-side slow state of the storing rank's worker (eager chunk steps ×1.8, graph steps unchanged, 40–80-s runs flipping mid-prompt), which ran with two EngineCores, the API server, the harness and NCCL proxies confined to a 6-core Slurm cpuset; whether SMT-sibling sharing / run-queue contention inside that cpuset is the cause (and pinning cures it) is the pending diagnostic 1603071 (+1603052 arms A/C).

## Next exact action (the exact first command/file/experiment for the next session)
1. `scripts/dp_ep/pc.sh 'squeue -u $USER -h -o "%i %j %t %M %S %N %R" | grep -E "c26|c27"; cd /data/run01/scxi253/inference; ls results/ | grep -E "1603071|1603052|1603014|1603015|1603006"; tail -n 3 logs/c26kv4-1603071.out logs/c26kv3-1603052.out logs/c26kv1-1603014.out logs/c26kv1-1603015.out logs/c27-1603006.out 2>/dev/null'` (this site's `tail` rejects `-3`; use `-n 3`).
2. When 1603071 (or 1603052) has any finished arm: `scripts/dp_ep/pc.sh 'cd /data/run01/scxi253/inference; P="LD_LIBRARY_PATH=/tmp/scxi253/libfix /tmp/scxi253/venv-vllm/bin/python"; R=$(ls -d results/c26kv4-1603071-* | head -1); cat $R/cpuplan.txt $R/gpus.txt; for a in A-on-shared D-on-squeezed B-on-isolated C-off-shared; do echo == $a; cat $R/$a/affinity-ready.txt $R/$a/pinner.txt 2>/dev/null; grep "\[r" $R/$a/waves.log; eval $P lab-scripts/modeseries.py $R/$a | grep -E "SLOW|FLIP|== rank"; eval $P lab-scripts/analyze_sidecar.py $R/$a --json results/c26-analysis/sidecar-1603071-$a.json | tail -n 45; done'` — score P1–P4 (README section "Slow state: the cpuset finding…"); pull the JSONs + printed outputs to `benchmarks/results/dp-ep-c26-pcie/raw/`. Same loop for 1603052 with arms `A-on-unbound B-on-numabind C-off-unbound` (expect B "server died"; check `B-on-numabind/server.log` for the numa_utils message and record it).
3. If P1 + P3 hold (core sharing is the mechanism): write the DP2 production-shaped follow-up — extend `pinner.py` to accept `--engine-cpus "a;b"` (per EngineCore, by pid order = DP rank), copy `sbatch_c26kv.sh` → `sbatch_c26kv5.sh` with arms on-shared / on-isolated / off, submit; the claim then becomes "with N cores per rank the connector's store phase exports ×1.8 to every rank in lockstep; pinning removes it" (engineering, measured).
4. When 1603014/1603015 finish: `modeseries.py` + `chunkcheck.py` on both (slow state at DP=1 ON → process-local; absent → node/neighbour-specific). When 1603006 finishes: `analyze_spec.py --dir results/c27-1603006-* --output results/c27-analysis/c27-1603006.json`, score the README predictions.
5. If nothing runs: the CPU-side items are the DP2 pinner extension (item 3, can be written ahead) and the arXiv duplicate search for task 27 (still not done).

## Files/artifacts created (paths)
- `scripts/dp_ep/analyze_sidecar.py` (new), `scripts/dp_ep/pinner.py` (new), `scripts/dp_ep/cpuplan.py` (new), `scripts/dp_ep/sbatch_c26kv4.sh` (new), `scripts/dp_ep/cpusidecar.py` (upgraded: siblings, cpu_busy, allowed)
- `benchmarks/results/dp-ep-c26-pcie/README.md` (section "Slow state: the cpuset finding and the placement diagnostic (round 61)")
- `docs/multigpu-opportunity-ledger.md` (row 26: round-61 note, gate column)
- Cluster: `lab-scripts/{analyze_sidecar.py,pinner.py,cpuplan.py,sbatch_c26kv4.sh,cpusidecar.py}`, job 1603071

## Risks / unresolved methodological issues
- The slow state is still N=1 (one run, one node, 07g6); 1603071 arm A on 3 cores may not reproduce it (fewer busy threads than the DP2 job: one EngineCore, no NCCL proxies) — arm D (2 cores) is the provocation; if neither reproduces it, the DP2 follow-up (item 3) is needed regardless.
- The cpusets of 1602608/1602609 were not logged; the 6-core budget is inferred from `--cpus-per-task 12` and the two logged jobs.
- `pinner.py` pins after the server is up (startup runs unpinned; first-touch memory placement unchanged) — fine for the CPU question, not a NUMA-memory test. It matches EngineCore by cmdline/comm ("EngineCore"); verify in `pinner.txt` that exactly one process is labelled ENGINE.
- Per-CPU busy from `/proc/stat` includes every job on the node (intended: a busy sibling from a neighbour counts), sampled at 0.5 s.
- 1603052 arm B will most likely die (numa_utils) — that is expected and documented; its A/C arms remain a within-node ON/OFF control.
- Task 27 `rndT` adds seeded temperature sampling (~+1 ms/step) to the rndT rank only — compare against `rndT|rndT`, never `rep|rep`. ITL p95 ≈ 2×p50 in temperature-0 streams; p50 + step trace are the robust signals. Home quota (1 GB) full — every cache is redirected in the sbatch scripts.
