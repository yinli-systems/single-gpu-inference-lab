# Autonomous Research State

## Timestamp
2026-09-19 05:25 (Mac, UTC+4) = 09:25 cluster clock (UTC+8). Written by autonomous round 55.

## Git (branch, SHA, dirty files)
branch `dp-ep-waves`, HEAD `bf8c8a6` (task 26 hog draft) on top of `bb647a3` (G1 killed, task 21 round 1) and `0f39577`; this final
state-file/ledger commit on top. Working tree clean after the final commit of this round.
Cluster copies of scripts: `/data/run01/scxi253/inference/lab-scripts/` (measure_contagion.py
there is the fixed round-2 version, md5 must match the repo file).

## Running Slurm jobs (id, purpose, expected output path)
- **1602499** `c21-dpep` RUNNING since 09:19 cluster time (node wqd10naf05g4): task 21 round 2 —
  `sbatch_c21.sh Qwen1.5-MoE-A2.7B-Chat graph multiport 512 "" "--features sampling,pen,topk,temp,plogp,pfx,logprobs20 --batch-sizes 8,32 --repeats 3 --seed 62"`;
  90 windows ≈ 47 min + 4 min startup → expected done ≈ 10:15 cluster time. Output:
  `/data/run01/scxi253/inference/results/c21-1602499-Qwen1.5-MoE-A2.7B-Chat-graph-multiport-q512/{contagion.json,waves.log,trace/step.jsonl.dp0,dp1}`,
  log `/data/run01/scxi253/inference/logs/c21-1602499.out` (ends with `C21_DONE`).
- Other jobs in `squeue -u $USER` (mbert149-glue, vft-smoke-lora, moe2b-ds-*, fork1b, moe7b) are the
  user's own unrelated work — never touch them.

## Completed this round
1. Analyzed the three G1 arms that finished after round 54: q128 graph (1602472) vs q128 eager
   (1602473); q512 graph with `--max-cudagraph-capture-size 512` (1602484) vs q512 eager (1602456).
   Both oracle gains 0.0 % (0/15 ITL cells, 0/12 TTFT cells). **G1–G4 KILLED** under the
   pre-registered gate; README verdict written (`benchmarks/results/dp-ep-g1-padding/README.md`,
   raw analyzer JSON/MD under `raw/g1-analysis/`).
2. Task 21 round 1 (job 1602481) finished and analyzed (`benchmarks/results/dp-ep-c21-contagion/`).
   Found that the `plogp` cells were invalid (vLLM 400: prompt_logprobs not allowed with
   stream=True); fixed the harness (non-streaming prompt train) and added decomposition features
   `pen`, `topk`, `temp`, `logprobs20`; submitted round 2 (1602499).
3. Duplicate-work check (GitHub search from the login node): vLLM PR #47540 (open, 2026-07-03)
   already reports the penalties-path DP straggler ("whichever rank holds the longest output
   stalls every other rank at the DP sync point") with a fix → sampling contagion downgraded to a
   replication. Ledger updated (G1 killed, 21 downgraded, 24 queued).

## Measured results (actual numbers only)
G1 round 2 (q=128, cap 128, pure padding): pd cells execute 128 tokens on the decode rank for
8/16/32 true (A_pad 1.88/1.78/1.60, exec_amp 1.83/1.66/1.49, PIECEWISE graphs both ranks); decode-rank
ITL p50 A/B = 23.5/49.3, 24.4/50.6, 25.5/51.1 ms (4k), TTFT 861/1874, 938/2389, 1154/2275 ms;
pd wave period 23.3–27.2 ms = pp period 23.7–27.4 ms (padding does not extend the wave).
G1 round 3 (q=512, cap 512): decode rank padded to 512 on chunk steps (exec max 512, exec_amp
1.81/1.56/1.42); chunk-step cuda p95 40.5–48.4 ms (padded graph) vs 47.9–58.8 ms (eager);
TTFT vs round 1 (cap 128): pd 8/4096 452→380, pd 8/8192 861→724, pp 32/8192 900→849 ms (−6…−16 %)
with decode ITL unchanged (37.4 vs 38.9; 17.7 vs 17.7).
Eager floor: 44–50 ms/step for every shape (8,8)…(512,512); graph waves 13.2 (B=8) / 16.4 (16) /
21.5 (32) ms.
Task 21 round 1 (plain-rank ITL p50 ratio vs plain|plain; B=8 / B=32; both placements):
logprobs 1.01,1.00 / 1.01,1.02; struct 0.95,0.94 / 0.96,1.00; sampling 1.13,1.12 / 1.09,1.10
(step period both ranks 14.3→15.9–16.2 and 21.1→23.1–23.6 ms); pfx (prefill train) 1.92,1.15 /
1.34,1.33 (B=8 bimodal across repeats 16–30 ms); plogp invalid (400s; peer ITL 0.97–1.03).

## Alive hypotheses (evidence + gate)
- Task 21 residual: which part of the sampling bundle is exported (pen vs topk vs temp), and the
  fixed prompt-logprobs cell — round 2 running. Gate unchanged (<5 % kill / 10 % alive / 20 %
  strong); anything reproducing PR #47540's penalties mechanism is a replication, not a line.
- Task 26 (PCIe KV movement vs EP communication) — not started; next after 21 round 2.
- Task 24 (group-aware chunk budget) — G1 data give the heavy-rank q ∈ {128, 512} points; needs a
  prior-art check and a fixed prefill-train harness before any oracle claim.
- P1 (FP8 EP transport upper bound), C1/29 (DP control-plane tax) — untouched, lower priority.

## Killed hypotheses (reason + evidence)
- **G1–G4 (cross-rank padding amplification, graph-vs-eager admission)**: oracle gain 0.0 % in all
  45 cells across three regimes; eager arm never wins (44–50 ms launch floor vs ≤29 ms graph waves);
  padded rank's step ≤ heavy rank's step so padding never lengthens the wave (round 2 pd period =
  pp period; round 3 padding to 512 leaves ITL unchanged). Evidence:
  `benchmarks/results/dp-ep-g1-padding/raw/{g1-oracle-graphcap128-1602471-vs-eager-1602456.json,g1-analysis/*.json}`.
- Task 21 for logprobs (top-5) and structured output (xgrammar): ≤2 % → killed.
- Task 21 sampling: real (+9–13 %) but mechanism + fix already upstream (vLLM PR #47540) → not a line.

## Blocked hypotheses (specific blocker)
- None hard-blocked. Note: WebSearch/curl from the Mac are not permitted in the headless session;
  GitHub API searches must run on the login node via `pc.sh 'curl -s https://api.github.com/...'`.

## Current strongest result (bounded wording)
Negative: on 2×4090 DP2/EP2 with Qwen1.5-MoE-A2.7B (vLLM 0.29.0), synchronized DP padding is
user-invisible (0/45 cells where true-shape eager beats padded graphs; wave = heavy rank's chunk in
every regime), while the same data show that a chunked prefill on one rank raises the *other*
rank's decode ITL by ×1.3–1.9 (B=32/8, q=512) — a synchronization cost, not a padding cost.
Engineering side-note: `--max-cudagraph-capture-size 512` cuts TTFT 6–16 % at q=512 with no ITL
cost. Cross-rank feature contagion equals the feature's own per-step cost (lockstep export is
1:1); only the sampling/penalties bundle (+1.7–2.4 ms/step) is measurable and it is known upstream.

## Next exact action (the exact first command/file/experiment for the next session)
1. `scripts/dp_ep/pc.sh 'squeue -u $USER -h -o "%i %j %t %M"; tail -3 /data/run01/scxi253/inference/logs/c21-1602499.out'`
   — if `C21_DONE`: `scripts/dp_ep/pc.sh 'cd /data/run01/scxi253/inference; LD_LIBRARY_PATH=/tmp/scxi253/libfix /tmp/scxi253/venv-vllm/bin/python lab-scripts/analyze_contagion.py --dir results/c21-1602499-Qwen1.5-MoE-A2.7B-Chat-graph-multiport-q512 --output results/c21-analysis/c21-round2-1602499.json > results/c21-analysis/c21-round2-1602499.md 2>&1; sed -n "/### contagion/,\$p" results/c21-analysis/c21-round2-1602499.md'`,
   then `scripts/dp_ep/pcget.sh /data/run01/scxi253/inference/results/c21-analysis/ benchmarks/results/dp-ep-c21-contagion/raw/` and add "Round 2" to that README
   (pen vs topk vs temp decomposition, plogp vs pfx difference, logprobs20). Close task 21.
2. Start task 26 (PCIe KV-vs-EP): `scripts/dp_ep/pcie_hog.py` is DRAFTED (untested; also copied to lab-scripts/) — a torch process on cuda:0
   doing pinned H2D / D2H copies of 4k/8k/16k-token KV equivalents — Qwen1.5-MoE KV ≈ 192 KB/token
   (24 layers × 2 × 16 heads × 128 × bf16) → 0.75/1.5/3 GB bursts — continuous vs 50 % duty vs
   rate-capped) and `scripts/dp_ep/sbatch_c26.sh` (copy of sbatch_c21.sh: server up, then plain
   decode windows B=8/32 on both ranks with the hog off/on; metric = rank 1 ITL p50/p95 and step
   period from `trace/step.jsonl.dp1`; the hog runs inside the same Slurm allocation on GPU 0).
   Gate: rank-1 p95 TPOT +10 % strong. Prior-art search first (login-node curl): vLLM issues
   "PCIe contention KV offload", "OffloadingConnector latency", LMCache/Mooncake PCIe interference.
3. Then task 24 prior-art check (dynamic/adaptive chunk size in vLLM issues+PRs, SGLang, Sarathi-Serve).

## Files/artifacts created (paths)
- `benchmarks/results/dp-ep-g1-padding/README.md` (rounds 2–3 + verdict), `raw/g1-analysis/{g1-oracle-q128-graph-1602472-vs-eager-1602473.json,g1-oracle-q128.md,g1-oracle-q512cap512-graph-1602484-vs-eager-1602456.json,g1-oracle-q512cap512.md}`
- `benchmarks/results/dp-ep-c21-contagion/README.md`, `raw/c21-round1-1602481.{json,md}`
- `scripts/dp_ep/measure_contagion.py` (non-streaming prompt train; pen/topk/temp/logprobs20), `scripts/dp_ep/analyze_contagion.py` (committed)
- `scripts/dp_ep/pcie_hog.py` (task 26 PCIe injector draft, untested — smoke-test it on a compute node first: `python pcie_hog.py --bytes 805306368 --duration 5 --log /tmp/scxi253/hog.jsonl`)
- Cluster: `results/g1-analysis/`, `results/c21-analysis/` under `/data/run01/scxi253/inference/`.
- `docs/multigpu-opportunity-ledger.md` updated (G1 killed, 21 downgraded, 24 queued).

## Risks / unresolved methodological issues
- ITL p95 ≈ 2×p50 in plain temperature-0 streams (bimodal token delivery, likely detokenizer
  chunks yielding no text on some steps) — p50 and the step trace are the robust signals; p95
  ratios across features with different token diversity are not comparable.
- pfx B=8 cells bimodal across repeats (16 vs 30 ms plain-rank ITL with identical step periods)
  — unexplained; do not quote the pooled pfx median without the spread.
- Contagion harness client runs on the compute node in one asyncio process; at B=32×2 streams
  plus a prompt train it may itself add jitter — check `waves.log` errors (0 so far).
- EP tracer hooks do not run inside captured graphs: graph-mode runs have step/iter traces only.
- The G1 kill is platform-specific in one respect: the 44–50 ms eager floor is a CPU-launch-bound
  property of this 24-layer/60-expert model on this host; the wave-bound argument (padded rank ≤
  heavy rank) is general for R=2.
- Home quota (1 GB) is full: every cache is redirected in the sbatch scripts — keep it that way.
