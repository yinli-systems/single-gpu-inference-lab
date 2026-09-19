# Autonomous Research State

## Timestamp
2026-09-19 05:4x (Mac, UTC+4) = 09:4x cluster clock (UTC+8). Written by autonomous round 56.
(Round 55's timestamp was ~30 min ahead of the real clock; trust `date` on the login node.)

## Git (branch, SHA, dirty files)
branch `dp-ep-waves`; commits this round: `5694920` (task 26 harness), `7972f8e` (c26 README, ledger 24/26),
`529fc46` (task 27 draft, ledger C1/29 + 22/23), then the final state/ledger commit (see `git log -1`).
Working tree clean after the final commit. Cluster copies in `/data/run01/scxi253/inference/lab-scripts/`:
pcie_hog.py, measure_pcie.py, analyze_pcie.py, sbatch_c26.sh, measure_spec.py, sbatch_c27.sh (md5 = repo files at 529fc46).

## Running Slurm jobs (id, purpose, expected output path)
- **1602508** `c26-dpep` (node wqd10nba06g6, started 09:16 cluster): task 26 PCIe arbitration —
  `sbatch lab-scripts/sbatch_c26.sh Qwen1.5-MoE-A2.7B-Chat graph multiport 512`; 9 hog specs × B{8,32} × 3 repeats
  = 54 windows ≈ 15 min after the server is up (server started 09:20). Output
  `/data/run01/scxi253/inference/results/c26-1602508-Qwen1.5-MoE-A2.7B-Chat-graph-multiport-q512/{pcie.json,waves.log,hog/*.jsonl,hog-standalone.jsonl,dmon.log,trace/step.jsonl.dp0,dp1}`,
  log `logs/c26-1602508.out` (ends with `C26_DONE`). SEE "Next exact action" — analyze it first.
- 1602499 `c21-dpep` (task 21 round 2): FINISHED_OR_NOT — see below.
- Other jobs in `squeue -u $USER` (mbert149-glue, vft-smoke-lora, moe2b-ds-*, fork1b, moe7b, f3b-prof) are the user's
  own unrelated work — never touch them.

## Completed this round
1. Task 26 built and launched: `scripts/dp_ep/pcie_hog.py` (start-file-gated bursts, bounded 512 MiB device
   window, `--rate-gbs` cap, `d2d`/`idle` controls, per-burst monotonic log), `measure_pcie.py` (hog specs
   off/idle/h2d/d2h/both/h2d:0.5/h2d:0.25/h2d:cap4/d2d × B), `analyze_pcie.py` (rank-1 ITL/step vs `off`,
   in-burst vs pause split), `sbatch_c26.sh` (standalone link bandwidth + `nvidia-smi dmon`). Job 1602508.
   Smoke test passed on the node: standalone h2d 23.1 GB/s, d2h 24.6 GB/s (PCIe 4.0 x16), d2d 454 GB/s.
2. Duplicate-work checks (login-node curl, GitHub API + arXiv API over https):
   - task 26: no report/implementation of KV-movement vs collective PCIe arbitration (adjacent #57555, #44294, #43470, #42212).
   - task 24: **downgraded** — vLLM 0.29.0 ships `--prefill-schedule-interval` (DP-aligned prefill cadence,
     `DPEngineCoreProc._should_throttle_prefills`, shared step_counter; PR #54627 extends to DP=1) and RFC #52906
     (P-PAS adaptive prefill budget, arXiv 2608.15171, code public). Remaining delta (peer-load-aware budget) is incremental.
   - tasks 22/23: only correctness work upstream (#53578, #43547); 22 is latency-neutral by the G1 argument
     (dummy rank's step ≤ heavy rank's step); 23's remedy is routing (BalanceRoute). Deprioritised.
   - task 28: analytic kill at EP2 — top-4 of 60 experts: P(all 4 on one rank) = 2·C(30,4)/C(60,4) = 0.112, so
     sparse dispatch saves ≤ 11 % of allgather bytes < the 1.3× phase-A gate (uniform routing assumed).
   - C1/29 first estimate from existing traces: step period = step CUDA time at B=8 and B=32 (both ranks) →
     control plane fully overlapped, visible cost < 1 % → provisionally killed (only a B=1 check could revive it).
3. Task 21 round 2 preview (2/3 repeats) → final numbers below if the job finished before this file was written.
4. Task 27 drafted (not submitted): `measure_spec.py` + `sbatch_c27.sh` (n-gram speculation, rep/rnd prompt kinds).

## Measured results (actual numbers only)
Task 21 round 2 (job 1602499, partial = repeats 0–1 unless marked final; plain-rank ITL p50 ratio vs plain|plain, F on r0 → r1 / F on r1 → r0):
- B=8: pen 1.18 / 1.18 (step period 12.6 → 15.0 ms both ranks); topk 1.17 / 1.12 (→ 14.4–14.9); temp 1.10 / 1.11 (→ 14.0–14.2);
  logprobs20 1.14 / 1.08; sampling 1.27 / 1.81 (the 1.81 cell has period 15.9 = same as sampling|plain → bimodal token delivery, not a longer step);
  plogp 1.29 / 1.20 (p95 ×2.2–2.3); pfx 1.21 / 1.22 (p95 ×2.0).
- B=32: pen 1.01 / 1.04; topk 1.06 / 1.05; temp 1.02 / 1.02; logprobs20 1.00 / 1.00; sampling 1.09 / 1.08; plogp 1.45 / 1.24; pfx 1.46 / 1.26.
- plogp ≈ pfx at both B → prompt logprobs add nothing beyond the prefill train itself.
- Confound: every decomposition cell (temp/topk/pen) carries `seed: 7` (per-request generator path); `temp` alone = +1.4–1.6 ms/step at B=8.
- plain|plain step period vs CUDA time: B=8 12.1/12.0, 13.1/13.2, 14.1/14.1 ms; B=32 21.4/21.4, 21.3/21.2, 21.6/21.6 ms.
Task 26: standalone GPU-0 link bandwidth h2d 23.1 GB/s, d2h 24.6 GB/s; d2d 454 GB/s. Cells pending.

## Alive hypotheses (evidence + gate)
- Task 26 (PCIe KV movement vs EP collectives): running (1602508); gate rank-1 p95 TPOT +10 % strong, <5 % kill.
  Mechanistic prior: no P2P on 4090 → NCCL EP tensors cross host memory over the same links the hog saturates.
- Task 27 (spec-decode acceptance skew, n-gram): harness drafted, untested; gate <5 % kill, >10 % continue.
- P1 (FP8 EP transport bound), 30 (barrier-free expert service replay): untouched; 30 needs eager EP traces (available from M1/G1 eager runs).

## Killed hypotheses (reason + evidence)
- G1–G4 (padding amplification / graph-vs-eager admission): oracle 0.0 % in 45/45 cells (round 55).
- Task 21 logprobs / struct: ≤2 %; sampling bundle: real +9–18 % at B=8 but = the feature's own per-step cost exported 1:1
  by lockstep, mechanism upstream (PR #47540); round 2 shows every sampler component (temp/topk/pen) exports +10–18 % at B=8 and ≤6 % at B=32.
- Task 28 (sparse PCIe EP dispatcher): analytic, ≤11 % byte saving at EP2 (< 1.3× gate).
- C1/29 (DP control-plane tax): provisionally, visible cost < 1 % at B ≥ 8 (period = CUDA time).
- Task 24: downgraded to engineering-only (prefill_schedule_interval + P-PAS exist upstream).

## Blocked hypotheses (specific blocker)
- None hard-blocked. WebSearch/curl from the Mac are not permitted in the headless session; GitHub/arXiv searches
  run on the login node (`pc.sh 'curl -s https://api.github.com/...'`, `https://export.arxiv.org/api/query?...`).

## Current strongest result (bounded wording)
Still negative-leaning: on 2×4090 DP2/EP2 (vLLM 0.29.0, Qwen1.5-MoE-A2.7B) lockstep exports each rank's per-step
cost 1:1 to the peer (sampler components +10–18 % at B=8, prefill chunks ×1.2–1.5 at B=32, prompt logprobs = the
prefill train), padding is user-invisible, and the DP control plane is fully overlapped at B ≥ 8. Task 26 (PCIe
arbitration) is the open measurement; its standalone link numbers are in.

## Next exact action (the exact first command/file/experiment for the next session)
1. `scripts/dp_ep/pc.sh 'squeue -u $USER -h -o "%i %j %t %M"; tail -3 /data/run01/scxi253/inference/logs/c26-1602508.out'` —
   when `C26_DONE`: `scripts/dp_ep/pc.sh 'cd /data/run01/scxi253/inference; mkdir -p results/c26-analysis; LD_LIBRARY_PATH=/tmp/scxi253/libfix /tmp/scxi253/venv-vllm/bin/python lab-scripts/analyze_pcie.py --dir results/c26-1602508-Qwen1.5-MoE-A2.7B-Chat-graph-multiport-q512 --output results/c26-analysis/c26-1602508.json > results/c26-analysis/c26-1602508.md 2>&1; sed -n "/### rank-1/,\$p" results/c26-analysis/c26-1602508.md'`,
   then `scripts/dp_ep/pcget.sh /data/run01/scxi253/inference/results/c26-analysis/ benchmarks/results/dp-ep-c26-pcie/raw/` and
   `scripts/dp_ep/pcget.sh /data/run01/scxi253/inference/results/c26-1602508-Qwen1.5-MoE-A2.7B-Chat-graph-multiport-q512/dmon.log benchmarks/results/dp-ep-c26-pcie/raw/`;
   fill the README "Results" section; apply the gate (rank-1 p95 +10 % strong / <5 % kill; use `d2d` and `idle` to attribute).
   If the hog cells show `rc=` errors or 0 bursts, read `hog/*.jsonl.out` first.
   If alive (>10 %): next = staggered/rate-capped policy oracle (hog `--rate-gbs` sweep 2/4/8/16 GB/s, and `h2d:0.5:0.75`) + PCIe
   counters from dmon; if killed: record and go to 2.
2. Task 27: `scripts/dp_ep/pc.sh 'cd /data/run01/scxi253/inference && sbatch lab-scripts/sbatch_c27.sh Qwen1.5-MoE-A2.7B-Chat graph multiport 512 "--speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":4,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}"'`
   (check `server.log` for the spec config being accepted under DP+EP+graphs; the harness is untested — watch the first `[r0]` line for
   accept rates: rep ≈ 1.0, rnd ≈ 0). Prior-art search first: vLLM/SGLang "adaptive speculative length data parallel",
   "num_speculative_tokens dynamic batch", DSpark adaptive verification (#52362/#54065), SmartSpec/AdaServe/SpecServe papers.
3. Task 21: if `C21_DONE` was not reached this round, run the round-2 analyzer command from round 55's state and add "Round 2" to
   `benchmarks/results/dp-ep-c21-contagion/README.md` (numbers above are from 2/3 repeats).

## Files/artifacts created (paths)
- `scripts/dp_ep/{pcie_hog.py (rewritten), measure_pcie.py, analyze_pcie.py, sbatch_c26.sh, measure_spec.py, sbatch_c27.sh}`
- `benchmarks/results/dp-ep-c26-pcie/README.md` (design, prior art, caveats; results pending)
- `docs/multigpu-opportunity-ledger.md` (24 downgraded, 26 running, C1/29 estimate, 22/23 prior art, 28 analytic kill)
- Cluster: `results/c26-1602508-*/`, `results/c21-1602499-*/`, `results/c21-analysis/` (round-2 analysis if run).

## Risks / unresolved methodological issues
- Task 26 hog is a separate process on GPU 0: context sharing on GPU 0 can slow rank 0 itself (and rank 1 via lockstep) without
  any PCIe mechanism — only rank-1 effects with h2d/d2h > d2d (and > idle) count as PCIe arbitration.
- `--gpu-memory-utilization 0.85` in c26 (vs 0.88 elsewhere): KV capacity ≈ 26k tokens/rank; B=32 × ~600 tokens fits.
- ITL p95 ≈ 2×p50 in temperature-0 streams (bimodal token delivery); p50 + step trace are the robust signals.
- Task 21 decomposition cells all carry a per-request `seed`, which is itself a slower sampler path — the temp/topk/pen split is not clean.
- Task 27 harness untested; n-gram speculation under DP2/EP2 + CUDA graphs in 0.29.0 not yet verified on this platform.
- Home quota (1 GB) is full: every cache is redirected in the sbatch scripts — keep it that way.
