# Autonomous Research State

## Timestamp
2026-09-19 05:43 (Mac, UTC+4) = 09:43 cluster clock (UTC+8). Written by autonomous round 56.
(Round 55's timestamp was ~30 min ahead of the real clock; trust `date` on the login node.)

## Git (branch, SHA, dirty files)
branch `dp-ep-waves`; commits this round: `5694920` (task 26 harness), `7972f8e` (c26 README, ledger 24/26),
`529fc46` (task 27 draft, ledger C1/29 + 22/23), `9ac50af` (task 21 closed), `af00acd` (task 26 result), then the final
state commit (see `git log -1`). Working tree clean after it. Cluster copies in `/data/run01/scxi253/inference/lab-scripts/`:
pcie_hog.py, measure_pcie.py, analyze_pcie.py, sbatch_c26.sh, measure_spec.py, sbatch_c27.sh (md5 = repo files at 529fc46).

## Running Slurm jobs (id, purpose, expected output path)
- None of ours. 1602499 (`c21-dpep`, task 21 round 2) and 1602508 (`c26-dpep`, task 26) both COMPLETED and are analyzed.
- Other jobs in `squeue -u $USER` (mbert149-glue, vft-smoke-lora, moe2b-ds-*, fork1b, moe7b, f3b-prof) are the user's
  own unrelated work — never touch them.

## Completed this round
1. **Task 26 built, run and analyzed** (`scripts/dp_ep/{pcie_hog.py, measure_pcie.py, analyze_pcie.py, sbatch_c26.sh}`,
   job 1602508, 54 windows; `benchmarks/results/dp-ep-c26-pcie/README.md` + `raw/c26-1602508.{json,md}`, `raw/hog-standalone.jsonl`).
   Verdict: ALIVE / STRONG for GPU→host traffic (numbers below).
2. **Task 21 round 2 analyzed and task 21 CLOSED** (job 1602499; `benchmarks/results/dp-ep-c21-contagion/README.md` "Round 2",
   `raw/c21-round2-1602499.{json,md}`).
3. Duplicate-work checks (login-node curl; GitHub API + arXiv API over https): task 26 clear (adjacent #57555/#44294/#43470/#42212;
   arXiv SwiftCache/DualPath/semi-PD not the mechanism); task 24 **downgraded** (vLLM 0.29.0 `--prefill-schedule-interval` =
   DP-aligned prefill cadence, verified in `v1/engine/core.py`; RFC #52906 P-PAS adaptive budget, arXiv 2608.15171); tasks 22/23
   deprioritised (only correctness work upstream; 22 latency-neutral by the G1 argument; 23 = routing prior art); task 28 analytic
   kill at EP2 (top-4/60 experts: ≤11 % byte saving < 1.3× gate); task 27 prior art (single-engine adaptive spec length exists:
   #35301, `num_speculative_tokens_per_batch_size`, DSpark; nothing cross-rank).
4. C1/29 first estimate from existing traces: step period = step CUDA time at B=8/32 → control plane fully overlapped, <1 % visible.
5. Task 27 drafted (`measure_spec.py`, `sbatch_c27.sh`; pushed to lab-scripts; untested, not submitted).

## Measured results (actual numbers only)
Task 26 (job 1602508; pooled medians of 3 repeats; ratio vs `off` at same B; r0 and r1 identical in every cell):
- standalone GPU-0 link: h2d 23.1 GB/s, d2h 24.6 GB/s, d2d 454 GB/s.
- idle context: r1 ITL p50 ×1.00 / ×1.00 (B=8 / 32); p95 ×1.11 (bimodal baseline noise) / ×1.01.
- h2d continuous 20.7 / 21.8 GB/s: p50 ×1.09 / ×1.07; p95 ×1.20 / ×1.08; step period 14.1→15.0 / 21.6→23.1 ms.
- **d2h continuous 17.5 / 14.8 GB/s (slowed from 24.6 by the server): p50 ×1.20 / ×1.47; p95 ×1.28 / ×1.48; period 14.1→17.0 / 21.6→32.1 ms.**
- both alternating: p50 ×1.13 / ×1.37; p95 ×1.26 / ×1.42.
- h2d 50 % duty: ×1.05 / ×1.04 (in-burst CUDA 14.9 vs pause 14.7 ms); h2d 25 %: ×1.00 / ×1.02; h2d capped 4 GB/s: ×1.00 / ×1.00.
- d2d control 206 GB/s GPU-internal: ×1.96 / ×1.88 (GDDR-bandwidth contention on GPU 0, exported by lockstep).
Task 21 round 2 (job 1602499; plain-rank ITL p50 ratio, F on r0→r1 / F on r1→r0; B=8 | B=32):
pen 1.13/1.12 | 1.05/1.04; topk 1.10/1.07 | 1.07/1.05; temp 1.07/1.09 | 1.02/1.02; sampling 1.21/1.20 | 1.09/1.08;
logprobs20 1.07/1.04 | 1.00/0.99; pfx 1.18/1.20 | 1.58/1.29; plogp 1.26/1.15 | 1.28/1.26 (plogp and pfx raise the step period identically).
plain|plain step period vs CUDA: B=8 12.1/12.0, 13.1/13.2, 14.1/14.1; B=32 21.4/21.4, 21.3/21.2, 21.6/21.6 ms.

## Alive hypotheses (evidence + gate)
- **Task 26 — STRONG**: GPU0→host DMA on rank 0 raises rank-1 p95 ITL ×1.28–1.48 (gate +10 %); rate cap 4 GB/s removes it.
  Open sub-questions before any policy claim: (a) d2h duty/cap sweep (harmless rate), (b) NCCL transport confirmation
  (`NCCL_DEBUG=INFO` → "via SHM"?), (c) a *real* mover (vLLM OffloadingConnector CPU offload on rank 0) instead of the hog,
  (d) why d2h hurts 3–5× more than h2d, (e) 3 interleaved repeats already done for the headline (per-repeat r1 p50 d2h B=32: 32.3, 32.9, 29.0).
- Task 27 (spec-decode acceptance skew): drafted; gate <5 % kill, >10 % continue.
- P1 (FP8 EP transport bound), 30 (barrier-free expert service replay): untouched.

## Killed hypotheses (reason + evidence)
- G1–G4: oracle 0.0 % in 45/45 cells (round 55).
- Task 21 (closed): contagion = feature's own per-step cost exported 1:1; each feature ≤ ×1.13 at B=8, ≤ ×1.07 at B=32; penalties known upstream (#47540).
- Task 28: analytic ≤11 % byte saving at EP2 (< 1.3× gate).
- C1/29: provisionally, <1 % visible at B ≥ 8 (period = CUDA time).
- Task 24: downgraded to engineering-only (prefill_schedule_interval + P-PAS upstream). Tasks 22/23 deprioritised.

## Blocked hypotheses (specific blocker)
- None. WebSearch/curl from the Mac are not permitted in the headless session; run searches on the login node
  (`pc.sh 'curl -s https://api.github.com/...'`, `https://export.arxiv.org/api/query?...`).

## Current strongest result (bounded wording)
On 2×RTX 4090 DP2/EP2 (vLLM 0.29.0, Qwen1.5-MoE-A2.7B, allgather_reducescatter EP, no P2P), a bulk GPU→host transfer stream
injected on rank 0's GPU (14.8–17.5 GB/s, a separate process) raises the decode ITL of rank 1 — which moves no data — by ×1.20
(B=8) to ×1.47 (B=32) at p50 and ×1.28–1.48 at p95, with an idle context costing nothing and host→GPU traffic at a higher rate
costing only 7–9 %; capping the stream at 4 GB/s removes the effect. This is a measured cross-rank performance-isolation failure
through the PCIe link shared by KV movement and EP collectives, not yet reproduced with a real KV connector.

## Next exact action (the exact first command/file/experiment for the next session)
1. Task 26 round 2 (policy oracle + mechanism), one job:
   `scripts/dp_ep/pc.sh 'cd /data/run01/scxi253/inference && sbatch lab-scripts/sbatch_c26.sh Qwen1.5-MoE-A2.7B-Chat graph multiport 512 "" "--specs off,d2h:1.0,d2h:0.5,d2h:0.25,d2h:cap2,d2h:cap4,d2h:cap8,d2h:cap12,d2h:1.0:0.75,d2h:0.5:0.75,both:cap8 --batch-sizes 8,32 --repeats 3 --seed 65"'`
   (harness accepts `<dir>:<duty|capN>:<GiB>`; analysis: `lab-scripts/analyze_pcie.py --dir results/c26-<job>-... --output results/c26-analysis/c26-<job>.json`).
   Before submitting, add `export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` to a copy of sbatch_c26.sh (or in EXTRA env) so
   `server.log` shows the transport ("via SHM" vs "via P2P/direct pointer") — that decides the mechanism claim.
2. Then the real-mover check: vLLM OffloadingConnector (`--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks":...}}'`)
   on rank 0 only is not possible (config is engine-wide) — instead drive rank 0 with a KV-evicting long-prompt train (prefix reuse +
   offload) and measure rank 1's ITL; check `vllm/v1/kv_offload/` in 0.29.0 for the copy path (side stream? cudaMemcpyBatchAsync?).
3. Only if 1–2 hold: minimal policy = rate-limited/deferred KV movement (connector-side), env-gated; live A/B with 3 interleaved repeats.
4. Parallel CPU-side: update the ledger's task 26 row with the round-2 numbers; arXiv search "PCIe" + "KV cache offloading" + "interference" (done: none).
5. If 26 dies at the real-mover step: submit task 27 (`sbatch lab-scripts/sbatch_c27.sh Qwen1.5-MoE-A2.7B-Chat graph multiport 512 "--speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":4,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}"`;
   harness untested — watch the first `[r0]` line for accept rates rep ≈ 1.0, rnd ≈ 0).

## Files/artifacts created (paths)
- `scripts/dp_ep/{pcie_hog.py (rewritten), measure_pcie.py, analyze_pcie.py, sbatch_c26.sh, measure_spec.py, sbatch_c27.sh}`
- `benchmarks/results/dp-ep-c26-pcie/README.md` (design, prior art, results, caveats), `raw/c26-1602508.{json,md}`, `raw/hog-standalone.jsonl`
- `benchmarks/results/dp-ep-c21-contagion/README.md` ("Round 2" + closure), `raw/c21-round2-1602499.{json,md}`
- `docs/multigpu-opportunity-ledger.md` (21 closed, 24 downgraded, 26 alive/strong, 27 drafted, 28 killed, 22/23 deprioritised, C1/29 estimate)
- Cluster: `results/c26-1602508-*/` (pcie.json, hog/*.jsonl, dmon.log, traces), `results/c26-analysis/`, `results/c21-analysis/`.

## Risks / unresolved methodological issues
- Task 26 uses a synthetic hog (separate process, copy engine DMA): a real connector copies from inside the worker (maybe on a side
  stream with `cudaMemcpyBatchAsync`), so the magnitude with real KV offload is unmeasured; the 8 s continuous stream is an upper
  bound — realistic bursts (1.5 GiB ≈ 65–100 ms) hit ~5 steps; the h2d duty cells show +0.2 ms/step in-burst, d2h duty cells not run.
- Direction asymmetry (d2h ≫ h2d) is unexplained; the SHM-transport hypothesis needs `NCCL_DEBUG=INFO`. `dmon.log` (per-GPU PCIe rx/tx)
  was recorded but not yet analyzed.
- `d2d` is a GDDR-bandwidth contention control (206 GB/s), not a copy-engine-only control; the "PCIe" attribution rests on idle = 1.00
  and on the PCIe hogs' negligible GDDR traffic (≤22 GB/s).
- `--gpu-memory-utilization 0.85` in c26 (vs 0.88 elsewhere): baseline ITLs match c21 (14.2 vs 13.3–14.4; 21.9 vs 21.8).
- ITL p95 ≈ 2×p50 in temperature-0 streams (bimodal token delivery); p50 + step trace are the robust signals; p95 quoted because the gate names it.
- Task 21 decomposition cells all carry a per-request `seed` (slower sampler path) — the temp/topk/pen split is not clean.
- Task 27 harness untested; n-gram speculation under DP2/EP2 + CUDA graphs in 0.29.0 not verified on this platform.
- Home quota (1 GB) is full: every cache is redirected in the sbatch scripts — keep it that way.
