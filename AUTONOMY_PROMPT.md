# Multi-GPU autonomy run — standing instructions

You are running headless **on the user's Mac** inside a watchdog loop (fresh session every ~50
minutes; this file plus `AUTONOMY_STATE.md` are your only memory). You drive ParaCloud over SSH —
the cluster has no Claude CLI and cannot reach Anthropic endpoints. Your job for the whole run:

> **Find one multi-GPU mechanism where the current runtime provably does unnecessary work, show an
> oracle gain >10%, then build the smallest fix.** Not "as many optimizations as possible".

Method (the lab's protocol, unchanged): measurement contract → oracle / upper bound with coverage
labels → pre-registered kill gate → minimal env-gated implementation → interleaved live A/B → record
positive, negative and superseded results with equal care. Every number links to raw data with the
exact command and git SHA.

## How to reach the cluster (every remote command)

Run remote work through the helper `scripts/dp_ep/pc.sh`, which wraps
`ssh -b <local-ip> paracloud` (the `-b` bind is required by this network):

```bash
scripts/dp_ep/pc.sh 'squeue -u $USER -h -o "%i %j %t %M"'
scripts/dp_ep/pc.sh 'cd /data/run01/scxi253/inference && sbatch lab-scripts/sbatch_m1.sh Qwen1.5-MoE-A2.7B-Chat eager multiport 512'
scripts/dp_ep/pcget.sh <remote-path> <local-path>     # rsync pull (use for result JSON, not 40 MB traces)
```

Heavy analysis of large traces should run **remotely** (`pc.sh '/tmp/scxi253/venv-vllm/bin/python ...'`
with `LD_LIBRARY_PATH=/tmp/scxi253/libfix`); pull only summaries into this repo. The login node is
slow and its SSH occasionally resets — retry once, then continue with other work. Local repo:
this checkout; the same branch is **not** cloned on the cluster, so copy any script you need with
`scripts/dp_ep/pcput.sh <local-file> <remote-dir>` (the job template lives at
`/data/run01/scxi253/inference/lab-scripts/`).

## Environment facts (verified 2026-09-19)

- Login node: internet OK (slow: ~100 KB/s mirrors, ~0.8 MB/s scp inbound), Slurm submit host,
  no GPU. **Compute nodes have no internet.** Home has a 1 GB quota (full): never write under `$HOME`.
- Workspace `W=/data/run01/scxi253/inference` (3.8 TB free): `models/` (Qwen1.5-MoE-A2.7B-Chat 27 GB,
  Qwen3-30B-A3B-FP8 31 GB), `venv-vllm.tar` (vLLM 0.29.0 + torch 2.13 cu130 + tracers, built at
  `/tmp/scxi253/venv-vllm`, untarred to node-local disk by the job script), `libfix.tar`
  (libstdc++ shim), `results/`, `logs/`, `lab-scripts/`.
- Repo lives on the Mac (branch `dp-ep-waves`). Commit locally; **never push, never open/comment on
  GitHub, never sign DCO.** Scripts are pushed to the cluster with `pcput.sh`.
- Slurm: `--partition gpu_4090 --qos gpugpu --gres gpu:2` (RTX 4090 24 GB, driver 580.82, PCIe, no
  NVLink), `--cpus-per-task 12`, **no `--mem`**. Jobs: `scripts/dp_ep/sbatch_m1.sh <model> <eager|graph>
  <internal|multiport> <q>`; outputs `$W/results/m1-<jobid>-*/` with `server.log`, `waves.json`,
  `trace/{iter.jsonl.dp0,dp1, step.jsonl.dp0,dp1, ep.0.jsonl, ep.1.jsonl}`. Copy the script for new
  campaigns rather than editing it in place.
- vLLM 0.29.0 DP2/EP2 works on this setup: `--data-parallel-size 2 --data-parallel-size-local 2
  --enable-expert-parallel --all2all-backend allgather_reducescatter`; multi-port external LB
  (`--data-parallel-multi-port-external-lb --api-server-count 1`, rank r on port 8300+r) gives
  deterministic per-rank placement. Startup ~4 min. GPU KV cache 29.5k tokens/rank at
  `--max-model-len 16384 --gpu-memory-utilization 0.88` on Qwen1.5-MoE.
- Tracers (installed in the venv): engine iteration trace (per-request chunks/KV, rank-suffixed),
  runner step trace (`num_tokens`, `padded_tokens`, `cg_mode`, `cuda_ms`, rank-suffixed), EP
  collective trace (`ep.<rank>.jsonl`: `seq`, `op`, `t_host_ns` monotonic host clock — the same
  timebase across ranks on one node —, `sizes` = tokens each DP rank brought to the wave,
  `cuda_ms` = this rank's collective duration incl. waiting for the peer). Under CUDA graphs the
  EP tracer's Python hooks do not run inside captured graphs — use it only in eager runs.
  Never subtract CUDA event times across GPUs.
- Analysis: `scripts/dp_ep/analyze_dp_ep_waves.py --dir <results dir>`; harness
  `scripts/dp_ep/measure_dp_ep_waves.py` (cells dd/pp/pd, per-rank decode streams with token
  arrival times, prefill TTFT). Python for analysis: `/tmp/scxi253/venv-vllm/bin/python` with
  `LD_LIBRARY_PATH=/tmp/scxi253/libfix` on the login node (numpy available), or system python3.

## NOVELTY OVERRIDE

Do not assume the previously proposed DP/EP phase-aware router is novel. BalanceRoute
(arXiv:2605.06113) already studies online routing for barrier-synchronized DP LLM serving. Keep it
only as line R1 (comparison / fallback).

The flagship hypothesis is the runtime-level synchronized-shape tax in current vLLM, verified in
`vllm/v1/worker/dp_utils.py` of the installed 0.29.0:

1. DP ranks synchronize `cudagraph_mode` by taking the minimum across ranks.
2. `should_dp_pad = synced_cudagraph_mode != 0 or should_ubatch`; when true, **all DP ranks are
   padded to the maximum token count across ranks** (`_post_process_dp_padding`).
3. Therefore a heavy rank amplifies work on otherwise light ranks even with fixed request
   assignments; `--enforce-eager` is natively the unpadded true-shape arm.

Define `A_pad = R · max_r(n_r) / Σ_r n_r`.

### Campaign G1 — cross-rank padding amplification, oracle
Controlled per-rank token pairs via multi-port placement: rank 0 prefill chunk q ∈ {32, 64, 128,
256, 512} (chunk set by `--max-num-batched-tokens`; prefix 4k/8k so the chunk has KV depth) while
rank 1 runs B ∈ {8, 16, 32} decode streams; plus homogeneous controls (dd, pp). Record per rank:
true tokens, padded tokens, graph mode, step CUDA ms, user-visible decoder ITL and prefill TTFT.
Arms: **A** graph mode (default) = graph + max-DP-padding; **B** `--enforce-eager` = synchronized
eager, true per-rank sizes; **C** hindsight oracle min(A, B) per cell. Report A_pad, T_A/T_B, the
crossover A_pad*, and the decode rank's ITL under each arm.
Gate: oracle <5% → kill; 5–10% engineering-only; >10% alive; >20% strong.

### Campaign G2 — cost boundary (only if G1 alive)
Fit the simplest predictor of when the whole DP group should choose graph+padded vs eager+unpadded
(inputs: per-rank token counts, graph bucket, A_pad). All ranks must still select the same mode. Do
NOT experiment with rank-local mixed graph/eager modes unless collective correctness is first
formally established.

### Campaign G3/G4 — env-gated padding-aware graph admission + live A/B (only if G2 predicts >10%)

### Line P1 — generic PCIe low-precision EP transport, upper bound only
Measure the fraction of the EP step spent in `allgather_reducescatter` bytes (from the EP trace and
sizes × hidden size × dtype); compute the best case if communication bytes halved (FP8 activations)
minus a measured quant/dequant kernel cost. No implementation unless the bound >15%.

### Line C1 — DP control-plane synchronization cost
Measure the per-step `coordinate_batch_across_dp` all-reduce cost (Gloo CPU group under async
scheduling) at small decode batches; report ms per step and its share of the step.

## Duplicate-work rule
Before claiming any novelty, search vLLM issues+PRs, SGLang, TensorRT-LLM, Dynamo, arXiv/MLSys/
OSDI/SOSP/EuroSys/ASPLOS/NSDI for: DP padding, rank padding, CUDA graph padding, heterogeneous DP,
graph admission, execution-shape amplification, synchronized MoE, barrier-shaped batching,
padding-aware serving. If an existing system implements the same graph-vs-padding controller,
downgrade immediately, record it in the ledger, and move to the next line. "Nobody has done this"
is never claimable; "no implementation found as of <date> in <sources>" is.

## Session discipline
- Start every session by reading `AUTONOMY_STATE.md`, `docs/multigpu-opportunity-ledger.md`,
  `git status`/`git log -5`, `squeue -u $USER`, and the newest results dir.
- Never idle-wait on a GPU job: analyze finished data, write the next harness, do the duplicate
  search, or advance another line. Poll Slurm at most every 2 minutes.
- Keep every artifact under `benchmarks/results/<name>/` with README, raw JSON/JSONL, exact commands.
- Before the session ends (budget ~50 min; stop by 45), atomically rewrite `AUTONOMY_STATE.md`:
  completed work, exact git SHA, Slurm job ids and states, result paths, measured numbers,
  alive/killed/blocked decision per line, the exact next command, unresolved risks.
- Never publish GitHub content, push, merge, force-push, sign DCO, or post comments.

---

# FRONTIER EXTENSION — TEN ADDITIONAL MULTI-GPU RESEARCH TASKS

Not mandatory sequential tasks. Treat as a **ranked opportunity pool**. Before spending >30 minutes
on any task: search current vLLM/SGLang/TensorRT-LLM/Dynamo source, issues, PRs and recent papers;
record the closest prior work; kill immediately if an active implementation already covers the same
mechanism. Prefer an upper-bound experiment before implementation.

**21. Cross-Rank Feature Contagion.** Can an expensive request-local feature on one DP/EP rank slow
unrelated plain-generation requests on another rank because ranks rendezvous at synchronized expert
steps? Features: logprobs, structured output, speculative decoding, sampling-heavy, prompt logprobs.
The question is not whether the feature is expensive but whether client A's feature imposes latency
on unrelated client B on another GPU — a distributed performance-isolation failure. Use
deterministic rank placement; measure the unaffected rank's TPOT, ITL, time-to-next-forward,
arrival time at the first EP collective, full step CUDA time; compare plain/plain vs feature/plain;
swap ranks. Gate: <5% kill, 10% alive, 20% strong, 50% priority upstream problem. Follow-up:
decompose into GPU sampling work / CPU output processing / scheduler delay / graph-mode change /
collective arrival skew.

**22. Dummy-Rank Compute Elimination.** When a DP rank has no local requests but must stay alive for
EP, how much unnecessary model computation does the dummy rank execute? Does the idle rank need the
whole local attention/normalization/sampling path, or only the expert-service portions peers need?
Phase A: rank0 sustained traffic, rank1 idle; measure rank1 CUDA time by operator family
(attention, router, dispatch, local expert GEMM, combine, normalization, sampling, misc); estimate
useful_remote_expert_work / total_dummy_GPU_work. Phase B: remove one obviously local-only component
at a time while preserving identical collective order (skip local sampler / logits / attention for
zero-real-token state / post-processing). Do NOT skip communication or change collective ordering
unless correctness is proven. Gate: <5% kill, 15% alive, 30% strong.

**23. Tail Dummy Amplification.** After initially balanced ranks diverge because request lengths
differ, one rank may spend the tail executing dummy forwards only because the other is alive. Define
dummy_tail_GPU_seconds = Σ_r GPU time after local useful work ended. Workloads: identical initial
request counts, controlled output lengths (128/128, 128/256, 128/512, 128/1024, 256/1024). Oracle:
perfect length cohorting (short with short, long with long), same requests and total tokens. Metrics:
aggregate throughput, per-request TPOT, dummy GPU ms, fraction of total GPU work that is dummy.
Continue if dummy GPU time >10% of total or oracle cohorting improves throughput / p95 TPOT >10%.
Do NOT build a learned length predictor first: test requested max_tokens, historical class, trivial
short/long bins; if two bins recover most of the oracle, stop there.

**24. Group-Aware Chunked-Prefill Shaping.** Each rank can have a different prefill workload but EP
makes ranks advance together. Choose each rank's chunked-prefill budget jointly so one rank does not
create an unnecessarily large synchronized shape for everyone. This is NOT routing: assignments are
fixed; the decision is how much work each already-assigned rank admits next. Estimate per rank
attention work, MoE token work, current decode work; choose q_r maximizing Σ useful_tokens_r /
predicted_wave_time subject to a TPOT/step bound. Oracle: enumerate q ∈ {64,128,256,384,512,768,
1024,2048} on the heavy rank (cheap for DP2). Compare native chunking, fixed conservative chunk,
oracle. Alive at >10% safe progress/throughput; strong if p95 TPOT improves >20% without materially
reducing prefill throughput. Direct multi-GPU continuation of the single-GPU geometry work.

**25. Prefix Locality vs EP Synchronization.** Routing repeated prefixes to the same rank maximizes
cache hits; balancing synchronized work minimizes stragglers. Does maximizing locality create enough
imbalance to lose overall? Several long shared prefixes with skewed popularity; compare round-robin,
least-loaded, prefix-affinity, perfect cache-affinity, perfect synchronized-work oracle, hybrid.
Metrics: prefix-cache hit rate, recomputed prompt tokens, rank work skew, EP wait/step latency,
TTFT, TPOT, throughput. Find the Pareto frontier; start with score = cache_saved_compute − λ ·
predicted_wave_imbalance and sweep λ. Alive if pure affinity loses >10% end-to-end and a hybrid
recovers ≥70% of both benefits.

**26. PCIe Arbitration: KV Movement vs EP Communication.** On PCIe-only GPUs, CPU↔GPU KV transfer and
GPU↔GPU EP communication may contend for the same root complex. Can a KV load on ONE rank degrade
serving latency on OTHERS? DP2/EP2, stable decode on both; inject on rank0 CPU→GPU reload and
GPU→CPU offload at 4k/8k/16k-equivalent KV; measure rank1 (no KV movement). Compare no transfer,
transfer concurrent with EP, transfer staggered between waves, bandwidth-capped transfer. Metrics:
EP collective latency, rank1 TPOT/ITL, H2D/D2H bandwidth, PCIe counters if available, step latency.
Strong if rank0 transfer raises rank1 p95 TPOT >10%. Minimal policy: defer/rate-limit bulk KV
movement during an EP-critical window — only if the oracle exceeds 10%.

**27. Speculative-Decoding Skew Amplification.** Across synchronized ranks, can different acceptance
regimes cause one rank's verification shape to inflate work or delay another? rank0 high-acceptance
(code/repetitive) vs rank1 low-acceptance (prose/random), compared to high/high and low/low, same
spec config. Per rank measure draft tokens, accepted, verification tokens, generated, padded
execution tokens, graph mode, step duration, collective arrival. Questions: does acceptance
heterogeneity change execution shape across ranks; is one rank padded to another's verification
width; does adaptive verification stay locally optimal under synchronous execution; can a rank that
already reduced its budget still pay for another's. Oracle: force matched or group-aware budgets.
<5% kill; >10% continue; >10% for group-aware over independent per-rank control = strong new line.

**28. Portable Sparse PCIe EP Dispatcher.** Can a direct variable-size expert dispatch using ordinary
NCCL/PyTorch collectives beat the generic allgather/reduce-scatter path without DeepEP, NVLink
kernels or datacenter-only features? Engineering line; novelty target is a practical portable
SM89/PCIe vLLM path with serving evidence, not sparse all-to-all itself. Phase A: from real router
outputs compute bytes_AGRS vs bytes_required_by_actual_destinations; kill if <1.3×. Phase B:
`torch.distributed.all_to_all_single` microprototype on exact router assignments. Phase C: serving
path only if the microbenchmark is >15% better. Gates: communication <10% kill; >20% communication
but <5% serving = record Amdahl negative; serving >10% = upstream candidate.

**29. DP Control-Plane Collective Tax.** Every step synchronizes tens of bytes (actual tokens, padded
tokens, ubatch decision, graph mode) through a distributed all-reduce. Phase A: instrument
`coordinate_batch_across_dp`, `_run_ar`, post-processing; measure per-step cost at decode B=1/4/8/16
and DP2/DP3; quantify control_plane_ms / total_step_ms. <2% kill; >5% alive; >10% for low-latency
decode = strong. Alternative prototype (same node only): shared-memory slot per rank, epoch counter,
memory fence, lock-free rendezvous, with identical semantics. Prove no stale epoch reads, rank
failure times out, no hot spin, async scheduling correct, graph mode and token agreement identical.

**30. Barrier-Free Expert Service — Upper Bound.** How much performance would be recoverable if
expert service were asynchronous instead of a global per-layer barrier? Upper-bound task only.
Step 1: trace per rank and layer attention-ready, dispatch-ready, remote expert demand, expert
compute start/end, combine completion, time waiting for peers. Step 2: conservative event-driven
replay preserving dependency ordering, measured compute durations and real bandwidth limits.
Step 3: report current measured step time vs asynchronous replay lower bound. <10% not worth
pursuing; 10–20% interesting; >20% strong; >40% potentially flagship. Step 4 (only if strong): one
isolated MoE layer prototype with P2P send/recv or async work queues. Do NOT redesign vLLM.

## GLOBAL PRIORITY AMONG THESE TEN
21 → 26 → 24 → 23 → 22 → 27 → 25 → 29 → 28 → 30
(ordered by time to falsify, 2–3×4090 feasibility, probability of a visible benchmark, upstream
relevance, research distinctness). Task 30 has the highest ceiling but the largest scope, so it
begins as trace/replay research only. For every task, preserve negative results and move on
immediately when its gate fails.

## RESEARCH PRIORITY (overall)
First: (1) DP/EP cross-rank padding amplification and graph-vs-eager oracle [G1–G4];
(2) cross-rank feature contagion [21]; (3) PCIe KV-vs-EP interference [26]; (4) group-aware chunked
prefill [24]; (5) tail/dummy-rank amplification [23/22]. Then choose among the remaining frontier
tasks by novelty, feasibility and observed effect.

## DECISION GATES (global)
<5% useful end-to-end effect: kill unless it exposes a correctness/reliability defect.
5–10%: engineering-only unless unusually general. >10%: continue. >20%: strong candidate.
At least 3 interleaved repeats for any headline positive result.

## GPU / SLURM SAFETY
Use `/data/run01/scxi253` for durable storage; compute-node `/tmp` only for disposable scratch.
Never fill the home quota. Put timeouts around servers and requests. Kill orphaned server processes
between conditions. Preserve logs before cleanup. Do not let one hung NCCL/vLLM process consume the
sprint: if a job is clearly deadlocked, collect evidence, terminate it and continue.

## GIT / EXTERNAL SAFETY
Local commits and local branches are allowed. Do NOT: push, force push, open or modify upstream
PRs/issues, publish GitHub comments, merge, sign DCO/Signed-off-by, request upstream labels,
trigger upstream CI, or claim independent human review.
