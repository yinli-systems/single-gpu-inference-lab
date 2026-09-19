# Multi-GPU autonomy run — standing instructions

You are running headless on the ParaCloud login node inside a watchdog loop (fresh session every
~50 minutes; this file plus `AUTONOMY_STATE.md` are your only memory). Your job for the whole run:

> **Find one multi-GPU mechanism where the current runtime provably does unnecessary work, show an
> oracle gain >10%, then build the smallest fix.** Not "as many optimizations as possible".

Method (the lab's protocol, unchanged): measurement contract → oracle / upper bound with coverage
labels → pre-registered kill gate → minimal env-gated implementation → interleaved live A/B → record
positive, negative and superseded results with equal care. Every number links to raw data with the
exact command and git SHA.

## Environment facts (verified 2026-09-19)

- Login node: internet OK (slow: ~100 KB/s mirrors, ~0.8 MB/s scp inbound), Slurm submit host,
  no GPU. **Compute nodes have no internet.** Home has a 1 GB quota (full): never write under `$HOME`.
- Workspace `W=/data/run01/scxi253/inference` (3.8 TB free): `models/` (Qwen1.5-MoE-A2.7B-Chat 27 GB,
  Qwen3-30B-A3B-FP8 31 GB), `venv-vllm.tar` (vLLM 0.29.0 + torch 2.13 cu130 + tracers, built at
  `/tmp/scxi253/venv-vllm`, untarred to node-local disk by the job script), `libfix.tar`
  (libstdc++ shim), `results/`, `logs/`, `lab-scripts/`.
- This repo checkout: `/data/run01/scxi253/multigpu-autonomy/single-gpu-inference-lab`, branch
  `dp-ep-waves`. Commit locally; **never push, never open/comment on GitHub, never sign DCO.**
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
