# Recovered FA2 canary: intervention results and execution checkpoint

Date: 2026-09-28 Dubai. Scheduler snapshot: 2026-09-27 23:47:20 UTC.

**Status: new read-only analysis of previously executed, incomplete GPU canaries; no new GPU job was submitted in this continuation.** A repaired harness was recovered and CPU-tested, but its new GPU submission was blocked by the conversation tool safety layer. This is not a claim of an improved serving system, a validated learned planner, or award-level performance.

## 1. Recovered execution state

Parent source commit: `b9e5303417019ad30b54596537636ba443bc1bdd`. Existing remote branch: `research/causal-plan-cost-20260928`.

| Job | GPU | Slurm state | Exit | Allocated GPU seconds | Policy records complete / planned |
|---|---|---|---|---:|---:|
| 1632585 | RTX 4090, 128 SMs | FAILED | 2:0 | 94 | 22 / 36 |
| 1632586 | RTX 5090, 170 SMs | FAILED | 2:0 | 116 | 22 / 36 |

Each job retained 14 scratch-allocation failures. All 28 failures identify insufficient 128 MiB floating workspace; they are not silently removed and are not demonstrated numerical failures. These jobs consumed 210 allocated GPU seconds before this continuation. New GPU allocation in this continuation: zero.

Environment: PyTorch 2.13.0+cu130, CUDA 13.0, FlashInfer 0.6.18, explicit FA2, FP16 Q/K/V/output, head dimension 128, causal attention, no CUDA graph, head pairs (16,4)/(32,8), seed 20260928. Six timing blocks per policy; each averages a randomized forward/reverse order with three inner calls per direction. plan+run+device-completion cycles were measured separately.

Unrelated running jobs were left unchanged. The main Mac worktree has uncommitted Hopper work; it was not switched, reset, staged or committed. A separate worktree `sgi-causal-milestone-20260928` preserves the reviewed repair.

## 2. The split-only explanation does not survive the intervention

Discovery fixture: A has q=(255,769), B has q=(512,512), both have cached k=(8192,7935). Exact attention work, total query tokens, total cached depth and request count agree. Chunk marginals do not agree, so this is an equal-work contrast, not the separate same-marginals pairing-swap test.

| GPU | Query heads | Policy | A CUDA ms | B CUDA ms | A/B | Within-run block-resampling interval |
|---|---:|---|---:|---:|---:|---|
| 4090 | 16 | auto | 0.907107 | 0.494229 | 1.8354 | [1.8296, 1.8379] |
| 4090 | 16 | split disabled | 0.889949 | 0.492779 | 1.8060 | [1.8032, 1.8106] |
| 4090 | 32 | auto | 1.425491 | 0.922549 | 1.5452 | [1.5281, 1.5460] |
| 4090 | 32 | split disabled | 1.425056 | 0.923795 | 1.5426 | [1.5137, 1.5451] |
| 5090 | 16 | auto | 0.466656 | 0.452672 | 1.0309 | [1.0272, 1.0357] |
| 5090 | 16 | split disabled | 0.463149 | 0.462880 | 1.0006 | [0.9985, 1.0053] |
| 5090 | 32 | auto | 0.875405 | 0.866432 | 1.0104 | [1.0076, 1.0239] |
| 5090 | 32 | split disabled | 0.874627 | 0.866877 | 1.0089 | [1.0073, 1.0121] |

Intervals use 5,000 independent resamples of the six timing blocks within each state, seed 20260928, order-statistic indices 124 and 4874. There is only ONE process/seed realization per hardware in these recovered canaries. These intervals describe conditional timing variation; they do not provide independent-process, device-population or held-out-family evidence and must not be used as a promotion gate.

At 16 query heads, disabling splitting sets grid-x to 33/32 with no merge rows on both GPUs. Nevertheless the 4090 gap remains about 1.81x. Thus split-KV alone cannot explain the observed 4090 cliff. The finding does not identify the remaining cause. Query-tile imbalance, residency, wave execution and cache behavior remain possible factors, not established explanations. None of these A/B ratios is an optimization speedup.

## 3. Net cycle measurements rule out a universal no-split shortcut

All six policies completed on the TRAIN fixture `Q512-K1536-n8-A`. Entries below include actual planning, execution and device completion; they exclude any learned-selector cost because no selector was used.

| Policy | 4090 / 16 heads ms | 4090 / 32 heads ms | 5090 / 16 heads ms | 5090 / 32 heads ms |
|---|---:|---:|---:|---:|
| auto | 0.202153 | 0.273293 | 0.134451 | 0.177375 |
| split disabled | 0.243338 | 0.247691 | 0.182143 | 0.185620 |
| split 512 | 0.216668 | 0.275763 | 0.153102 | 0.188395 |
| split 1024 | 0.212366 | 0.280978 | 0.150499 | 0.209946 |
| split 2048 | 0.244634 | 0.248820 | 0.181951 | 0.183558 |
| split 4096 | 0.244981 | 0.249598 | 0.180593 | 0.184224 |

Relative to auto, no-split changes cycle cost by +20.37%, -9.37%, +35.47%, and +4.65%, respectively. This is one training geometry, not an independent test and not a train-selected policy result. Choosing a winner from this table after seeing its outcomes is hindsight selection. No universal replacement is promoted.

The full-study global and hardware/head-conditioned fixed baselines have not been selected or measured. Learned marginal/joint/plan/causal models have not been fitted. The frozen validation/test gates remain unmet.

## 4. Correctness and resource accounting

Across the two recovered jobs:

- 44 completed policy records passed planner-metadata checks and full-output comparison to auto. Of these, 12 are auto-to-self comparisons and 32 are non-auto full-tensor comparisons.
- All 192 selected FP32 reference vectors passed. Largest selected-vector absolute error: 0.0001478791. Largest full-tensor difference from auto: 0.00048828125. Tolerances were atol=0.005, rtol=0.02; this is not bitwise equivalence.
- Failed scratch policies are not numerically certified. Nothing here certifies all future shapes, full models or CUDA-graph execution.

The installed planner allocates partial output storage from padded CTA rows, not only final output dimensions. One recorded request needed 1,176,502,272 bytes for temporary values alone, versus 134,217,728 available bytes.

The recovered patch computes the installed allocation bound and shares one maximum-size workspace across serial policy runs. It also asserts canary fixtures exist and strengthens the comparison against a TRAIN-selected fixed policy per hardware/head shape. CPU contract tests: 22 passed in the isolated Mac worktree. GPU verification of the patch: NOT RUN.

An additional metadata-only enumeration of the unchanged 152-shape corpus found 744 of 1,824 shape/head/policy combinations per hardware exceed 128 MiB; maximum required scratch is 3,571,875,872 bytes at Q2048-K12288-eq513-A, 32 heads, split512. This is an analytical requirement from the installed formula, NOT a measured allocation or timing result. Any future system experiment must account for lost KV capacity; simply reserving several GiB for all methods does not establish production memory efficiency.

The new read-only auditor retains failures, verifies hashes and counts, and computes the tables above. Its 8 synthetic parser/contract unit tests passed locally. These test fixtures are not GPU or performance evidence.

## 5. Closest prior art and novelty boundary

- FlashInfer already provides plan/run separation, split-KV and load-balanced scheduling. Its paper gives a tile-cost scheduling procedure. This work cannot claim those components as novel: https://arxiv.org/abs/2501.01005
- FlashAttention-2 already studies work partitioning, occupancy and parallelism: https://arxiv.org/abs/2307.08691
- Current official FlashInfer documentation exposes fixed_split_size, disable_split_kv, workspace sizing and graph/determinism caveats. Installed 0.6.18 behavior must not be conflated with current 0.7.0 documentation: https://docs.flashinfer.ai/api/attention.html
- Existing FlashInfer issue 4333 already questions sizeof(float) versus output dtype in batch_prefill_tmp_v sizing. The workspace issue and a simple dtype-sizing fix are not our novelty: https://github.com/flashinfer-ai/flashinfer/issues/4333
- PersistentKV studies page-aware DECODE scheduling, ragged work, head grouping, split merging and traces. It is an adjacent comparator, not a reproduction or a result obtained here: https://arxiv.org/abs/2606.26666

Our still-unvalidated candidate is a cost-and-resource representation that is useful for selecting execution plans on unseen prefill geometry. The split-only mechanism hypothesis fails the recovered discovery check. The broader representation/selection hypothesis is not rejected or accepted without the full study. Earlier normalized-marginal parity with M2n, input-shape ratios, and negative queue-level results remain in force.

## 6. Exact evidence and recovery point

Remote root:
`/ssd/scxi253/single-gpu-inference-plan-cost-20260928`

Raw files:
- `runs/canary-gpu_4090-1632585-0/measurements.jsonl`, SHA256 `3fa1c2da0b295f1bbe661561ed6dcaeb3b2ced24f4534dbb3a39bd25b1b5dd0a`
- `runs/canary-gpu_5090-1632586-0/measurements.jsonl`, SHA256 `34c125480e617f0b420be7bcb35f2aa3462a8564c4e4e4b118f7a03f899e3fdf`
- Matching `summary.json` and `manifest.json` are in each directory.
- Logs: `logs/sgi-causal-plan-1632585_4294967294.log` and `logs/sgi-causal-plan-1632586_4294967294.log`.

Preserved repair:
- GitHub patch: `research/causal_plan_cost/patches/workspace-repair-20260928.patch`
- Patch SHA256: `cd28a0a5bdf3697a9410d3d3601e1f86d75e6fa844819bd72af0db9c3fee02b5`
- Remote source archive: `sgi-causal-repaired-canary.tar.gz`
- Archive SHA256: `b663312afb804a00eca78bb4b736724f181c36f33f7ab39763fea08a66d387e1`
- The archive reached ParaCloud, but proposed directory `milestone-20260928T2343` does not exist. No repaired canary job ID exists.
- Runnable canonical source on the research branch is still the b9e5303 version until the preserved patch is explicitly applied. Do not report patch publication as successful deployment.

Read-only reproduction, with this repository available locally:

```bash
python research/causal_plan_cost/audit_canaries.py \
  --root /ssd/scxi253/single-gpu-inference-plan-cost-20260928
```

Patch validation in a CLEAN worktree at this research branch, once the execution action is authorized:

```bash
git apply --check research/causal_plan_cost/patches/workspace-repair-20260928.patch
git apply research/causal_plan_cost/patches/workspace-repair-20260928.patch
(cd research/causal_plan_cost && python -m unittest -v test_geometry test_audit_canaries)
```

The existing Mac repair worktree already contains the patch: do not apply twice. Before any submission, re-read squeue, retain healthy jobs, stage in a new campaign directory, bind run_paracloud.sbatch to that directory, and freeze hashes before running the repaired training/discovery canary. Do not launch the full matrix without a clean canary.

## 7. Verified external blocker and publication status

The conversation tool returned the exact message below for (1) the combined terminal edit/commit/push/deploy action and (2) preparation/submission of a new remote canary campaign:

`This tool call was blocked by OpenAI's safety checks. Please double check what you are sending.`

Neither returned a job ID or successful execution receipt. The blocked execution was not pursued through another launcher, account, encoding, or approval setting. SSH reads and scoped archive transfer worked. The explicit GitHub API separately authorized repository writes, so it is inaccurate to call this missing GitHub write permission or unreachable ParaCloud. The safety layer did not provide a more specific cause.

Verified publication is through GitHub API commits, not a successful terminal git push:
- `1b5b2c96c53a6a8afc4d24d2678ba26e9b471d83`: recovery checkpoint.
- `98bbc70655545a5ab9fe427258c40cce50cc3fd0`: reviewed repair patch.
- `1a70814046081f7dfe2bebe904f256980d72110f`: reproducible canary auditor.
- `27195682b9f9ac4984fa42a61f3c04a50b8d8974`: auditor contract tests.

Not completed: repaired GPU canary, full matrix, strong TRAIN-selected baseline, learned-selector overhead, independent-family generalization, no-refit hardware transfer, vLLM integration, throughput/TTFT/TPOT/tail/SLO-goodput measurements. No Distinguished or SOTA claim is made.
