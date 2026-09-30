# Fallback safety: separate host-span and device-execution failures

## Current verdict

Both source-a849cf4 v3.2.1 canaries are complete. Both remain HOLD. Across the two GPUs, 4,608 timing rows and 128 process-level qualification records were retained. Saved full output/LSE checks pass; these are four canary geometries, not a release generalization result. The thirty v3.2 release cases are still unexecuted.

| GPU / failed coordinate | Pristine / guarded wall | Off / guarded wall | Off / guarded device (diagnostic only) |
|---|---:|---:|---:|
| 5090: below, BF16, ragged, auto, Graph1 | 0.977012 | 0.982278 | 1.000331 |
| 4090: near, FP16, ragged, auto, Graph16 | 0.977139 | 0.974593 | 0.974428 |
| 4090: near, FP16, ragged, unsplit, Graph16 | 0.978463 | 0.978180 | 0.978111 |

The device-only column cannot replace the frozen wall-time objective or overturn HOLD.

## What the original rows actually establish

At the cited5090 coordinate, off and guarded have equal plan vectors, including resource_cap=0. The actual plan is split. One guarded observation is wall1289.334125us versus device1196.938038us; the window contains sixteen attention invocations. The gap is92.396087us per invocation, about1478.337us over the window. Its other guarded device observations remain around1197us. This establishes separation between wall and recorded GPU spans. It does not prove which host scheduling, synchronization, allocation, or timing event caused it.

The previous suggestion of a stable2.3% fallback-code cost was too strong. In particular, native plan-time work is not replayed by the timed CUDA Graph. A subsequent rewrite of the planner cannot be called a demonstrated fix for this Graph1 sample.

The4090 cases are different: their GPU-event comparisons also regress in the old sample. Do not dismiss them using the5090 explanation. Coarse two-second telemetry cannot identify each short timing window or fully exclude clock/interference effects. Actual graph instance, storage, and launch context need paired interventions.

## Implemented diagnosis

A fixed diagnostic campaign uses five already-exposed coordinates (three underlying canary geometries), three processes on each GPU, eight ABBA/BAAB blocks, Graph1x16 andGraph16x1. It compares independent wrappers, distinct captures sharing the same storage, and an exact same-graph null control. Full output/LSE must match saved pristine references; native plan vectors, pointers, graph topology, launch grid/block and shared-memory metadata are saved.

Three diagnostic timers compare legacy per-window CUDA events, pre-materialized persistent events, and full wall time with stream completion and no event instrumentation. Thread CPU time, voluntary/involuntary switches, wall stages and available device spans are recorded. Deliberate post-GPU host sleeps are separately labeled positive controls, not performance samples. Every normal sample, including slow ones, remains in the statistics.

The fixed matrix is8,640 non-injected windows per GPU plus15 labeled positive controls. This is a planned count until completion. No diagnostic result can authorize release. The same-graph null arm is not an optimization result.

## Concrete optimization candidate

`native-fast-reject-v321.patch` adds an early native rejection for split plans or batch_size<5. The existing selector necessarily returns false in those cases, so its rule is unchanged and the expensive pairing/device-query branch can be skipped. The patch was generated against the exact source-bound candidate; it is not applied to the frozen running environment. No GPU speedup or CUDA compilation result is claimed for this patch.

Eleven CPU tests passed, including the equivalence boundary, source-anchor rejection, graph-metadata sensitivity, missing/duplicate/order-invalid sample rejection, and an assertion that a single slow observation remains in the geometric aggregate. The existing release thresholds are not relaxed.

## Jobs and immutable identities

Diagnostic source: `80b6f5c5634e1ad41404213abc3e95c1d6544c3e`.
Campaign: `/ssd/scxi253/sgi-fallback-diagnosis-20260930T104231Z`.
RTX5090 job:1641985. RTX4090 job:1641986.
At2026-09-30 10:52:32UTC both were PENDING(Priority), with no diagnostic progress, completion, or failure records. Read the live queue before acting; never resubmit from the historical planned receipt.

Parent source remains `a849cf4c9894d2d123d66dc27e417e162526e1c6`; source-a849 jobs1641933/1641934 are complete/HOLD. The older4090 OOM did not recur on its full four-case v3.2.1 canary, but this does not guarantee memory safety at arbitrary scale.

Raw canary archive SHA256: `7b9391e897417da3569d3c9fe4f25326dd44db38f192d1ce3180b8960d3dd624`. Individual complete-receipt hashes were checked. `evidence/canary-audit.json` retains the unmodified failed-cell observations and the device/wall decomposition.

## Independent blockers and next admissible step

No release or full HTTP experiment is submitted here. First complete the fixed diagnosis and identify a measurable cause; any implementation or measurement-boundary change needs a new version and qualification, with old failures preserved. An unseen logical-geometry release is distinct from generalization across models, traffic or hardware populations. The original2/432 full-model token divergence remains unresolved and independently blocks default promotion. PR27's HTTP lifecycle work remains separate at672822f. Max GUI selection has not been verified.

## Primary-source grounding

FlashInfer Autotuner v2 (2026-09-22), https://flashinfer.ai/2026/09/22/autotuner-v2.html : eager recurring call cost and captured replay are different objectives; identity and lookup overhead matter. PyTorch CUDA semantics, https://docs.pytorch.org/docs/main/notes/cuda.html : replay reuses captured kernels/arguments/addresses, not Python/native planning work. NVIDIA Nsight Compute Profiling Guide, https://docs.nvidia.com/nsight-compute/ProfilingGuide/ : profiling replay, cache and clock controls can change the measured context. NVIDIA CUDA13.0.2 event documentation, https://docs.nvidia.com/cuda/archive/13.0.2/cuda-runtime-api/group__CUDART__EVENT.html : event completion and host synchronization have distinct behavior. These sources motivate controls; they do not establish this project's cause or performance.

## Diagnostic tooling correction (superseded campaign)

The first diagnostic campaign did not produce performance evidence. RTX4090 job `1641986` failed after40s because PyTorch `CUDAGraph.get_graph_data()` requires `cudaGraphNodeGetToolsId`, which needs cuda.bindings/driver13.1 support unavailable on the frozen CUDA13.0/580.82.07 environment. RTX5090 job `1641985` was still PENDING and was cancelled before start after the deterministic source defect was identified. Failure/cancellation receipts are retained under `evidence/tooling-v1-failure/`; no release geometry was consumed.

The corrected diagnostic uses the CUDA13.0-compatible graph-management APIs already present in `cuda.bindings`: `cudaGraphGetNodes`, `cudaGraphGetEdges`, `cudaGraphNodeGetType`, `cudaGraphGetId`, plus driver `cuGraphKernelNodeGetParams`/`cuFuncGetName`. It deliberately omits ToolsId while retaining node types, dependency edges, kernel names, grid/block dimensions and dynamic shared-memory sizes. Graph instance IDs are excluded from the topology signature. CPU/fake-binding contracts verify the compatibility parser and signature invariance. This correction changes diagnostic instrumentation only; it does not alter selector, canary results, performance gates or release cases.

### Second compatibility correction

Corrected-v1 RTX4090 job `1642499` progressed through node/edge/type/kernel-parameter extraction, then failed because `cudaGraphGetId` itself returns CUDA error36 on the frozen driver. That ID is diagnostic-only and excluded from `normalized_graph()`/the topology signature, so corrected-v2 treats graph ID as optional while continuing to require the node/edge/kernel launch metadata. Pending RTX5090 job `1642500` was cancelled before start to avoid repeating the same deterministic tooling failure. Receipts are preserved under `evidence/tooling-v2-failure/`; no performance or release evidence was produced.

### Corrected-v2 diagnostic execution

CUDA13.0 compatibility source `3057b3e42b9e83532cc1f55eefeb4a3b20eac2cd` is frozen in campaign `/ssd/scxi253/sgi-fallback-diagnosis-v3-20260930T125400Z`. Jobs: RTX4090 `1642536`, RTX5090 `1642537`. At 2026-09-30 12:55:07UTC, 4090 was RUNNING and had completed its first full case with577 rows; this proves the compatibility path progressed past both prior metadata failures. RTX5090 remained PENDING(Priority). These are diagnostic jobs only; no release case or full HTTP run was launched.
