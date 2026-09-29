# Order-guard reassessment and research priorities

Date: 29 September 2026. This is a post-campaign research/audit note, not a new GPU campaign or performance claim. The frozen measurement source, policies, manifest, original results, failed gates, and default-OFF decision are unchanged. No new GPU allocation was submitted for this review.

## Evidence inspected

- Campaign: `/ssd/scxi253/single-gpu-inference-order-guard-20260929T1450Z`.
- Measurement commit: `9e09364d13ce06f4dd391c42855e40f8fbbaa261`.
- Manifest: `fb6c14b78cc03f8d02b7b16ff8efa9f382ce313b13e46781f5506a78be7cfa37`.
- Existing analysis: `/tmp/order-guard-formal-analysis-1547/summary.json`.
- SHA256 of inspected summary: `9d21c307bc94f1ef3ccd8e031452067ab420c26e2b89d71e40b5a9bae397122d`.
- Summary records 24 formal tasks, 196,992 timing rows, 8,208 qualification records, 1,368 plans, and 727,296 selected FP32 reference vectors. These are not independent workloads.
- For this audit, raw `plans.json` and `qualification.json` in every formal run were checked against their existing completion-receipt hashes. The old-regression descriptor equality was checked directly in qualification records, not inferred from latency parity.

## Corrections to the earlier interpretation

### The old 42% regression was avoided by retaining identity

For `regression-5090-42pct`, identity and locality_packet8 have the SAME descriptor SHA256 in all 12 relevant qualification comparisons per GPU family (two dtypes x two requested split settings x three repeats). All 24 comparisons agree. The algorithm did not execute a faster reordered schedule on this sentinel. It returned the native order.

Accurate interpretation: the fallback avoided exposing this state to the previously harmful permutation. This does not establish a cache-locality fix, successful replacement schedule, or universally safer ordering.

### Most fresh plans do not change

|GPU|Changed fresh plans|All fresh plans|Change fraction|Changed all plans|All plans|
|---|---:|---:|---:|---:|---:|
|RTX 4090|90|576|15.625%|120|684|
|RTX 5090|132|576|22.9167%|168|684|

Counts include dtype, requested split regime and repeat, not unique workloads. In the fresh graph/warm slice, the post-hoc changed-only descriptive geomeans are 1.001000 for 4090 and 0.998718 for 5090. This subset is a mechanism diagnostic, not a replacement primary endpoint. The unchanged subset also has non-unit timing ratios, demonstrating why near-parity measurements cannot be attributed to the ordering algorithm alone.

### Four of five nominal 5090 primary regressions are no-op cases

|Case|Precision|Requested split|Mode/cache|Changed repeats|Native median us|Native/candidate|
|---|---|---|---|---:|---:|---:|
|fresh-00-same|FP16|unsplit|graph/warm|0|14.528|0.945312|
|fresh-08-opposite|BF16|auto|eager/flush128MiB|0|39.920|0.925988|
|fresh-09-opposite|BF16|auto|eager/flush128MiB|0|67.360|0.948141|
|fresh-18-opposite|FP16|auto|eager/flush128MiB|3|129.392|0.968683|
|fresh-23-opposite|BF16|auto|graph/flush128MiB|0|1105.024|0.899601|

A/A fails on the last row; it passes on the first four. Four rows cannot establish a harmful descriptor permutation because the permutation did not change. They instead flag sampling uncertainty, context, label/order effects or measurement-path differences requiring diagnosis. Passing one A/A comparison does not prove every other identical-work comparison is equivalent.

The old A/A margin is max(3 us, 2% of native median). At 14.528 us, 3 us is about 20.65%, whereas a 1% slowdown is about 0.145 us. Passing this absolute-tolerance A/A gate cannot certify 1% resolution there. Preserve the old gate and its failure decision; do not widen it after observing results. A new protocol must declare its detectable effect and no-op controls before new measurements.

The 133/912 4090 and 194/912 5090 A/A failures are not confined to tiny kernels: the >=1 ms bins contain 57/296 and 70/229 failures respectively. Do not assume launch overhead alone explains all instability. Multiple driver versions occur in 4090 records, and multiple physical devices occur in both families. No hardware-population inference is justified.

### There are still strong adverse cases with passing controls

`fresh-01-opposite` is already a two-request case: query [1024,63], cached prefixes [128,16384]. Under BF16/unsplit/graph/warm, causal_heavy has native/candidate ratio 0.572112 on 4090 and 0.562335 on 5090, with passing A/A controls. This is a better bounded mechanism target than another broad heuristic sweep. It is now exposed development data, never a future unseen test.

## Positive finding: the constructive collision reaches the actual planner

For collision-0 A/B, inspected FP16 repeat-0 auto plans on BOTH GPU families use query tile 128, KV chunk 128 and split enabled, producing 22 versus 20 work descriptors. Under requested unsplit, both produce 12. The states preserve query, cached-prefix and total-KV marginal multisets and logical W=49,344, as specified by the previously tested construction.

This is actual planner-metadata evidence rather than only an abstract fixed-split calculation. It is not proof of a stable latency gap, nor of exactly 22 versus 20 physical CTAs: head expansion and launch mapping must be inspected separately. The current broad timing controls do not justify promoting this to a universal performance law.

## Current primary-source research

1. FlashInfer v0.7.0 release and Autotuner v2: https://flashinfer.ai/releases/ ; https://flashinfer.ai/2026/09/22/autotuner-v2.html ; https://flashinfer.ai/2026/09/22/flashinfer-v07.html . Deployment-matched eager/graph measurement, default-path comparison, environment-sensitive persistence and reuse are upstream work, not our novelty. Verify whether each actual attention path participates; a generic autotune context must not be assumed to tune an arbitrary private FA2 descriptor permutation.
2. CUTLASS grouped schedulers: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html . Descending-work sorting is established, and the documentation explicitly notes that sorting may reduce performance. Triton's grouped matmul tutorial likewise establishes program-order/L2-locality tuning: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html . Ordering and locality alone are not a new-to-literature contribution.
3. FlashInfer paper: https://arxiv.org/abs/2501.01005 . Load-balanced scheduling and plan/run separation already form part of FlashInfer's design.
4. NVIDIA profiling guidance: https://docs.nvidia.com/nsight-compute/ProfilingGuide/ . Replay mode, cache flushing, clock control and launch serialization change the measured context. Profiled durations must not replace unprofiled deployment timing.
5. FlashInfer timing API: https://docs.flashinfer.ai/generated/flashinfer.testing.bench_gpu_time.html . Distinguish CUDA-event, CUDA-graph and CUPTI timing scopes. Longer repeated graph timing measures a different cache/reuse regime from a single cold invocation unless explicitly controlled.
6. CUDA scheduling semantics: https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/01-introduction/programming-model.html . Descriptor array order does not guarantee actual SM issue order. Do not interpret packet8/window32 as hardware waves.

## Real upstream opportunities and boundaries

### FlashInfer PR 5188 — specific source review, not reproduced bug report

Read PR metadata, full patch and the relevant source at head `9f84a6da61615513be13a53e1957b9131f0133a7`. It is open and targets SM90 paged prefill backend selection for short per-request queries. This is not a 4090/5090 performance claim.

https://github.com/flashinfer-ai/flashinfer/pull/5188

The shown short-query guard is nested in `_backend == "auto"`, sets `_backend` to a resolved backend, and calls GPU `max().item()`. The added tests construct a new wrapper each time. High-value tests to add before reviewing publicly: repeated planning on one wrapper (short -> long and long -> short), heterogeneous query lengths, explicit backend override, FP8 query compatibility, graph lifecycle, and shared versus disjoint KV pages. Inspect all reset paths before claiming sticky selection is a confirmed bug. Measure whether the host indptr already available in plan can remove redundant device reduction/synchronization. GPU reproduction requires a supported SM90 device.

### FlashInfer issue 3935 — relevant RTX 5090 end-to-end regression

https://github.com/flashinfer-ai/flashinfer/issues/3935

The issue is open, but comments already include a source-level repack attribution for one historical 0.6.12 -> 0.6.13 configuration, non-reproduction on a different TP/device setup, and a September 2 correction that the old head-dimension gate does not fix that day's main. Do not present that old fix as novel or current. Reproduce the reporter's path, compare current supported source, attribute the remaining runtime loss and coordinate with the existing investigation before proposing a patch. Our D128 ragged BF16/FP16 ordering assay does not solve this hybrid-model/D256/FP8/paged serving problem.

### FlashInfer PR 5687 — relevant prior art, different kernel

https://github.com/flashinfer-ai/flashinfer/pull/5687

Open, inspected head `ecbee2f0871cb282c96885bdffa7a64dc3ce7491`. Persistent tail-queue scheduling for SM90 VSA/cake. It is not dense FA2, and H100 workload geomeans cannot be ranked against our 4090/5090 run-only geomeans. Do not duplicate its collector or claim its scheduling principle as new.

### vLLM RFC 55259 — contribute adversarial fixtures, not a duplicate interface

https://github.com/vllm-project/vllm/issues/55259

The proposal already specifies schema-v3 exact ordered [new_tokens,kv_read_tokens] rows and graph-boundary sampling. A useful complementary artifact is a small source-conformant suite of feature collisions and transition tests. It remains proposed functionality, not a deployed guarantee. The paired FPM scope must be checked against the actual PR revision, not inferred from the RFC alone.

## Recommended execution order

### Stage A: qualify measurement before learning a dispatcher

Freeze a small diagnostic suite containing literal identical-work labels, identity_repeat, locality no-ops, the 14.528-us case, the two-request large regression, and a >=1-ms unstable case. Same GPU UUID and pinned source/driver within comparisons; record affinity, allocation overlap, clocks and power. Use balanced, predeclared A/B placement. Separate pure kernel, graph replay, native plan+run and external serving costs. Pre-create timing objects where appropriate. Keep single-call, steady replay and cache-pressure conditions separate. Calibrate a sub-1% claim against matching-resolution controls; report 'unresolved' instead of inventing precision. Do not pool arbitrary shard block numbers as though they were one shared physical process.

### Stage B: isolate one large reversal

Keep raw Q/K/V and output addresses fixed. For the two-request exposed case, compare native, reverse and heavy order while recording actual kernel symbol, grid/block, registers, shared memory, split/merge metadata and request-to-descriptor mapping. Collect small diagnostic profiler captures separately for DRAM/L2 traffic, occupancy and stall/imbalance evidence. Intervene on request grouping or supported split/tile controls only under explicit correctness contracts. Never assign a stale plan to another shape. Establish which intervention removes a loss before building a predictor around a guessed hardware mechanism.

### Stage C: current-backend and production-opportunity checks

Reproduce the bounded witnesses on 0.7/current pinned supported implementations. Distinguish FA2-only wins from wins over native auto and other supported backends. Measure the actual engine's plan-signature reuse distribution and attention share before assuming 36 layers or thousands of future repeats. Synthetic same-buffer reuse is not serving reuse. Use an oracle over tested candidates only as an optimistic diagnostic ceiling, not an achieved policy. If lifecycle overhead exceeds available savings, switch to offline calibration, a cheaper structural patch, or the direct upstream regression work.

### Stage D: candidate research, not a promised new algorithm

After A-C, investigate a plan- and reuse-aware selective scheduler. Group descriptors by actual request/KV segment reuse structure rather than arbitrary fixed packet widths. Model discrete padded loop work, data-footprint/reuse proxies, reduction work and CPU plan cost. Generate a bounded candidate set including identity; learn relative advantage, not just absolute latency, with independent calibration and OOD rejection. The mathematical objective is an empirical lifecycle decision, not a per-call safety theorem:

`N * lower_bound(T_native - T_candidate) > probe_cost + extra_plan_cost + invalidation_cost + N * dispatch_cost`.

An unknown reuse horizon abstains BEFORE expensive probing. Previously incurred cost remains charged after fallback. The lower bound must be calibrated at the declared unit and under stated assumptions; a learned uncertainty score is not automatically a frequentist guarantee. Prefer a compact decision tree/native rule first; add Bayesian or bandit complexity only if it beats simpler and upstream tuning baselines under the same budget.

### Stage E: genuinely fresh validation and application

All 57 states are now exposed. Preserve them as development/regression fixtures and generate a new family-level train/calibration/test split before tuning. Score unconditional all-traffic utility, selected-state coverage, conditional gains, worst-case/tail slowdowns, and total tuning/plan/dispatch cost. Correctness requires exact output/LSE for order-only changes, and a separately declared numerical/model-quality contract for backend changes. Check ragged/paged, page aliasing, graph transitions, dynamic plans, multiple streams and lifetimes. Only a pristine native full-model comparison with actual TTFT/TPOT/tail latency/goodput can support a serving claim. No finite suite proves a universal worst-case >=0.99 guarantee.

## Decision

Do not launch another broad fixed-order sweep or train a high-capacity dispatcher on unresolved tiny effects. First correct the no-op interpretation, establish trustworthy effect resolution, and explain the compact real reversal. Preserve the new actual-planner collision as a separate deterministic result. The research upside lies in a validated mechanism and a measured low-overhead remedy or an actionable upstream regression fix, not in renaming existing heuristics.
