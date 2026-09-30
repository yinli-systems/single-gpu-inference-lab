# Selector v4.1 — isolated tactics and fail-closed deployment-matched autotuning

## Scope and immutable history

The complete v3.2.2 release remains development evidence and remains HOLD. Its 96/96 runs must not be repeated or relabeled fresh. RTX4090 exposed strong selected gains but fallback/disabled-overlay regressions; RTX5090 additionally exposed a selected `opp-04` regression. Those 30 shapes and every earlier v3/v4 manifest are in the historical deduplication ledger.

The sibling v4-autotune branch established development evidence for separate native/resource kernel symbols in private JIT caches on RTX4090/5090. V4.1 integrates that mechanism with the safe-autotune identity, cache, cross-fit and freshness contracts. The sibling smoke is supporting evidence only; this source revision requires its own smoke.

Default and serving promotion remain OFF. The historical 2/432 full-model token divergence is unresolved and independently blocks production promotion.

## Mechanism: isolate mutable CUDA function state

`cudaFuncSetAttribute` targets a concrete device function. V4.1 therefore duplicates the ragged and paged FA2 prefill kernels into distinct `...ResourceKernel` symbols. The native kernel source span is byte-identical to official FlashInfer0.7. Only the resource clone receives the 64KiB dynamic-shared-memory attribute and launch. The legacy plan entry remains ABI-visible and defaults to native policy0; an explicit research plan entry supports only native0 and resource-cap1.

Every GPU job uses campaign-private pristine/candidate JIT roots. After compilation, `audit_binary.py` requires native/resource ragged and paged symbols to be co-resident in compiled tactic modules. A candidate cache hit is rejected at runtime unless the wrapper exposes both `plan_resource` and `resource_kernel_isolation=True`.

## Eligibility and unsupported paths

Static eligibility is deliberately broad and physical: FA2, causal, D128 QK/VO, 32/8 heads, supported dtype/layout, actual unsplit plan, batch>=5, max cached prefix>=8192, valid device shared-memory limits. It is not the final selector.

A cap arm is executed only when static eligibility passes and the source-bound runtime probe succeeds. Split/ineligible/unsupported cells use a second independent native wrapper as an explicitly recorded null arm; they cannot publish or apply a cap tactic. Every paired cell records actual tactic, cap support, probe error if any, plan flags, exact output/LSE and a `native -> cap -> native` isolation check.

## Identity, cache and deployment modes

Tactic identity binds GPU name/UUID/SM count, driver, CUDA, Torch, FlashInfer, nvcc, official overlay, candidate source, resource-binding hash, shared-memory limits, full ordered q/cached geometry, dtype/layout/page size, actual split, plan signature, and execution-specific measurement policy.

Eager full-call, Graph1 replay and Graph16 replay are distinct identities. A tactic from one mode/GPU/software environment cannot alias another. The cache is content-addressed, atomically published under a lock, corruption/conflict rejecting, environment isolated and native-on-miss. Runtime validation occurs before plan/capture; every failure falls back native.

## Measurement and cross-fit

Each cell covers FP16/BF16, ragged/paged and requested auto/unsplit. Pristine references and candidate paired runs are separate processes on one physical GPU per shard. Three process repeats use process order pristine/paired, paired/pristine, pristine/paired.

Each process uses 16 ABBA/BAAB blocks. Eager and graph windows target >=96ms and every scored window must be >=12ms. Release timings are unprofiled. Nsight/Nsight Compute mechanism profiles are separate evidence because profiling can alter clocks, caches, replay and launch serialization.

For every identity, two repeats train and the third is held out, rotating across all three folds. Selection requires exact outputs, static/runtime eligibility, >=32 training blocks, training geomean>=1.05, bootstrap95% LCB>=1.01, every training block>=0.99, and native/cap duplicate controls within +/-0.5%. Any failed condition chooses native. Held-out results are never used to alter that fold's choice.

## Qualification gates

Canary/release require:

- exact complete output and LSE for every arm and eager/Graph1/Graph16 replay;
- native-after-cap exactness;
- compiled symbol isolation and stable active-clock telemetry for every shard;
- nonempty cap selection across both dtypes, layouts and all execution modes;
- selected held-out geomean>=1.05;
- selected point worst, block worst and simultaneous joint-min95% LCB >=0.99;
- whole-policy worst>=0.99;
- independent candidate-native versus official-pristine point worst and simultaneous joint-min95% LCB >=0.99, with both duplicate-control CIs inside reciprocal +/-1%;
- held-out controls resolved;
- cache publication/reload round trip.

No 0.99 threshold may be loosened after seeing canary or release. A failed canary makes its shapes development evidence; selector/threshold changes require a new untouched holdout.

## Freshness and stages

The manifest contains 2 exposed dev cases, 10 untouched canary cases, 48 untouched release cases and 12 untouched stress cases. Fresh q/cached pairs, q vectors and cached vectors are exactly deduplicated against all recorded historical manifests, including the sibling v4 branch.

Order is mandatory: CPU/source validation -> dual-GPU private-cache smoke -> dual-GPU canary -> dual-GPU release -> stress -> paired full HTTP. Full HTTP additionally requires complete token parity and nonregressive throughput/goodput/TTFT/TPOT. The unresolved historical divergence remains a separate blocker even if performance passes.

## V4.2.0 superseding contract (2026-10-01)

V4.1 history above is retained as development evidence. V4.2 restores the official
15-field plan. Scheduler, C++ plan/native host runs/native dispatch/native kernels,
Python wrapper methods and native module custom ops are source-identical. Only
resource run/dispatch/kernel symbols and resource custom ops are appended. The
module factory registers additional ops outside recurring native execution.

`run_resource` is opt-in and requires standard FA2, an unsplit official plan and
matching paged K/V strides. Unsupported resource probes become labeled native null
arms. `runtime.selected_run` resolves the validated cache choice after official planning, before capture;
the official `wrapper.run` is never replaced. This research cache remains separate
from upstream Autotuner v2 pending validated integration.

Frozen measurement: 3 independent process repeats, 24 ABBA/BAAB blocks, pristine
calibrated >=384ms target windows and >=120ms minimum scored windows, no sample
trimming. Eager includes official plan+run; Graph1/Graph16 are replay only, not full
serving steps. Every binary audit compares all native attention SASS against the
independent pristine build, normalizing only PC labels and whitespace. Missing,
extra or differing native kernels block qualification.

Smoke covers both dtypes/layouts and native->resource->native. Source-bound dual-GPU
smoke authorizes exposed dev only. Both dev passes authorize canary; both canary
passes authorize release; both release passes authorize stress. All original 0.99
floors remain, including candidate-native/pristine simultaneous LCB. Actual chosen
policy/pristine point worst is additionally required >=0.99. Held-out oracle regret
is reported as `oracle latency / chosen latency - 1`, including P50/P90/P99, worst
and >1% counts. Oracle evidence never trains the tactic.

The original case hash and every stage hash are unchanged. Canary/release/stress
consumption must be recorded at dispatch even if an execution fails. HTTP/default
promotion remains OFF until the historical 2/432 divergence is explained and full
serving token parity and SLO/performance gates pass.

V4.2 exact dev uses `srun --cpu-bind=cores`; each process narrows only its allocated
core set and records affinity. NVML hardware queries and telemetry target the UUID
reported by Torch for the actual CUDA device. Missing UUID or cross-repeat affinity
drift is a blocking failure. Resource FFI absence leaves the official native module
available; cached resource choices are rejected if the actual 15-field plan differs.
