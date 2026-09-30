# Selector v4: safe deployment-matched tactic autotuning

## Decision and scope

V3.2.2 is permanently HOLD and its 30 release geometries are exposed development data. V4 does not relax its 0.99 safety gate or relabel those cases as fresh. It replaces the static guarded selector with two explicit tactics: `native` and `resource_cap`. Static logic only determines physical eligibility; a tactic is selected separately for each full environment/operation/execution identity by deployment-matched measurement. Until a valid cache entry exists, native is mandatory.

This remains a research prototype. Historical 2/432 full-model token divergence is unresolved, so even a dual-GPU release PASS authorizes only controlled full-serving validation, never default enablement.

## Identity and cache contract

Environment identity includes GPU name/UUID/SM count, driver, CUDA, Torch, FlashInfer, NVCC, backend source hash, official overlay hash and shared-memory limits. Operation identity includes eager/Graph1/Graph16 mode, layout, dtype, actual split, heads/dimensions/page size, ordered q/cached geometry, tactic-neutral plan-core signature and frozen timing window. Measurement policy is identity-bearing.

The cache is content-addressed below `v4/<environment_hash>/entries/`. Each entry embeds the full canonical identity, selected tactic, evidence receipt and provenance. Publication uses a same-directory temporary file plus `os.replace`. Missing, malformed, foreign, stale or runtime-rejected entries are cache misses and fall back to native. A cap entry is invalid unless its exact frozen thresholds passed.

## Static eligibility

Eligibility is deliberately broad and fail-closed: standard causal FA2, FP16/BF16, supported 32/8 heads with D128, ragged/page16 layout, batch at least five, cached prefix at least8192, actual unsplit plan, and sufficient shared-memory capability. Eligibility is not a performance prediction.

## Frozen tuner threshold

Using only exposed v3.2.2 development data, leave-one-process-out analysis froze: training geomean native/cap at least1.05, paired bootstrap95% LCB at least1.01, every training block at least0.99, exact outputs, native/cap duplicate controls within reciprocal±0.5%, and at least16 training blocks. Failure of any check selects native. These thresholds cannot change after v4 canary starts.

## Measurement and cross-fitting

Each cell is measured in three independent processes. Every process has eight ABBA/BAAB blocks comparing native and cap in the same candidate binary. Eager measures recurring plan+run wall time. Graph1 and Graph16 use independently captured graphs. Pristine pilots freeze all timing-window counts; every scored window is at least12ms and the same count is reused by every arm and process.

For each of three folds, two process repeats train the tuner and the third is an untouched held-out evaluation. Native fallback contributes exactly1.0 to policy performance. Promotion requires exact output/LSE, selected held-out point and block worst at least0.99, simultaneous joint-min95% LCB at least0.99, whole-policy worst at least0.99, resolved held-out controls, nonempty selection in all three execution modes, and both dtypes/layouts.

## Fresh data hierarchy

The deterministic manifest contains10 canary,48 release and12 stress cases. It records hashes for every historical manifest and rejects any exact `(q,cached)` collision. Canary must PASS on both4090 and5090 before release. Release must PASS on both GPUs before stress or controlled full-serving work. Stress is transfer evidence, not a substitute for release.

## Primary design references

FlashInfer Autotuner v2 PR3861 provides the managed environment-hashed cache, atomic per-entry publication, deployment-matched `MeasurementPolicy`, self-describing tactic validation, and default-candidate coverage guard used as design precedent. NVIDIA Nsight Compute profiling remains diagnostic-only because profiler replay/cache/clock controls can change the execution boundary. No profiler timing enters release qualification.
