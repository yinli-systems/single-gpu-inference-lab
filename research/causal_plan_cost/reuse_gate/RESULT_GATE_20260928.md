# Direct matched reuse gate — completed 2026-09-28

This is an actual completed GPU experiment, not CPU-cost division or a synthetic amortization projection. Measurement source commit ddfd23a9e3aa5661457f2c4626b7d9e899042963. All six test jobs finished, three independent processes/seeds per GPU, with no test rows dropped. Preserve the earlier 660d128 unamortized dispatcher rejection.

## Frozen direct experiment
54 fresh exact geometries (18 bases and3 metadata states each),9 held-out query/cache families, head configuration32/8,D128,36 distinct Q/K/V allocations by layer,FP16,FlashInfer0.6.18 explicitFA2,eager. No weights,MLPs,serving scheduler or network: NOT full-model inference. All arms get the same memory ceiling, policy-cache capacity and active-plan reuse opportunity.

Each episode has four segments. At the primary36-layer setting this means144 actual attention calls. Unchanged state uses one plan; alternating A/B/A/B and eviction A/B/C/A replan once per36-layer segment. The decision LRU holds two metadata keys; no disk-persistence claim.

Four arms:stock auto, TRAIN-only memory/head/reuse-conditioned fixed, frozen native cycle policy PRIMARY, frozen native run policy SECONDARY. The fixed policy uses old TRAIN-only cycle+(r-1)*run as its selection proxy; it is not asserted to be the empirically optimal fixed choice under the new data-traffic regime. Both fixed and stock auto are reported. No fitted coefficient was changed.

## Primary36-layer results
Ratios are fixed-policy wall time divided by native-cycle wall time. Above one is favorable. Nominal95% intervals use20,000 paired whole-family bootstrap draws over9 families,seed20261011, after median reduction of three process repeats. They are conditional suite evidence, not device-population guarantees or multiplicity-adjusted claims.

| GPU | Scratch MiB | State sequence | Native vs TRAIN fixed;95%CI | Native vs stock auto;95%CI | >5% regressions vs fixed |
|---|---:|---|---|---|---:|
|4090|128|unchanged|0.9847 [0.9523,1.0040]|0.9980 [0.9971,0.9990]|2/18|
|4090|128|alternating|0.9917 [0.9694,1.0082]|0.9974 [0.9964,0.9983]|2/18|
|4090|128|eviction|0.9896 [0.9663,1.0043]|0.9980 [0.9971,0.9987]|2/18|
|4090|512|unchanged|1.1669 [1.0684,1.2862]|1.1676 [1.0690,1.2869]|0/18|
|4090|512|alternating|1.1416 [1.0469,1.2615]|1.1334 [1.0454,1.2403]|0/18|
|4090|512|eviction|1.1477 [1.0512,1.2667]|1.1411 [1.0510,1.2489]|0/18|
|5090|128|unchanged|1.0000 [0.9998,1.0002]|0.9998 [0.9996,1.0000]|0/18|
|5090|128|alternating|0.9998 [0.9996,1.0000]|0.9998 [0.9995,1.0003]|0/18|
|5090|128|eviction|0.9998 [0.9997,1.0000]|0.9999 [0.9997,1.0001]|0/18|
|5090|512|unchanged|1.0137 [0.9898,1.0528]|1.0523 [1.0154,1.0970]|1/18|
|5090|512|alternating|1.0240 [0.9960,1.0652]|1.0633 [1.0268,1.1049]|1/18|
|5090|512|eviction|1.0209 [0.9951,1.0614]|1.0600 [1.0213,1.1064]|0/18|

Only4090/512MiB passes the primary and stock-control gates in all three sequences.5090 gains over stock do NOT establish gains over the stronger fixed baseline.128MiB is NOT promoted on either GPU. Do not reinterpret a secondary selector or intermediate reuse count as the primary winner.

This experiment simultaneously changes the data-traffic context and memory constraints relative to the earlier microbenchmark. It does not causally attribute the entire benefit to amortization alone. The1/2/4/8/16/36 direct measurements are retained; no new universal crossover count is inferred from a favorable average.

## Numerical, cache and memory validation
3,888 complete timing rows;1,296 cells after process-repeat reduction;5,557,248 timed attention invocations across all layer counts,states,budgets andarms. Inner invocations are not independent observations.

Deduplicated checks:22,248 non-auto full-tensor comparisons and25,920 selectedFP32 vectors passed. Max absolute difference vsauto0.00048828125; vsFP320.00012940168380737305. Tolerancesatol0.005/rtol0.02; not bitwise or full-model equivalence. All36 layer outputs/allstates/selectedmodes are checked before timing, not copied back and scored after every timed call.15,552 arm/row cache-counter checks pass, including actualplan/run/hit/miss/eviction counts.

Peak PyTorch allocated9,326,039,552 bytes; max distinct layer tensors7,832,567,808 bytes. Allocator peaks include reference/validation temporaries; these are not full-serving KV capacity or totalGPUprocessmemory.512MiB scratch is a real resource requirement, not free reuse.32 CPU contract/statistical tests passed;216 additional fresh-geometry native decisions matched unchanged frozen models before fresh timing.

## Follow-up boundary
The supported4090/512MiB scope may proceed to cached-transition output and full-vLLM path qualification. The ragged FP16 eager result must not be silently transferred to paged caches, BF16, CUDA Graphs or a different FlashInfer version. No full-vLLM throughput,TTFT,TPOT,tail orSLO-goodput result is claimed by this gate.

## Evidence
Root `/ssd/scxi253/single-gpu-inference-plan-cost-20260928`.
Campaign `campaigns/reuse-gate-v1`, jobs4090:1632759_0/_1/_2,5090:1632760_0/_1/_2. Preceding canaries1632714/1632715 retained separately.
Audit `artifacts/reuse-gate-audit-v1/direct_results/metrics.json`, SHA256 `2cd464b70015eeef2089ed427876bc96a29c48079970b4fcdf08e9f9b9cd456a`.
Per-cell values/replicas: `direct_results/per_cell.json`. Recovered wave and old-cost break-even diagnostics remain under `artifacts/reuse-gate-audit-v1/recovered/`.

All writes/launches used the normal authorized tool paths. No safety refusal was bypassed, no healthy task was cancelled, no new resources purchased. This milestone is AI-assisted and makes no award or global novelty guarantee.
