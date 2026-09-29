# Section-six execution: timing resolution, resource intervention and native-cycle evidence

## Decision

**Not a completed A–E production qualification.** A/C produced 11 of 12 complete diagnostic processes; the final RTX 4090 / 0.7.0 repeat timed out and remains unscored. B completed a source-isolated resource intervention. D completed a pristine/disabled/candidate native planning-plus-execution comparison. E was NOT executed: the prototype is deliberately exact-witness-only, not a general dispatcher or qualified full-model integration. All previous default-OFF decisions remain.

The useful advance is a measured, current-version intervention that removes one harmful work-order reversal and preserves a small benefit after real native planning overhead. It is not a new-to-literature scheduling theorem, SOTA claim, upstream acceptance, or serving speedup.

## A/C: measurement qualification and official-version replication

Frozen diagnostic source: `42ec44cd60865626e11e6bea1c0ba2c0a72b51b6`. Six already-exposed geometries, FP16/BF16, auto/unsplit requested split settings, explicit FA2, initialized CUDA events, literal identical-work labels, 12 balanced ABBA/BAAB blocks, and three fresh processes per version on one allocated GPU. Single eager call, single graph call, 32-call steady graph and single graph call after a 128-MiB write are separate conditions. Cache pressure is not proven L2 eviction. Clocks were recorded, not locked; whole-node exclusivity was not established.

Both no-op labels must have a 90% conditional process/block interval inside [1/1.005, 1.005] before interpreting 1% effects at that cell. Many single-call cells fail this resolution requirement. For RTX 5090 / 0.7.0, 10/24 graph-one warm cells and 22/24 graph32 steady cells pass; the latter does not substitute for the former. The large two-request BF16 / unsplit regression passes matching controls in every scored regime and persists on 0.7.0: graph-one warm native/causal-heavy is 0.560199 [0.559884,0.560512].

Completed A/C evidence: 253,440 timing records, 1,584 qualification records and 264 plans. RTX 4090 / 0.7.0 has only two complete repeats; no confirmatory interval is assigned to that incomplete subgroup. The timed-out run's environment, progress and scheduler cancellation remain in the archive. An attempted time-limit extension was denied; no privilege workaround or replacement sample was used.

Both 0.6.18 and 0.7.0 native headers match their official tag Git blobs. The 0.7 package is an isolated, SHA-verified PyPI wheel overlay, not a shared runtime upgrade. One unused NCCL extension dependency was absent; this experiment qualifies the tested single-GPU FA2 calls, not the whole package. Explicit FA2 results do not qualify all auto-selection paths, FP8, CuTe, paged or distributed backends. Cross-version launch order was fixed and does not support a causal version-speedup claim.

## B: resource-reservation intervention

Exposed witness: queries [1024, 63], cached prefixes [128, 16384], total KV [1152, 16447], BF16, unsplit, GQA 32/8, D128. The actual kernel has grid [34, 1, 8] (272 CTA blocks) and block [32, 4, 1]. A host-only native source patch varies dynamic shared memory from 49,152 to 65,536 bytes. Device kernel source, mathematical work, per-order descriptor identities and Q/K/V/output addresses remain unchanged. Separate profiling verifies actual launch sizes and kernel symbols; profiler durations are not benchmark observations.

Both GPUs report 102,400 shared-memory bytes per SM. Therefore shared-memory capacity alone limits co-resident blocks to at most 2 versus at most 1. This is a resource-capacity bound, NOT measured CTA-to-SM issue order. The profiler's occupancy estimator incorrectly reports zero active blocks for the successful 64-KiB launch. That raw zero is retained in `resource-bounds.json`, not rewritten as a measured one. The intervention implicates resource constraints but does not uniquely distinguish cache effects, block issue order or other residency effects.

Three processes per GPU, all exact-output/LSE checks and literal A/A controls passed on this witness. B contains 3,456 unprofiled timing records and 30 qualification records. At graph-one warm, RTX 5090 native48/heavy48 is 0.56021, while native48/heavy64 is 1.02098. RTX 4090 gives 0.57208 versus 1.00772. This removes the large harmful-order penalty; the meaningful improvement over the original native path is only about 0.8% / 2.1%, not a general 1.8x speedup.

The first preparation attempt stopped on a source-anchor assertion before any candidate GPU job: 0.7 moved ragged dispatch into an Impl function. The corrected port pins the exact header and checks unchanged device source before creating a new isolated overlay. The failed preparation and original copy remain documented.

## D: actual native plan-plus-run, against pristine 0.7.0

Native measurement source: `463e786be9a269a9fe7d335049ea5b32a3aea576`. The exact-causal order calculation runs inside the real C++ planner, not Python metadata mutation. Three modes—pristine official wheel, modified source with feature disabled, and candidate enabled—rotate launch order across three fresh processes per mode on one physical GPU per family. All Q/K/V/output/LSE hashes match across modes for each repeated input seed. Actual descriptors and 48/64-KiB launches are verified.

Twelve blocks per process measure real plan generation, sorting, metadata transfer, graph execution and completion. Import, JIT, initial capture and qualification are excluded startup costs. 32 calls means 32 actual attention invocations on fixed buffers, not 32 different model layers. Blocks in separately launched modes are resampled independently; they are not misrepresented as physically matched blocks. All intervals are conditional on the exposed witness, not unseen workload or device-population guarantees.

|GPU|Calls per plan|Pristine/disabled [95%CI]|Pristine/candidate [95%CI]|
|---|---:|---|---|
|NVIDIA GeForce RTX 4090|1|0.999355 [0.996397, 1.002170]|1.008597 [1.003758, 1.014097]|
|NVIDIA GeForce RTX 4090|32|1.000689 [0.999806, 1.002426]|1.011484 [1.007225, 1.019333]|
|NVIDIA GeForce RTX 5090|1|1.000179 [0.998276, 1.002103]|1.017900 [1.016035, 1.019839]|
|NVIDIA GeForce RTX 5090|32|1.000024 [0.999750, 1.000359]|1.020943 [1.019619, 1.021756]|

All within-mode A/A controls satisfy the declared resolution screen. D has 18 completed processes and 1,728 native-cycle measurements. Native C++ helper validation includes 200 differential comparisons and seven invalid-input rejections.

## E: not run, not silently declared complete

The native helper rejects everything outside this exact exposed witness. It is not a safe shape-general policy, threaded API, paged-cache integration, dynamic graph lifecycle or multistream ownership contract. No candidate was inserted into vLLM/SGLang full-model serving. No TTFT/TPOT, HTTP throughput or SLO-goodput improvement is claimed. Fresh-family validation and production callers remain required. All six diagnostic geometries are exposed; none can be renamed an unseen holdout.

## Accounting, failures and reproduction

Six allocated jobs: five completed, one timed out. Allocation including the timeout is 4,921 GPU-seconds (1.366944 GPU-hours), not a price estimate. Completed records across the different phases total 258,624, not 258,624 independent workloads. Shared runtimes, credentials and unrelated jobs were not modified. Preparation failure, timeout, denied extension, missing repeat and profiler-estimator anomaly are retained.

The 380-file evidence archive is SHA256 `4fec24462c6367d0fad4dd3cf74419bb3b7dbadd47cb918acc9637582b0e27c7`. It contains raw timings, qualifications, launch traces, immutable measurement sources, logs, source bindings and accounting. No model weights are required.

```bash
python -m pip install numpy
python -m unittest discover -s research/section6 -p 'test_*.py' -v
python research/section6/native_cycle/test_helper.py --out /tmp/section6-helper.json
python research/section6/reproduce.py --out /tmp/section6-reproduction-unique
```

Use a new output directory. The reproducer checks the archive and every member hash, refuses unsafe paths, re-runs all three analyses on CPU and compares every JSON field. Floating-point last-bit differences, if any, are explicitly listed rather than rounded away to claim byte identity. Exact result-table reproduction is required. A CPU reanalysis is not a new GPU run or independent third-party replication.

Raw reports are under `evidence/available-evidence-v1/`, `evidence/residency-analysis-v1/` and `evidence/native-cycle-analysis-v1/`. See `STATUS.json` for machine-readable completion boundaries. The manuscript/recruiter claim supported now is a scoped source-integrated resource/order intervention with a small native-cycle win, not a universal optimization.
