# Selector-v3.2.1 native-policy measurement qualification

## Version, exposure, and default state

The selector expression and v3.2 manifest are unchanged. The original v3 30-case release is exposed development data. The v3.2 canaries at `672e156` are retained, including job1641803's eager/disabled-overlay HOLD. They cannot qualify this revision. The same four v3.2 canaries are development fixtures; the thirty v3.2 release geometries remain unexecuted at revision freeze. No threshold has been loosened to manufacture a pass.

This revision requires a new source archive, isolated overlays, and new canaries. A pending/running earlier-version job must not be duplicated or have its source modified. Legacy plan entry defaults OFF. The historical 2/432 full-model token divergence independently prevents default promotion.

## Native operation and timing boundary

The isolated FA2 binary exposes an explicit native plan policy: OFF=0, forced cap=1, unchanged guard=2. The existing native plan argument list is retained and forwards to OFF. Python wrapper dispatch uses a private experimental attribute only on standard FA2; unsupported/custom backends fail closed. No Python plan-vector inspection, reconstruction, or round-trip audit occurs inside scored eager calls. This is a research ABI, not a claimed compatible upstream patch.

Eager scores complete recurring `plan + run` wall time. Two unscored sixteen-call pristine pilots set a power-of-two call count, at least16 and at most4096, targeting at least48ms at the faster pilot rate. That count and reference are frozen on first pristine execution for every later arm/process in the same shard. Every scored eager window must be at least12ms; a shorter window fails with its duration retained, rather than being silently rerun or discarded. Graph1 scores sixteen single-replay launches; Graph16 scores one sixteen-call replay. Both are normalized per kernel invocation and remain separate modes.

Each process repeat contains eight blocks. Actual off/guarded and off/cap comparisons use ABBA on even blocks and BAAB on odd blocks inside one candidate process. Three process repeats use one physical GPU per shard; process order is pristine/paired, paired/pristine, pristine/paired. Canary uses one process repeat and two blocks. Only unprofiled timings enter qualification; profiler-controlled clocks/cache/replay are separate diagnostic evidence.

## Correctness and provenance

Every eager, Graph1, and Graph16 check separately poisons output and LSE before execution and requires exact equality against an independent pristine full output/LSE. FP32 reference sampling is retained but not mislabeled as exhaustive FP32 checking. Case-local tensor/graph ownership ends before the next case; garbage collection is outside timing. Device input hashes, references, calibrated counts, raw ordered operation/plan/environment identity payloads and their hashes are retained.

The analyzer verifies artifact hashes, source revision, full expected matrix, exact ABBA roles/positions, actual native policy flags, frozen selector calculation, separate replay checks, calibrated counts, normalization, and hardware/runtime identity. A legacy `candidate_binary_sha256` field denotes the explicitly labeled backend SOURCE-file digest, not proof of a loaded compiled binary. Campaign-private JIT storage and source binding do not substitute for future upstream binary/ABI validation.

## Canary and release gate

Canary requires exact numerics, nonempty selection, selected/whole-policy/disabled-overlay absolute point worst >=0.98, and paired whole-policy point worst >=0.98. Passing canary is not release or serving evidence.

Release preserves the original absolute gates and also requires independent paired gates: selected and whole-policy point worst >=0.99; selected joint-min95% lower bound >=0.99; selected Graph16 aggregate95% lower bound >1; all selected position-control90% intervals inside reciprocal +/-0.5%; disabled-overlay90% intervals inside reciprocal +/-1%, with disabled-overlay point worst >=0.99 and every disabled-overlay +/-1% equivalence check resolved. The selected-position +/-0.5% condition is not silently imposed on unselected positions. Bounds are conditional bootstrap summaries on frozen measured devices/geometries, not population-wide safety guarantees.

`run.sbatch` now calls `authorize_release.py` before release execution. The preflight recomputes both GPU canaries from current-revision raw files and refuses missing, stale, corrupt, or failing evidence; it never submits jobs. A canary pass authorizes only release qualification, not default use or HTTP testing. Full paired HTTP additionally requires both GPU release gates and keeps the historical numerical investigation separate.

## Reproduction

Run `python -m unittest discover -s research/selector_v32 -p 'test_*.py' -v` in a Python environment with NumPy. Use `prepare_guarded.py` only against the hash-bound isolated official-0.7 source. Freeze all source, manifest, protocol, and prior-failure hashes in a new campaign; never overwrite the source of an existing job. Execute canary on each GPU family first and run `analyze.py --stage canary --shards 1` against that campaign. All previous failed and superseded attempts remain available.

Primary references: FlashInfer Autotuner v2 (2026-09-22), https://flashinfer.ai/2026/09/22/autotuner-v2.html ; NVIDIA Nsight Compute Profiling Guide, https://docs.nvidia.com/nsight-compute/ProfilingGuide/ ; NVIDIA CUDA Graph numerical troubleshooting, https://docs.nvidia.com/dl-cuda-graph/troubleshooting/numerical-errors.html . These motivate measurement/lifetime checks; they do not establish this project's performance or diagnose its old token divergences.
