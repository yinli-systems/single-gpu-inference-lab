# Result: descriptor order matters, but the candidate is not a safe default

Completed 29 September 2026. This work contains an actual source-integrated FlashInfer FA2 patch, real multi-GPU measurements, complete negative controls, and offline reproduction. It is not an upstream merge, a universal scheduling algorithm or a full-model serving speedup.

## What was isolated

The earlier row-permutation assay changed both request order and physical concatenation. Here Q/K/V, output buffers and their addresses are fixed. The FA2 plan's dimensions, split configuration, output slots and merge destinations are fixed. Only the complete bijection of (request index, query tile, KV tile) work descriptors changes. This narrows an important confound: runtime differences cannot be attributed to repacking or moving those input/output tensors in this experiment.

The intervention does not isolate SM issue order from memory-access order, locality or hardware scheduling. No profiler timing is used as benchmark timing. It does not prove that an exact hardware mechanism has been fully identified.

## Completed evidence

Two one-GPU canaries passed before six formal runs (three processes per GPU family). The formal matrix contains 30 states: 14 prior discovery states and 16 states from eight newly fixed held-out geometries. It crosses two dtypes, two split settings, five policies, two execution modes and six randomized blocks.

- 43,200 formal timing records and 3,600 exact-output qualification records.
- 480 canary timing records, retained separately and not used to select a winning policy.
- Two actual source-compiled native integration pilots: 3,456 total cycle measurements and 288 qualification records.
- All ten jobs completed with exit 0:0. Allocated GPU time, including canaries: 1.412778 hours. These repetitions are not 47,136 independent workloads.
- GPU scope: RTX 4090/5090, explicit FA2, FlashInfer0.6.18,32 query/8 KV heads,D128,FP16/BF16,512MiB scratch. No paged cache, full-model, HTTP, TTFT/TPOT or SLO-goodput qualification.

## Primary predeclared heavy-first policy

Ratios are native time divided by candidate time; above one is favorable. This is RUN ONLY: the Python diagnostic metadata-copy/validation cost is separately recorded and excluded from the run timer. Primary results use all held-out states, not a favorable subset.

|GPU|Held-out CUDA-graph ratio|Conditional95% interval|A/A equivalence failures, all formal cells|
|---|---:|---|---:|
|NVIDIA GeForce RTX 4090|1.044987|[1.042904, 1.046279]|1|
|NVIDIA GeForce RTX 5090|1.023595|[1.022662, 1.024233]|11|

Intervals describe three process repeats and six matched blocks conditional on this frozen suite. They are not uncertainty over unseen workloads, GPU populations or a multiple-comparison-adjusted claim. A/A is the native descriptor order repeated, not a different policy.

All CUDA-graph A/A controls pass. The4090 failure is a discovery/eager cell; the5090 has ten discovery/eager failures and one holdout/eager failure. These are NOT silently excluded, and the protocol's global clean-control requirement is not met. Failure to demonstrate equivalence is not automatically proof of a systematic difference.

### The important regression

For RTX5090, holdout-01/opposite/unsplit regresses in both dtypes. Its graph ratio is about0.7052–0.7057: the candidate takes roughly42% longer than native. These cells' A/A controls pass. This is a material adverse result, not merely a noisy baseline. Of64 held-out graph cells, two exceed5% slowdown on5090; none do on4090, whose worst cell ratio is about0.9708. The mean win is therefore not a safe unconditional dispatch rule.

No post-hoc switch to request-reverse or interleave is made even where another policy wins. No guard fitted to these held-out failures is claimed as validated.

## Actual native plan-plus-run pilot

A default-off helper was inserted into the real FA2 plan construction source and compiled in a private package overlay. Native descriptor hashes match the frozen Python policy. The shared runtime remains unchanged. Full output and LSE match bitwise in the tested eager and fixed-shape graph paths; selected independentFP32 vectors check semantics.

The native cycle timer includes planning, sorting, metadata transfer, actual graph replay and completion. One process perGPU was used: these are descriptive pilot results, without confirmatory confidence intervals. Reuse36 means36 actual attention calls per plan, not dividing a CPU measurement by36.

|GPU|Held-out plan+1-call ratio|Held-out plan+36-calls ratio|
|---|---:|---:|
|NVIDIA GeForce RTX 4090|1.004725|1.004351|
|NVIDIA GeForce RTX 5090|1.026942|1.028006|

The4090 native held-out effect is under0.5%; the5090 pilot shows approximately2.7–2.8%. The native pilot covers only the SAME-pairing side of the eight held-out geometries, whereas the formal run-only matrix covers both sides. It therefore does not cover the severe opposite-pairing regression above. Different processes, subsets and timing contexts mean the gap between these two studies cannot be attributed solely to planning overhead.

The native identity comparator is the disabled path of the same experimental compiled source, including its environment check. This is not a pristine upstream-main serving A/B. The environment flag is experimental, not an endorsed public API; dynamic/padded CUDA-graph plans and paged units are rejected when enabled. A thread-safe public interface, all callers, paged transitions and complete-engine qualification remain undone.

## Real upstream context

vLLM PR52061 is an open native forward-pass metrics proposal; issue55259 is its related self-benchmark RFC, not a deployed interface on our installed vLLM. Exact geometry fixtures and measurement-boundary tests are plausible complementary contributions, not a reason to duplicate its collector. FlashInfer PR5687 already studies scheduling in another SM90 VSA path, so longest-work-first or load balancing is not claimed as invented here. FlashInfer PR5176 already merged its graph-mask fix; this work does not claim that fix.

The reviewed FA2 planner helper is text-identical in0.6.18,0.7.0 and pinnedmain10c80d75ba63a5b574e619d5167683e2a1d1a625. This is source-applicability evidence only, not whole-runtime or0.7 GPU replication. See UPSTREAM_RESEARCH.md and upstream-audit/source-audit.json.

## Reproduction and release boundary

The archive contains140 hashed files and has SHA25670ba29c0e70e1178cc296996bfbc98cefbc52953744d727aabb4545f0c96410b. It retains completed measurements, immutable measurement source, logs, qualifications and accounting; no weights or model downloads are required for CPU analysis.

A fresh Mac extraction and CPU-only analysis regenerated both result tables byte for byte. The two JSON summaries differ at only THREE floating-point values, each by one last-bit-scale difference (approximately1e-16); all keys, hashes, counts and other values agree. The exact differences are retained in offline-reproduction.json. JSON byte identity is NOT claimed. This is same-workflow second-environment analysis reproduction, not independent third-party GPU replication.

## Decision

**NOT PROMOTED as a default optimization or serving change.** Keep the useful controlled mechanism experiment, the working default-off C++ source prototype, the exact-output evidence and all regressions. Before any deployment claim, freeze a separate guarded-policy protocol, include the opposite-pairing failures, test full plan/run overhead against pristine upstream, and qualify actual engine callers. Do not tune on these holdouts and call the retest unseen data. The previous full-engine-v2 no-promotion decision is unchanged.
