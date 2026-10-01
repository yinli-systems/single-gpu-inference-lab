# V4.2 real-serving integration boundary

This directory prepares the HTTP stage. It does not change the frozen kernel
campaign b769e7c, consume new kernel cases, or authorize serving promotion.
Submit the full-serving qualification only after exact-source dual-GPU canary,
release and stress verdicts pass. The independent HTTP ownership test is a
different diagnostic; its PASS does not satisfy this performance/parity gate.

## Actual production operator coverage

SGLang's ordinary radix-prefix path merges causal ragged current-token attention
with **noncausal paged cached-prefix attention**. Its FlashInfer backend supplies
**page size 1**. Dynamic batching updates the plan and metadata buffers between
model steps. Source inspection establishes these calls; actual workload geometry,
split decisions and metadata tracking are recorded by the Native-only full-model diagnostics. Their timings are invalid and their Resource registries are empty.

Frozen v4.2 identity supports causal attention and paged page16. Current-main
experimental functional tests cover both mask modes at page1 and page16, while its runner
requires a prepared immutable plan and metadata with observable version counters.
Inference tensors without version tracking fall back to native. These scopes must
be independently extended and tested before claiming a production speedup.

## Geometry diagnostic

`geometry_server.py` is an explicit diagnostic launch entry point. Set
`SGI_PREFILL_GEOMETRY_DIR` to a new campaign-owned directory and launch the same
complete SGLang model/HTTP workload through this module. It is installed in
multiprocessing children, records only uncaptured real wrapper calls, and lets the
original forward/plan/run execute. It never selects a resource tactic.

Every distinct record includes actual ordered query/KV lengths, page size,
official plan vector, split flag, dtype/shape/strides, effective forward options,
metadata version availability, GPU UUID, and source hashes. Invalid dimensions or
more than256 geometries fail the diagnostic. CPU readbacks/fsync make its timings
invalid; use a separate, uninstrumented process for performance measurements.

## Integration and acceptance

1. Bind exact FlashInfer/SGLang sources, package/build metadata, compiler/SDK,
   physical GPU UUID, full model weights, tokenizer and workload before execution.
   Keep native and resource modules isolated and preserve the native planner.
2. Qualify page1/noncausal numerical output and LSE, native-after-resource, native
   binary identity, unsupported/split fallback and actual cached geometry.
   Prepared-runner evidence alone never authorizes dynamic metadata reuse.
3. Bind tactic cache entries to the actual operator and execution envelope.
   Validate host-known metadata when the scheduler updates it. Missing or stale
   certificates remain native. Calibrate outside measured requests and CUDA
   capture, freeze decisions before performance, and report coverage/fallbacks.
4. Measure ordinary radix caching, continuous batching and guarded-prefix,
   balanced-prefix, short-prefill, decode and mixed traffic. Use unique request
   IDs, verify cached tokens, rotate arms across at least three independent paired
   allocations, and retain every response/error/timeout and measured block.
5. Report throughput, strict-SLO goodput, TTFT/TPOT p50/p95/p99 and confidence
   intervals, including every workload-level regression. GPU replay speed alone
   cannot qualify full serving. CUDA Graph trials must include actual scheduler
   metadata/input updates and the complete serving-step boundary.
6. Run deterministic full token/logprob parity separately from performance. On a
   mismatch preserve all requests in the batch, scheduler positions, logical and
   physical KV mapping, full relevant activations, logits and sampled tokens at
   and before the first divergence. Instrumented captures are diagnostic only.

The historical2/432 mismatches remain unreplicated and unattributed. Later equal
forced-history replays do not reconstruct missing original pre-divergence state.
Report new replay results separately and retain the historical blocker. Defaults
and serving promotion stay OFF until their own independently bound gates pass.

## Owned metadata snapshot functional evidence

The explicit eager-only `ServingPlanLease` bridge snapshots inference-mode metadata into owned version-tracked tensors, binds the current plan epoch and CUDA stream, and requires notification before untracked writes. Its two exposed BF16 ragged-causal/paged1-noncausal GPU checks passed on RTX4090/RTX5090 in jobs1644779/1644780. Corrupt snapshots, same-valued new plans and explicit invalidation route to the original owner. These functional checks do not authorize full HTTP, replay or performance; the frozen v4.2 canary5090 verdict is HOLD.

## Explicit reusable eager epochs and frozen registry

`ReusableServingPlanLease` requires scheduler release before any native planning or foreign metadata write. It recopies metadata into tracked ownership, verifies actualGPUmetadata through the public same-geometry eager rebind and reuses only a previously prepared module/acceptedcertificate. Rejection isterminal; retiredleases cannot resume. Dual numerical smoke and supplemental real cache-hit/tactic/count witnesses passed on exposed BF16ragged projectionstride6144 and page1noncausal inputs; all9 actualresource calls wereexact. Both4090ragged calibrations wereinconclusive and keptnative. All rawcache/certificate/call records arearchived in evidence/v42-reusable-serving-epochs-managed/. These tests do not qualify Graphmetadata updates, fullHTTP orperformance.

`ServingEpochRegistry` has CPU state and exact-source dual-GPU tests. Register only explicit pre-trained results, thenfreeze. Unknown/inconclusive geometries use theoriginalowner. The scheduler must call before_metadata_update before thefirstmetadata write, includingTriton/fast planningpaths. One publicrebind perkey/epoch isshared acrossmodel layers. Graph/tracing staysnative. Registering a managedtactic cannot force the publicrunner to bypass its independentcertificate/cache checks. Its attemptcounter isnot anactualkernel launchcounter; anindependentHTTPprofile isstillmandatory. The reviewed single-call SGLang prefix hook has passed Native-only full-model checks (112 exact output tokens). Actual packed/tuple KV routing through the adapter and frozen registry passes dual47 functional tests with432 independently witnessed Resource calls and real cache hits. These are minimal-driver composition checks, not Resource-enabled fullHTTP scores.


## Current exact-source prefix boundary and HTTP prerequisites

`prefix_backend_hook.py` checks the complete backend source hash and replaces
exactly one cached-prefix call in `forward_extend`; it preserves the original
future compiler flags. Native owner methods and every other call stay intact.
`prefix_router.py` uses Native's actual packed/tuple KV unpacker, dispatches only
already-bound frozen eager leases, and keeps Graph/tracing/unknown geometry Native.
Independent actual kernel-launch evidence is required in the later HTTP profile.

`training_session.py` permits an explicit premeasurement session only after all
four formal stages on both GPUs and their raw source/evidence ledgers pass.
The first certificate and actual winner are saved before registration. Partial
sessions preserve failure evidence and cannot resume or authorize serving.
No actual HTTP training campaign or server bootstrap is installed by this module.

`http_stream.py` and `http_client.py` collect full SSE token streams and every
failed response. Client receive times include server queues and co-delivered
tokens share a timestamp. A missing terminal DONE, conflicting cumulative prefix,
incomplete output, missing cache hit or nonpositive metric blocks completion.
Raw writes occur outside the measured block. All failed requests remain in the
block archive and a failed block has no success-only throughput summary.
`metric_gate.py` still requires the complete40 metric matrix, joint0.99 floor,
throughput gain and guarded-prefix confidence gain. These preparations do not
close the historical2/432 token differences or grant serving/default promotion.

## Complete HTTP pipeline preparation (not yet executed)

`http_pipeline.py` requires the terminal formal kernel PASS and independently
validates all eight kernel stage analyses before creating an HTTP campaign.
It freezes committed helper blobs, every SGLang source byte, all model shards
and tokenizer bytes, actual normal-package source, SDK and source ledgers.
Four complete models × two GPU families × three stages × three independently
paired allocations require72 allocations and24 stage verdicts, with at most
four allocations in flight and a finite14-day controller deadline. Each HTTP
allocation requests an initial8-hour limit. Existing public development jobs
retain their original limits and are not repurposed as formal stage evidence.

`http_pair.py` runs pristine/candidate, candidate/pristine, pristine/candidate
arm orders for the three allocation indices on the same physical GPU and CPU
affinity with separate JIT caches. Startup, explicit calibration, full frozen
warmup, complete scored requests and independent profiling are separate phases.
No failed request, allocation or calibration is automatically retried.

`http_analysis.py` reconstructs full tokens and timing metrics from every
original SSE event, exact Native/pristine SASS from the archived disassembly,
and every opted-in paged-prefix Resource instruction. It checks full natural
token parity, separately checks complete token/top5 logprobs in deterministic
processes, requires actual Resource launches and65536-byte launch allocation
in independent CUPTI traces, and records real idle metadata/Graph boundaries.
Performance clocks must have at least10 actual scored samples at≥90%GPU
utilization per arm, with p95/p05≤1.05. Warmup clocks cannot satisfy this gate.
All40 workload/metric points and their joint95%LCB retain the0.99 floor.

`http_archive.py` refuses live allocations and unresolved submission intents.
After every known allocation stops it preserves all partial/failed HTTP arms,
full model/source bindings, raw SASS/profile/telemetry, and managed decision
caches. Only rebuildable top-level JIT/SDK data and Python bytecode are omitted.
Every archived member is independently hashed after writing the archive.

Exact d14c22b preparation passed180 Paracloud CPU tests with10 GPU skips.
Its304 original source/log/XML members are archived in
`evidence/v42-http-preparation-cpu-d14c22b/`. The later archive helper has3
local CPU tests. None of these preparation tests is actual full-model Resource
HTTP evidence. Historical2/432 token differences remain unresolved; even a
later complete HTTP PASS will leave default activation disabled pending closure.
