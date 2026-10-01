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
split decisions and metadata tracking still require execution evidence.

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
