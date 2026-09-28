# Matched reuse gate after 660d128 — 2026-09-28

Prospective protocol. Preserve the rejected unamortized dispatcher and all earlier negative results. AI-assisted work, no prize or novel-reuse claim.

## Recovery and immutable inputs
Start at public milestone 660d12838302c6d00666496c5db917cb3a69b6f4. Recovered wave-threshold-v1 has six completed 144-record runs; native-policy-v1 has only two completed six-cell canaries, not a full test. Do not duplicate these runs. Independent-audit frozen selection SHA256 is 6d99f3d0699afafe3977d5a60c6fa8407f0a33296b9d433f4a3a6a5ec30223c5. The earlier execution session's analysis selection d41c87f9e2f1c8eeac8ab411f63925e0eb2781f0229f8d7974ae6b6463a81f1c has identical individual fitted-model hashes and fixed policies (checked before this protocol). Reuse that session's validated CPython native bridge without refitting. Do not change model coefficients from test observations.

## Amortization is a hypothesis, not measured speed
For a valid reuse interval of r layer calls, let deltaP=P_candidate-P_baseline, C be incremental selector/lookup/guard overhead, and deltaU=U_baseline-U_candidate. A linear diagnostic model predicts a benefit iff r*deltaU > C+deltaP. If deltaU>0, the first positive integer is max(1,floor((C+deltaP)/deltaU)+1). If deltaU<=0 there need not be an eventual break-even; report an empty or finite profitable interval rather than forcing a positive crossover. Existing event and plan measurements supply only a diagnostic approximation. A CPU median divided by layer count is NOT experimental evidence.

The true test times lookup/selection, resource guard, public plan, every run, and device completion together on the GPU worker. All baselines receive the same plan reuse, identical cache capacity, metadata, tensors and memory ceiling. Warmup, compilation and tensor allocation are outside steady-state timing and reported separately.

## Scope and realistic constraints
Pin installed FlashInfer0.6.18, FA2 ragged causal attention, FP16, D128, NHD, no positional transform/window. Eager is the first supported boundary; no inference to CUDA Graph results. Official Qwen3-4B and Qwen3-8B configs both specify 36 layers, Hq32/Hkv8/D128. Use 36 distinct layer Q/K/V allocations, not repeated calls on a single cached tensor. This is attention-stack replay with real architecture dimensions, NOT either complete model. Original BF16 model configs do not turn an FP16 assay into a BF16/model validation.

Scratch ceilings:128MiB and512MiB, plus retained metadata. A plan over budget falls back to stock auto, then unsplit if necessary; never silently enlarge the ceiling. Every arm uses the same serial scratch reservation. Fixed policies are additionally selected per GPU/head/budget from TRAIN records, applying this exact fallback. Report actual peak allocation and layer KV bytes. Do not infer unchanged serving capacity without weights and scheduler integration.

Cache identity includes ordered q/k lengths, heads,D,dtype,layout,causal/window/position mode,GPU/runtime/backend,execution mode,scratch ceiling. Cache policy decisions only in a bounded LRU; retain at most the currently active materialized plan. Reuse it only while its full key and lifetime remain valid. Tests must cover every key field, invalidation, eviction, rejection before execution, and new data with unchanged geometry. Changing geometry does not make all subsequent layers invalid: recompute once per compatible group.

## Arms and rollout stages
Arms:stock auto; TRAIN-selected budget/head fixed; frozen native cycle-cost selector PRIMARY; frozen native run-cost selector SECONDARY. No choosing the better selector after seeing test. A real per-shape empirical tuner may be included as a separately labelled strong engineering control with independently collected calibration and separately charged cold tuning; it is not the official autotuner unless that concrete path actually invokes the official runner.

First recover existing canary/wave evidence and run CPU contracts. Then a new TRAIN/discovery-only correctness/reuse screen at r=1 and36. On a valid screen, freeze and measure new combinations Q={640,1280,2560}, K={2048,4096,8192}, n={2,8}, with deterministic asymmetric partitions; all are independent from old fitting/testing/diagnostic exact geometries. Nine whole query/cache families are statistical clusters. Generate no outcomes before publishing the corpus hash.

For each interval perform actual r calls with r in {1,2,4,8,16,36}; intermediate values are sensitivity,36 is the architecture-matched endpoint. Test unchanged geometry, alternating A/B, and eviction sequence A/B/C/A with capacity2; all arms see identical sequences. Changed states are defined from metadata without inspecting timing. Three independent process/seeds per GPU (4090/5090), randomized forward/reverse arm blocks. Do not treat inner calls/layers as independent samples.

## Gates and permitted conclusions
Primary promotion: at36 layers, direct wall-clock family-macro fixed/native_cycle ratio has nominal paired family-bootstrap95% lower bound>1, no correctness/cache-validity failure, all memory ceilings respected; report every >5% regression. Claim portability only where each GPU passes separately. A valid negative canary can stop an unpromising route but is not independent generalization. If fresh tests fail, reject that deployment route; do not retune these tests. Eager gains do not authorize graph-mode gains.

Only after a scoped positive direct integrated gate perform a separately registered full-vLLM test with matched resource capacity, native baseline reuse, model output checks, throughput,TTFT/TPOT,p95/p99,and SLO goodput. No operator-to-system projection.

## Prior art / applicability to verify
Official2026-09-22 posts https://flashinfer.ai/2026/09/22/autotuner-v2.html and https://flashinfer.ai/2026/09/22/flashinfer-v07.html already cover deployment-matched eager/graph tuning and persistent valid winner reuse. https://docs.flashinfer.ai/api/pod.html already documents cross-layer structures. Inspect the pinned ragged FA2 implementation: installing0.7 or entering a tuning context is not proof this0.6.18 wrapper has an autotuner runner. Preserve version-versus-method separation; reuse/keying/tiling themselves are not novelty.

## Safety and publication
Use allowed scoped audit/implementation/tests and the existing explicit GitHub publication flow. If an action is rejected, retain its exact response and do not reroute the same action through another tool/account/encoding. No new purchases, cancellation of healthy work, or alteration of shared environments. Record actual job IDs; never fabricate a launch from a script alone.
