# Full-engine continuation after c052416 — 28 September 2026

## Recovered fact, not an inferred failure
Under the user's renewed instruction, the same Remote Desktop Commander start_process read that previously failed was allowed at 06:38:58 UTC. No alternate channel, account, encoding, approval setting or workaround was used. Job1632883 is FAILED1:0,225 allocated GPU seconds. Its first auto arm returned1 after62.010488s: FlashInfer sampling JIT could not find curand.h. No model performance result exists from that job. Earlier restriction and checkpoint remain in history.

Its paged qualification genuinely passed:576 non-auto layer/transition comparisons,128 selected FP32 vectors, max abs differences0.000244140625 versusauto and0.000056371092796325684 versusFP32; tolerance0.005/0.02. It tested NHD/HND and changed physical pages with unchanged logical geometry. This is paged-kernel evidence, not a complete model result.

Both the existing vLLM runtime archive and existing CUDA13 Python package contain nvidia/cu13/include/curand.h and curand_kernel.h. The configured CUDA_HOME omitted that separate header directory. Add explicit compiler/include and library search paths in an isolated job runtime; retain FlashInfer sampling, precision, math, engine version, weights and frozen native decisions. The header fix is a COMMON prerequisite for all arms, not a candidate optimization. Compile a tiny CUDA header probe before sampling warmup.

## Frozen methods and workloads
Continue the earlier full-engine protocol:Qwen3-4B-Instruct-2507 cached weights,FP16 model/KV,TP1,RTX4090,FlashInfer0.6.18,FA2,eager,vLLM0.29.0. All arms get512MiB scratch,2048KV blocks of16tokens,max1024batchedtokens,max32sequences,prefixcachingdisabled,maxmodellength16384.

Compare stock auto,TRAIN-fixed none,and frozen native_cycle. The fixed comparator was trained using the old cycle+(r-1)*run proxy; it is not claimed empirically optimal in the new serving regime. Every arm uses the same adapter/key checking/decision-cache opportunities. Materialized page plans are rebuilt by the normal engine metadata path. Native weights are not retuned. Unamortized660d128 rejection and128MiB/5090 no-promotion remain unchanged.

Workloads and token IDs are unchanged from the earlier committed engine_screen.workload():burst8requests atstart,promptlength512/1024/2048/4096 repeated,96 outputtokens; staggered12requests at50ms spacing,promptlength256/1024/4096/8192 repeated,48/96outputtokens. PythonRandom seed20261020/20261021,tokenIDs100..9999,temperature0,ignoreEOS. These are SYNTHETIC token loads, not natural-language quality tasks. No workload search based on performance.

## Measurement hardening, common to all arms
Prior code wrote each step to shared disk inside timing and only warmed one256-token request. This continuation buffers observations in bounded process memory and writes files after the workload clock stops. It runs each whole frozen workload once without scoring inside EVERY model process, then clears router statistics before the measured repeat. Same treatment for all arms; no selection from warmup timings. Log warmup and model-load cost separately.

Explicit cumulative outputs are required. Check output-prefix identity, monotonic token counts, exactly planned request IDs/counts/lengths, and actual activation of the qualified prefill adapter. Keep unknown, incomplete, empty, expired and failed results in receipts. Offered TTFT uses the predetermined arrival timestamp, including admission lag. Admitted TTFT and per-request averageTPOT are separate. Token times are engine-observation times, not physical GPU times; p95/p99 across20requests are sparse sample quantiles, not precise tail guarantees. No HTTP/network claim.

Include model config and weight-shard SHA256,backend/router source hashes,driver/GPU UUID,actual engine options,KV block count,workspace,request/token evidence,plan choices and fallback counts. No profiler or output tensor capture inside timing. The common lightweight adapter logging overhead remains charged.

## Staging, statistics and gates
First one finite qualification job executes all3arms in a predetermined order. Do not duplicate1632883:it is terminal and its failure is now understood. Do not launch full repeats until the repaired screen has complete coverage and exact per-request greedy-token equality across arms. If a numerical or interface gate fails, record the failure and diagnose before widening scope.

Only after that screen,run six independent process-order jobs (all6 permutations of3arms),new process perarm,maximum3concurrentGPUs. The screen is excluded from confirmatory timing. Repeats use the same frozen workload; they are execution repeats,not extra independent tasks. Counterbalanced orders address first-process/cache order effects. Post-warmup JIT or logging must not be silently excluded from raw timing.

Primary:paired geometric-mean output-throughput ratio against both controls,computed first withinjob across2workloads,then across6jobs. Nominal95%bootstrap over completejob triplets (20000,seed20260928). Report each workload and everyrepeat. Preserve greedy equality,zero failures,unchanged capacity,strictSLOgoodput(TTFT<=2s,TPOT<=50ms) and lenient5s/100ms. Promotion requires lowerCIbound>1 against both controls and no aggregateSLOgoodput regression; do not switch to a favorable subgroup. No significance assertion when all SLOcounts arezero orsampletoo small.

If fullmodel throughput fails the gate,retain the scoped attention-stack improvement but reject serving promotion. Do not project operator speedup onto model metrics. No awards or globalSOTA are promised.

## Capacity and safety
At06:40UTC the normal scheduler view showed0free4090/5090 and no user jobs. A bounded one-GPU qualification may queue while CPU inspection/tests run; do not purchase resources or cancel others. Check actual state before every submission. If the tool rejects another action,retain its exact response and do not reroute it. Publish code/results through the existing authorized GitHub API branch and verify remote commits.
