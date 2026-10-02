# Real Native Graph serving boundary and complete HTTP baseline

This separately bound diagnostic does not modify the running c34d5d9 workflow,
its helper archive, training gates or future holdout cases. Formal Resource HTTP
continues to require all eight exact-source formal kernel stage PASS records.
Native HTTP measurements here are baseline observations, not Resource gains.

Each allocation runs two separate complete-model servers on its actual GPU:
an uninstrumented ordinary Native Graph/overlap server followed by a Native-only
Graph observer. Each server receives the full five-workload, four-block warmup
and scored matrix, then an independent CUPTI profile. One allocation supplies
no independent-process confidence interval. Instrumented timings are invalid;
the uninstrumented baseline alone supplies descriptive HTTP/SLO observations.
Keep all failures and SSE events. Never retry a scored block or drop a request.

The observer binds the actual reviewed SGLang ModelRunner, DecodeCudaGraphRunner,
FullCudaGraphBackend and FlashInfer backend source bytes. It notifies the
metadata adapter before every real decode `load_batch`, ahead of input-buffer
copies and GPU-only metadata glue graph replay. Existing Python-only update
hooks cannot establish this per-replay property on the glue fast path.
Resource remains disabled. Future explicit Resource entry points must adopt
and independently qualify the load notification before claiming this boundary.

Record actual captured graph object IDs and pointers/shapes of input IDs,
positions, sequence lengths, output cache locations and request pool indices.
Hash the actual active GPU values after Native load; hash actual output logits
after replay. Repeated graph keys must retain their original graph and buffer
identities while actual payloads change. Readbacks invalidate performance.
Actual CUDA graph launch events and full-model attention kernels must be
independently found in the profile. Counters alone never establish GPU replay.
Frozen natural token streams of both processes are compared separately; a
new discrepancy is retained and does not become historical parity closure.

The HTTP report independently reconstructs each legacy metric from all raw SSE
events before adding requests/s, logical and uncached input tokens/s, output
tokens/s, admitted-request latency tails, client-arrival ITL tails and 30 fixed
TTFT/TPOT SLO threshold combinations. Zero goodput stays zero. Co-delivered
tokens yield zero client ITLs and are retained. These are not device emission
times. This is a fixed-workload descriptive grid, not an offered-load sweep or
a production-wide Pareto/frontier claim.

First run: complete Qwen2.5-Coder-1.5B on RTX4090 and RTX5090. Extend the independently
bound diagnostic only after these first smoke outcomes have been retained and
audited. No promotion, new kernel holdout consumption or Resource Graph execution
is authorised by this protocol. Historical 2/432 remains unresolved.

## Retained first outcomes and separate cached parity scope (2026-10-02)

Source `51be73c056a2a063d99e998525cfcd0e097ad1fd` ran complete
Qwen2.5-Coder-1.5B BF16 on both cards, jobs1650925/1650926. Each card retained
all20 scored blocks and128 requests/7,936 output tokens in each arm. The
independent CPU/API/GPU correlation audit witnessed34 actual Graph launches and
952 Native attention kernels per card. Each observer recorded2,395 actual loaded
GPU epochs, with unchanged graph/buffer identities and at least three different
GPU payloads in two capture groups. These are real Native serving-boundary
observations. Resource execution inside captured serving Graphs remains unqualified.

RTX4090's natural Native/observer tokens were identical. RTX5090 had4 discrepant
blocks and9 discrepant requests, covering900 differing token positions. The first
5090 outcome remains `HOLD_NATURAL_TOKEN_PARITY`; a later diagnostic cannot replace
it or attribute its cause. Separately, ordinary uninstrumented Native/Native
restart controls passed all20 blocks on both cards in jobs1651180/1651139.
Those controls establish repeatability for their own runs, not historical closure.

The subsequent global deterministic control never scored a block. On4090,
SGLang disabled radix caching for FlashInfer and the first cached warmup failed;
on5090, startup failed compiling DeepGEMM because the CUDA13 namespace header
`cccl/cuda/std/utility` was absent. Original logs and first failed attempts stay
retained. The [official SGLang deterministic-inference support matrix](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/deterministic_inference.mdx) declares
FlashInfer deterministic radix caching unsupported. Do not override its backend
support whitelist or remove the cache-hit requirement to make this protocol pass.

Source `f4113d9732d44ab1d6f62e29ddfaf0119876cdc6` freezes a new independent
`fixed_schedule_observer` preflight before any of its scored measurements. It
keeps the same complete five workloads, four warmup/four scored blocks, input
cells, expected cache hits, sampling and output budgets. Only this separately
scoped parity mode sets client concurrency and resolved server max-running-requests
to1. Both real server configurations must retain radix cache, CUDA Graph and
ordinary overlap, with global deterministic inference OFF. Every output token
and complete top-5 logprob array must match exactly, without tolerance or truncation.
The observed GPU input and profile proofs remain mandatory. This proves only
`fixed_single_request_with_radix_cache_and_cuda_graph`; it establishes neither
production batch invariance nor ordinary concurrent Native/observer parity.

Future full HTTP still requires its own **ordinary concurrent pristine/candidate
natural token parity** as well as the additional cached fixed-schedule parity.
All ordinary performance concurrency, metric floors, complete windows and
sample counts remain frozen. A fixed-schedule PASS cannot bypass a concurrent
parity HOLD, missing Resource kernel proof, or any formal kernel stage.
`continue_graph_http.py` binds a separate committed source packet, waits for
both cached Graph preflights and all eight exact-source formal kernel verdicts,
then may run all72 paired full-model HTTP allocations. It stops on any failure
and retains raw logs, every response/error, decisions, profiles and disassembly.
Even an experimental HTTP PASS leaves default/serving promotion OFF and the
original2/432 divergence unresolved.

The 72-allocation HTTP protocol permits Resource only in eager execution;
`graph_serving_audit.correlated_decode` rejects a Resource kernel with a nonzero
graph ID. Passing this protocol therefore does not qualify Resource inside a
captured SGLang graph. That integration needs its own frozen source, actual
metadata-load boundary, changing GPU input epochs, unchanged graph/buffer
identities, exact outputs, correlated Resource graph launches and stale-epoch
rejection. Fixed-driver Resource evidence and Native serving evidence cannot
jointly substitute for those missing measurements.

The canonical descriptive first-run page is
[`artifact/native-graph-http-20261002-r2/index.html`](../../../artifact/native-graph-http-20261002-r2/index.html).
Its generator independently verifies all499 original archive members and
reconstructs all raw SSE metrics; two builds produced seven identical outputs
and an identical manifest. Earlier preview directories are intermediate artifacts.
