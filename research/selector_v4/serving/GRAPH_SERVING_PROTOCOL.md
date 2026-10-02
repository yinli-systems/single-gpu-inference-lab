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
