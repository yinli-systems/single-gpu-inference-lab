# Complete stream evidence v2

The original deterministic job 1639849 failed with an incomplete stream in pristine,
second isolated decode-11 request. Its old collector discarded partial response fields.
No claim is made that request ID reuse caused that failure, nor that this was a CUDA failure.

New diagnostic (not a retry replacing old evidence): fixed seed 42, existing upstream
deterministic inference flag in BOTH pristine and cap. Record every SSE frame, completion
count, tokens, finish reason, DONE marker, HTTP status and errors before failing. Keep old
workloads, block zero and output limits. Individual probes have distinct wire IDs;
this changes their lifecycle, is not a proven fix and cannot validate reuse of an old RID.
The observer's artificial per-token arrival fields are explicitly diagnostic, NEVER serving
latency estimates. No performance analysis consumes this directory.

One GPU, pristine and cap once each, maximum40 minutes. Abort on incomplete output,
retain raw frames. Standard job1639848 remains separate, as do old formal outputs and gates.
