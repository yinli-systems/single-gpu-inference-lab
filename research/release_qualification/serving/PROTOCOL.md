# Multi-model full-serving release protocol

Four complete local models are exercised through real SGLang HTTP streaming on allocated RTX 4090 GPUs. Every request ID is unique. Radix prefix caching remains enabled and every prefix workload must report at least 8192 cached tokens.

Performance uses the ordinary serving configuration, three paired physical allocations per model, four measured blocks, rotated mode order and five workloads: guarded long-prefix prefill, balanced long-prefix control, ordinary short prefill, decode and mixed. Correctness is a separate upstream deterministic configuration with logprob collection; its timing is never mixed into performance results.

Profiler evidence is collected outside measured blocks. Guarded mode must use 64-KiB launches on the eligible long-prefix workload and preserve native launches on the balanced control. `cap` is retained as an unconditional oracle, not a release candidate.

No aggregate improvement can hide a request failure, prefix-cache miss, output mismatch in the deterministic arm, unresolved source binding, or a workload-level regression exceeding the declared safety gate.
