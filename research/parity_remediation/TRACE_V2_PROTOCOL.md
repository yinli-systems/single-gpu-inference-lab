# Trace v2: preserve first failure, repair diagnostic transport separately

The first trace job1639851 produced1898 sampler-boundary records in pristine, then
stopped at an incomplete repeated-ID isolated request. No cap trace was executed,
and this data cannot be called a cross-arm comparison. Its failure remains retained.

V2 combines the lossless raw SSE collector and unique isolated-probe wire IDs with
the same immutable sampler observer. Both arms use seed42 and overlap disabled.
One allocatedGPU, pristine then cap,25-minute bound. Upstream deterministic mode
is NOT enabled in this trace, so it remains separate from the deterministic wire job.
All original workload blocks remain. No source or historical artifact is overwritten.

No inference speedup from this synchronous observer. Matching complete histories and
batch signatures is required before comparing logits. Equal sampler-boundary hashes
are not evidence that the original uninstrumented failure is explained. Different
hashes alone are not an identified first operator. Snapshot-count and storage limits
remain8 full-logit files perprocess.
