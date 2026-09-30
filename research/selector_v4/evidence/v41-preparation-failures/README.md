# V4.1 campaign preparation failures

1. First attempt: shell redirected prepare output into `root/receipts` before the root existed. Python never executed; no root, job, or case consumption.
2. Second attempt: source ledger and candidate overlay generation completed, then receipt creation failed because the safe manifest intentionally had no sibling-only `release_hash`. No job was submitted and no case was run. The partial root is retained at `/ssd/scxi253/single-gpu-inference-selector-v41-smoke-20260930T202230Z`.

The fix introduces immutable per-family `stage_hashes` for dev/canary/release/stress. It does not change any shape, selector threshold, measurement gate, or freshness classification.
