# Preserved pre-GPU preparation failure

The first preparation attempt at commit 59729c7 stopped at an exact source-anchor assertion before patching, compiling, allocating any residency job, or measuring a candidate. FlashInfer 0.7 moved the ragged dispatch body into `BatchPrefillWithRaggedKVCacheDispatchedImpl`; the old boundary unintentionally selected the following paged body. No result is attributed to this attempt.

The corrective port pins the actual 0.7 wheel header SHA256 e66bce2652c0a3c5b54f88510aff2b4b48ff6a52c4ee23fa97f91fa06a935b87, isolates the ragged Impl body only, validates both unique anchors and unchanged device-kernel text before creating a new overlay. The old unmodified overlay and source are retained. Candidate semantics and diagnostic cases are unchanged; no candidate GPU results existed at the correction point.

An unrelated attempt to extend the already-running 4090 A/C job from 35 to45 minutes was denied by Slurm. No alternate identity or privileged path was attempted. The original job and its frozen data remain intact; completion/timeout must be reported from receipts.
