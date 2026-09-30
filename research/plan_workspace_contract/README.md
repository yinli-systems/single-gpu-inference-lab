# FlashInfer plan-workspace contract audit

The FA2 scheduler materializes plan metadata in a wrapper-owned pinned CPU buffer and enqueues an asynchronous H2D copy. Rewriting that source before the copy completes would satisfy NVIDIA's pinned-memory race condition.

For the non-speculative SGLang path used by the historical Qwen full-serving experiment, public `plan()` first performs blocking device-to-host metadata reads (`qo_indptr.to("cpu")`, KV indptr/last-page reads). A later iteration therefore synchronizes the stream before it can rewrite the pinned plan source. The specialized sync-free `fast_prefill_plan` is installed only for DFLASH target-verify and draft-extend-v2 CUDA-Graph paths in the inspected SGLang version; it was not used by the historical non-speculative Qwen workload.

Accordingly this contract risk is recorded but is not claimed as the cause of the old 2/432 token divergence. A future fast-plan repair should use per-inflight-plan buffers or event-guarded reuse and validate the speculative callers separately. Racecheck cannot diagnose this host-pinned/global-memory race because NVIDIA documents Racecheck as a shared-memory hazard tool.
