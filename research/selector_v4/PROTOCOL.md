# v4 qualification protocol

## Immutable identities

Every record binds GPU name/UUID/SM count, driver, CUDA, Torch, FlashInfer, source archive, overlay hash, kernel-symbol hashes, operation geometry, layout, dtype, requested and actual split, plan vector and execution mode. Eager, Graph1 and Graph16 are separate tactic namespaces.

## Candidate pool

The pool requires causal FA2, head dimensions 128/128, heads 32/8, batch at least 5, maximum cached prefix at least 8192, an unsplit plan, anti-monotone query/cache pairing and at least 2.4 block-waves:

`padded_batch_size >= max(40, ceil(2.4 * num_sms / num_kv_heads))`.

This is not the final selector.

## Calibration

For each environment/operation/mode identity, collect at least three fresh processes and eight paired blocks per process. Exact complete output and LSE parity are mandatory. Null-control 90% CI must remain inside reciprocal ±0.5%. Cap is published only when native/cap point speedup is at least 1.02 and hierarchical-bootstrap 95% lower bound is at least 1.01. Otherwise publish native.

Persistence is atomic and hydrated once; steady-state lookup performs no file I/O. Missing, corrupt, stale or incomplete entries return native.

## Evaluation

Canary first verifies both kernel symbols, native/pristine parity, cap parity, ragged/paged and Graph replay on RTX 4090 and 5090. Only then may release consume the 48 frozen cases. Final release reports oracle regret, chosen-tactic worst case, native fallback equivalence, calibration overhead, cache correctness and full output parity by execution mode and GPU.

Nsight/diagnostic profiling is excluded from release timing. Full HTTP serving requires release PASS and separately preserves the historical token-divergence HOLD.

## JIT cache identity

`FLASHINFER_WORKSPACE_BASE`, not merely `XDG_CACHE_HOME`, is bound to a campaign-private path. Pristine and v4 builds use different workspaces. A binary audit must find native and `ResourceKernel` symbols in both generated CUDA source and the compiled shared object. Shared-home cache reuse invalidates the run and is retained as failed tooling evidence.

## Calibration/evaluation separation

Calibration uses three processes and eight ABBA blocks to freeze a tactic JSON. Evaluation uses three additional processes and eight blocks, and binds the exact decision-file SHA256 into every environment record. Whole-GPU release analysis is recomputed from all raw shard evidence; per-shard summaries cannot authorize promotion.
