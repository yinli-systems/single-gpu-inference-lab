# vLLM request-free KV prefetch primitive

- `0001-request-free-prefetch.patch` — `OffloadingConnector.request_free_prefetch(block_hashes,
  group_idx) -> ACCEPTED | DEFERRED | COMPLETED | UNSUPPORTED`, plus
  `BlockPool.cache_prefetched_blocks()` and three counters. Default off.
- `DESIGN.md` — what already existed (request-free `submit_load`, `unpin_blocks` residency) and
  why the input is a per-block hash sequence rather than offload keys (keys are chunk-strided and
  carry only the chunk's last block hash).

Tests: 12 new unit tests on the repo's offloading-connector harness. Regression check: the full
`tests/v1/kv_connector/unit/offloading_connector/` suite was run on unmodified `main` and on this
change in the same environment; the failure sets are identical (the ~104 failures are version
skew in that CPU-only environment). The first draft read options from `spec.config`, which
crashed 96 existing tests whose spec is a `SimpleNamespace`; that run is what caught it.

Motivation and the no-predictor / no-ranking / reserve-gate design come from research line E
([`benchmarks/results/l20-kv-prefetch-oracle/`](../../benchmarks/results/l20-kv-prefetch-oracle/)).
