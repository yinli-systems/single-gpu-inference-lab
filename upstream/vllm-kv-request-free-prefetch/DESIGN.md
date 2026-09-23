# `[KV Offload]` request-free prefetch primitive — design notes

Target: the PR zhhangBian invited on vllm-project/vllm#57103 (comment 2026-09-20):
"a minimal request-free CPU→GPU prefetch execution primitive (resolve → capacity
check with reserve → reserve destination → async load → accepted/deferred/completed),
default off, no predictor, no ranking".

Read against upstream main `feb87b19`.

## What already exists (so the PR does not rebuild it)

| piece | where | note |
| --- | --- | --- |
| async CPU→GPU transfer, request-free at the API level | `vllm/v1/kv_offload/base.py::OffloadingWorker.submit_load(job_id, src_spec, dst_spec)` | takes a job id and a GPU destination spec; no request id anywhere |
| completion polling | same class, `get_finished() -> list[TransferResult]` | already per-job |
| CPU-tier residency lookup | `OffloadingManager.lookup(key, req_context)` / `prepare_load` | scheduler-side, keyed on block hash + group |
| blocks that are readable but owned by no request | `BlockPool.unpin_blocks(blocks, on_reuse)` | "join the tail of the free queue and count as free" — exactly the residency class a prefetch needs |
| free-capacity view | `BlockPool` free queue | the capacity check can be local, as the measurements argued |

So the worker side needs **nothing new**. The missing piece is entirely scheduler-side:
a destination reservation for a load that no request owns.

## The primitive

```
prefetch(keys: Sequence[OffloadKey], reserve_blocks: int) -> PrefetchOutcome
```

1. **Resolve** — `OffloadingManager.lookup` per key; drop keys that are not
   CPU-resident. If none remain → `COMPLETED` (nothing to do).
2. **Capacity check** — `free_blocks - reserve_blocks >= len(resolved)`; otherwise
   → `DEFERRED` (no eviction, no ranking: the caller retries or gives up). This is
   the local feasibility rule the #57103 measurements supported.
3. **Reserve destination** — `BlockPool.get_new_blocks(n)`, register each block's
   hash in the prefix-cache map, then hand them straight to the
   `unpin_blocks` residency class so they are last-resort eviction candidates
   while the load is in flight and after it completes.
4. **Submit** — one `submit_load` job per key batch, carried in the connector
   metadata the scheduler already builds.
5. **Complete** — on success the blocks are ordinary cached blocks and the next
   request's prefix lookup hits them; on failure they are freed. Outcome is
   reported as `ACCEPTED` at submit time and `COMPLETED`/`FAILED` on the poll.

Default off: gated behind the existing offloading connector plus an explicit
`enable_request_free_prefetch` flag, so no scheduler path changes when unset.

## Deliberately out of scope for this PR

- **Ingress.** No HTTP/envelope surface — `kv_hints` (#53421 / #53423) is a separate
  open transport PR, and the RFC layers it above this. The primitive is exercised
  from tests and callable by an in-engine policy.
- **Timing / prediction.** The measurements posted on #57103 showed "prefetch at
  pause" keeps 95–100 % of the oracle while fixed-lead and EWMA predictors keep
  5–50 %, so the primitive takes no lead time and no predictor.
- **Ranking / admission utility.** Same measurements: under sustained occupancy
  every admission policy, including one that knows the true resume order, raised
  aggregate resume latency (+10…57 %). Hence a plain feasibility check against a
  reserve, and never evicting to make room.
- **Eviction protection / pinning.** Retention priority is #37003 / #38514's scope.

## Open questions to settle in the PR

1. Where the entry point lives: a method on `OffloadingConnectorScheduler`, or a
   `KVCacheManager` API that the connector calls. The connector is narrower and
   keeps the scheduler core untouched, but the block-pool reservation has to be
   reachable from it.
2. Whether a prefetched-but-unused block should be accounted separately in the
   existing offloading stats (probably yes, one counter for accepted/deferred and
   one for completed/failed, using the spec's `build_metric_definitions` hook).
3. Group/tier handling for multi-group KV (the offload keys carry a group index
   already, so the batch must be grouped by it before submit).
