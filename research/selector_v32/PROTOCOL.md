# Selector-v3.2 paired fresh qualification

## Why v3.2 exists

The original v3 30-case campaign is exposed development evidence. Its release analyzer incorrectly treated eager device-event time as full recurring eager-call latency, and its within-arm duplicate labels were described too broadly as treatment ABBA. The raw v3 data and HOLD verdict remain immutable. They cannot promote v3.2.

## Frozen selector and fresh geometry

The selector rule is unchanged: strict anti-monotone query/cached-prefix coupling, at least five requests, max cached prefix at least 8192, an unsplit actual plan, and `padded_batch_size >= max(40, floor(2*num_sms/num_kv_heads)+1)`. `manifest.json` contains four new canaries and thirty new release geometries. `FRESHNESS_RECEIPT.json` proves zero exact geometry overlap with v3 and its extension before any v3.2 GPU timing.

## Measurement boundaries

One official FlashInfer 0.7.0 process measures pristine. One guarded candidate binary measures `off`, `cap`, and `guarded` in the same process by changing only `PrefillPlanInfo.resource_cap` after an otherwise identical native plan. Candidate plan vectors must be identical except for that final flag. This removes candidate build and process identity from the off-versus-guarded comparison.

Eager scoring uses recurring `plan + run` wall latency in a repeated window of at least 12 ms, divided by the number of calls. CUDA Graph scoring uses captured Graph1 and Graph16 replay latency. Device-only eager timing is diagnostic and cannot enter the release gate. Release timing is unprofiled; Nsight Compute/CUPTI evidence is mechanism-only.

Each process repeat contains eight blocks. The paired candidate process executes real treatment ABBA/BAAB for `off` versus `guarded`; a separate paired sequence compares `off` versus forced `cap`. Pristine has a within-arm position-balance control. Three process repeats run on one physical GPU per shard, with pristine/paired process order balanced across repeats.

## Correctness and identity

Every arm uses identical Q/K/V inputs and is compared bit-for-bit against an independently executed pristine full output and LSE. Ragged/paged, FP16/BF16, requested auto/unsplit, eager/Graph1/Graph16, complete plan vectors, environment identity, candidate binary digest, window count, GPU UUID, driver, CUDA, Torch, FlashInfer, official overlay and source archive are recorded. A result cannot cross GPU family or execution identity.

## Gate

Promotion is fail-closed. It requires exact numerics; all selected-cell position controls to fit inside a 0.5% 90% equivalence interval; pristine/off disabled-overlay controls to fit inside a 1% 90% equivalence interval; selected and whole-policy point worst at least 0.99; selected conditional joint-min 95% lower bound at least 0.99; Graph16 selected aggregate 95% lower bound above 1; and disabled-overlay point worst at least 0.99. These are conditional claims on frozen devices/geometries, not hardware-population guarantees.

No full HTTP validation may start unless both GPU families pass this fresh gate. The historical 2/432 token divergence remains unresolved and independently blocks default promotion.
