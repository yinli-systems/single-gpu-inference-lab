# Selector-v3 fresh qualification protocol

## Frozen decision

Selector-v3 is a plan-time tactic selector. It enables the 64-KiB resource cap only when all of the following hold:

1. at least five requests and a cached prefix of at least 8192 tokens;
2. every comparable query/cached-prefix pair is anti-monotone (no concordant pair), with at least `batch_size-1` strict discordant pairs;
3. the actual FlashInfer plan is unsplit;
4. `padded_batch_size >= max(40, floor(2*num_sms/num_kv_heads)+1)`.

It does not use synthetic regime labels. The decision is serialized in `PrefillPlanInfo` so CUDA Graph replay cannot change tactic without a new plan.

## Fresh evidence boundary

The 30 release cases in `manifest.json` were created after v1/v2 results were exposed. They include strict-opposite descriptor boundaries, one-swap near-counterexamples with identical marginals, same-paired, mixed and tied-prefix cases. No v1/v2 case is renamed or reused as fresh evidence.

Each GPU family runs independent processes covering pristine/off/cap/guarded, FP16/BF16, ragged/paged, requested auto/unsplit, and three execution identities: eager, Graph1 and Graph16. Three process repeats and eight randomized ABBA/BAAB blocks are required. Exact full output and LSE equality to an independently executed pristine arm is mandatory.

## Tactic identity

There is no persistent winner cache in this prototype. Every timing and qualification receipt nevertheless records the identity required by deployment-matched tuning:

- GPU name/UUID/SM count, driver, CUDA, Torch, FlashInfer, Python, cuDNN, nvcc, complete modified-backend digest, immutable official-overlay digest and selector version;
- measurement policy and eager/Graph1/Graph16 execution mode;
- layout, dtype, actual split state, heads, head dimensions, page size;
- ordered query/cached-prefix geometry, ordered-pair hash and complete plan vector.

Release timings are unprofiled; Nsight Compute/CUPTI mechanism profiles are separate diagnostic artifacts and cannot enter the release timing matrix.

A future persistent cache must use this complete identity and reject drift. It must not reuse an eager result for CUDA Graph or a result from another driver/backend build.

## Promotion gate

Release remains HOLD unless exact numerics pass and, for every selected eager/Graph1/Graph16 cell, matching controls resolve 1%, point worst and conditional joint-min 95% lower bound are at least 0.99. Graph16 selected aggregate 95% lower bound must exceed 1.0. Whole-policy and disabled-overlay point worst must also be at least 0.99. These are conditional experiment gates, not population guarantees.
