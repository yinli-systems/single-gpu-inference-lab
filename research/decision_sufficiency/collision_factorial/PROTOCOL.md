# Fresh collision-factorial protocol

Prepared and committed **before any GPU timing from this campaign**.

## Question

Can two chunked-prefill states that are exactly indistinguishable under a strong
cost representation require opposite descriptor-order decisions, and does a
resource-residency intervention contract that decision difference?

This is a mechanism/decision experiment, not a serving benchmark.

## Frozen collision construction

The manifest contains one qualification-only pair and five formal A/B pairs.
Each pair is constructed on CPU before timing and is required to have exactly
the same:

- request count;
- query-length multiset;
- cached-KV-length multiset;
- total query and total cached KV;
- exact causal analytical attention work;
- pinned-Vidur-style 64-token KV lookup coordinate;
- augmented marginal moments, including the total-length square moment.

The paired request geometry must differ.

Formal request counts are 6, 10 and 14. The complete geometries are different
from all checked-in plan-order cases and the resource-generalization manifest.
The historical discovery-n8 / eqC-a,b witness is not part of this suite.

No case is added, removed or resized after GPU results are inspected.

## Factorial interventions

Two descriptor-order actions:

- identity: native descriptor order;
- heavy_first: descending estimated visible-key work using the already
  implemented and audited plan-order helper.

identity_repeat is an A/A control and is never treated as an optimization.

Two resource states:

- pristine: official source-pinned FlashInfer 0.7.0;
- cap: the previously qualified resource-only overlay that raises the
  eligible 48-KiB dynamic shared-memory launch reservation to 64 KiB.

The cap overlay does not change request order, device-kernel source or the
mathematical attention operation. Its source/header binding must be checked
again inside every new process. The previously measured resource campaign is
not reused as a timing sample here.

## Fixed execution matrix

Formal matrix:

- RTX 4090 and RTX 5090 are separate hardware scopes;
- FlashInfer 0.7.0, FA2 ragged causal attention;
- 32 query heads / 8 KV heads / head dimension 128;
- FP16 and BF16;
- auto and forced-unsplit planner modes;
- eager single-call and fixed-shape CUDA-graph 16-call regimes;
- four fresh process repetitions per GPU family;
- twelve randomized matched timing blocks per process.

Each Slurm repetition runs both resource states on the same allocated GPU.
Resource-state order is balanced across four repetitions:

    rep 0: pristine -> cap
    rep 1: cap -> pristine
    rep 2: cap -> pristine
    rep 3: pristine -> cap

A separate one-process canary per GPU family uses only the qualification pair
and two timing blocks. Canary timing is infrastructure evidence only and
cannot select cases, thresholds or policies.

No whole-node exclusivity or locked-clock claim is made. GPU UUID, driver,
power limit, clocks/utilization telemetry and co-resident compute processes are
recorded.

## Correctness and ownership

Every state/resource/dtype/split combination must:

1. validate the actual FlashInfer plan descriptor map;
2. check selected rows against independent FP32 causal attention;
3. establish an identity baseline;
4. preserve full output and LSE bit-for-bit under identity_repeat and
   heavy_first, in eager and graph replay;
5. preserve Q/K/V/output/workspace addresses during descriptor interventions;
6. verify graph output equals eager output exactly;
7. verify resource-source bindings before timing.

A correctness, source-binding or descriptor-contract failure stops that process
and retains partial evidence.

## Timing protocol

For every block and execution mode, two separate matched comparisons are run:

- identity vs heavy-first;
- identity vs identity-repeat.

Placement alternates ABBA/BAAB by block. Candidate construction and descriptor
application are timed separately and excluded from the kernel-only primary
endpoint. Before each scored observation the same fixed warm-up rule is used.
No timing row is removed post hoc.

CUDA-event device time and synchronized wall time are both retained. Graph-16
device time is divided by 16 and is the **primary mechanism endpoint** because
the prior studies show materially better timing resolution there. Eager-one is
secondary.

This does not make graph-16 equivalent to 16 transformer layers or 16
independent requests.

## Statistical analysis fixed before timing

Formal process/block intervals use a hierarchical bootstrap:

- resample four process repetitions with replacement;
- within each sampled process, resample twelve matched blocks;
- 10,000 draws;
- fixed RNG seed in the checked-in analyzer.

For each state/resource/context:

- A/A control is qualified only when its 90% native/control ratio interval is
  fully inside [1/1.005, 1.005];
- heavy_first is resolved better than native only when the whole 95% ratio
  interval is above 1.01;
- native is resolved better only when the whole interval is below 1/1.01;
- all other cases are unresolved.

Cellwise intervals are conditional on these fixed GPUs/geometries and are not a
familywise or hardware-population guarantee.

## Primary endpoint

For graph16, count formal A/B pairs where, under **pristine** resource state:

1. both A and B A/A controls resolve;
2. A and B have exact equality under analytical work and augmented marginal
   moments by manifest construction;
3. A and B require opposite resolved order actions.

Also report the conservative normalized two-state minimax-regret bound using the
least-favorable endpoint of each 95% ratio interval.

A positive result is an independently timed confirmation that the named
representation is decision-insufficient for this action set and hardware scope.
It is not a theorem about all cost models.

## Resource-interaction endpoint

For every fixed pair/context, define

    D_resource = log(native/heavy on A) - log(native/heavy on B)

and report

    C = |D_cap| - |D_pristine|.

The resource intervention **contracts the order-decision contrast** only when
the upper endpoint of the 95% bootstrap interval for C is below zero.
Expansion requires the lower endpoint above zero; otherwise the interaction is
unresolved.

This endpoint is independent of whether either individual order action crosses
the 1% decision threshold.

## Representation ladder

The final analysis reports collision counts for:

1. aggregate sums;
2. exact analytical work;
3. pinned-Vidur-style key;
4. augmented marginal moments;
5. full paired request geometry.

Execution context (GPU, dtype, split and eager/graph regime) is never silently
discarded when constructing a decision witness.

Absence of a collision at a richer representation is **not proof of
sufficiency** outside this frozen suite.

## Promotion boundaries

This campaign cannot by itself justify:

- a default-on descriptor reorder;
- a default-on 64-KiB resource cap;
- a learned dispatcher;
- an end-to-end serving speedup;
- an upstream performance claim.

Any later dispatcher must charge feature extraction, candidate construction,
planning, metadata transfer, dispatch, invalidation, failed probes and fallback.
Full serving promotion separately requires a pristine current-version engine,
matching output tokens, verified candidate-path hits, TTFT, TPOT, throughput and
joint SLO-goodput.

## Stop conditions

Retain the result and stop escalation if:

- no fresh exact-W/moment collision produces an opposite resolved decision;
- A/A resolution is inadequate;
- the cap interaction is inconsistent or expands the contrast;
- source or numerical qualification fails;
- or a richer pre-decision representation already separates all confirmed
  witnesses at lower cost than any proposed control mechanism.

The experiment is allowed to fail. Cases, thresholds and endpoints remain
frozen either way.
