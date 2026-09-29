# Decision-Sufficient Execution Contracts

This research line asks a narrower question than "can we predict latency better?":

> **Does the information visible to a runtime contain enough information to choose the right execution action?**

The unit of evidence is an **exact representation collision with an opposite measured decision**. Two states collide when a named feature map produces the same key for both states. They become a decision witness only when the frozen evidence says different actions are better, after the campaign's own A/A resolution gate and a predeclared effect margin.

This directory is deliberately downstream of the existing pairing, descriptor-order, residency, and resource-generalization studies. It does not alter or relabel their evidence.

## Why this is different from another predictor

A lower prediction MAE does not establish that a scheduler or dispatcher can make the right choice. Conversely, a representation can have noticeable timing error but still preserve the identity of the best action. The audit therefore evaluates the information interface directly.

For two indistinguishable states and two actions, if the first state prefers action A by gap `d1` and the second prefers action B by gap `d2`, any policy that sees only the shared representation has minimax regret

```
d1 * d2 / (d1 + d2)
```

under that two-state/two-action problem. This is elementary decision arithmetic, not claimed as a new theorem. The research contribution, if the program succeeds, must come from **real execution states, strong audited representations, controlled interventions, and a low-overhead remedy**.

## Current implementation

- `src/l20_stack/decision_sufficiency.py`
  - exact request-geometry representations;
  - a pinned-Vidur-style prefill lookup coordinate;
  - fail-closed resolved-action records;
  - opposite-action collision detection;
  - normalized two-state minimax-regret lower bounds.
- `audit_resource_summary.py`
  - consumes existing resource-generalization `manifest.json` and `summary.json`;
  - never launches GPU work;
  - only accepts cells whose A/A controls resolve and whose full 95% interval clears the requested margin;
  - audits a hierarchy from aggregate sums through paired geometry plus execution context.
- root `tests/`
  - synthetic exact-collision and exclusion tests run in normal CI.

## What counts as a valid finding

A valid witness must state all of the following:

1. exact feature map and source/version;
2. exact equality of the feature key;
3. fixed action set;
4. matched metric and decision horizon;
5. passing control-resolution rule;
6. opposite resolved action preference;
7. raw evidence provenance;
8. whether the reported regret is absolute or normalized.

An approximate feature match, a point-estimate reversal with an unresolved control, or two different metrics is **not** a witness.

## What this does not claim

- Absence of a collision is not proof that a representation is sufficient.
- A collision for an aggregate representation is not a collision for a richer request-level representation.
- A normalized regret lower bound is not a latency value.
- Post-execution profiler counters are not automatically available to a pre-execution scheduler.
- The resource cap is not a universally better policy.
- This branch is not a production dispatcher and does not promote any frozen experiment.

## Research target

The strongest target is a sequence of independently validated results:

```
representation collision
        |
        v
opposite execution decision
        |
        v
minimal counterexample
        |
        v
identify the missing pre-decision information
        |
        +---- observe it cheaply ----+
        |                            |
        +---- or control execution --+
                                     v
                       lower net decision regret
                                     |
                                     v
                        full-model serving effect
```

If a strong existing representation already separates every important decision under the measured budget, that is a valid negative result and should stop the line rather than trigger ad-hoc feature growth.
