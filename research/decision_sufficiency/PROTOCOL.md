# Frozen protocol for the next decision-sufficiency stage

Prepared 2026-09-30 before any **new** GPU experiment for this research question.

## Separation from active campaigns

The current resource-generalization confirmatory campaign remains frozen and is not modified, resampled, cancelled, or extended because of this work. Its existing development and confirmatory results become **exposed evidence** for this line as soon as they are inspected.

This stage begins with offline analysis only. Any later GPU confirmation requires a newly frozen, non-overlapping suite.

## Primary question

For a named runtime information interface `phi(state)`, can two real execution states have exactly the same visible representation but require opposite choices between two already-qualified execution actions?

The first action set is intentionally bounded:

```
A = {pristine, cap}
```

No learned multi-action controller is introduced before this binary question is resolved.

## Representation ladder

Audit in increasing information order:

1. aggregate request count / total query / total cached KV;
2. exact analytical attention work;
3. pinned-Vidur-style prefill lookup coordinate;
4. augmented marginal moments;
5. full paired request geometry;
6. paired geometry + dtype;
7. paired geometry + dtype + layout;
8. paired geometry + dtype + layout + split mode.

Each representation is executable code. Feature equality must be exact. Quantized or approximate collisions must be registered as a separate study.

## Decision qualification

For a resource-generalization cell with reported ratio

```
r = cost(pristine) / cost(cap)
```

and a requested relative margin `m=0.01`:

- `cap` is resolved only if the cell's A/A controls resolve and the entire 95% ratio interval is above `1+m`;
- `pristine` is resolved only if the controls resolve and the entire interval is below `1/(1+m)`;
- all other cells are unresolved and cannot create a witness.

The first offline audit uses `calls=16` and `run_device_us` so single-call timer resolution and end-to-end cycle overhead are not silently mixed into the primary collision search. Other horizons are reported separately.

## Primary endpoint

For each named representation:

- number of resolved states by preferred action;
- number of exact opposite-action collision pairs;
- strongest normalized two-state minimax-regret lower bound;
- complete witness list with execution context.

The normalized bound uses pristine cost = 1 independently in each state. It is dimensionless and cannot be reported as microseconds.

## Promotion gate

A representation-insufficiency claim requires at least one exact collision whose two decisions both pass the qualification rule.

A useful systems result additionally requires a **new confirmatory experiment** showing the collision or its predicted consequence on an unexposed geometry/context family.

No automatic feature is added after seeing a witness and then evaluated on the same evidence as if it were held out.

## Counterexample minimization

If an exact collision is found, minimization is structural, not statistical cherry-picking:

1. freeze the witness and raw evidence;
2. derive which dimensions can be removed while preserving the feature equality by construction;
3. generate smaller candidate states without timing them;
4. register the candidate list and stopping rule;
5. only then measure the candidates.

The original witness remains visible even if a smaller one is later found.

## Remedy comparison

Only after a confirmed collision, compare three classes under a common cost ledger:

- **observe more**: expose the smallest pre-decision feature that separates the witness;
- **control more**: apply a qualified execution constraint that contracts the harmful state variation;
- **baseline**: preserve the native path.

All probe, plan, feature-extraction, metadata-transfer, invalidation, dispatch and fallback costs remain charged. A fallback after a failed probe does not erase the probe cost.

## Full-system requirement

A kernel-level success is not a serving claim. A final promotion requires:

- pristine current-version baseline;
- matching output tokens;
- actual candidate-path hit evidence;
- full model;
- real request path;
- TTFT, TPOT, throughput and joint SLO-goodput;
- request failures and unfinished requests retained;
- all setup / probing / dispatch overhead included according to the declared reuse horizon.

## Stop conditions

Stop and retain the negative result if:

- strong paired/request-level representations show no qualified opposite-action collision in the frozen search space;
- candidate information is unavailable before the decision;
- measurement resolution cannot support the requested margin;
- a cheaper fixed/native policy matches the proposed method after full costs;
- or full-model serving shows no material downstream benefit.

The objective is not to force a positive optimization result. It is to identify the smallest execution-state interface that is actually sufficient for a useful decision, or establish where that goal fails.
