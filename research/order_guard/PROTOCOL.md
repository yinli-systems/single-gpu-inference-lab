# Frozen next-stage protocol: qualification before performance promotion

## Scope and status

Frozen before any new GPU observation. Measurement state: **NOT RUN**. Existing PR #22 states are exposed development data, even when previously called holdout. The new manifest has 57 named entries (48 fresh, eight constructed, one exposed sentinel) and case hash `fb6c14b78cc03f8d02b7b16ff8efa9f382ce313b13e46781f5506a78be7cfa37`.

All source, policy constants, case definitions and analysis boundaries must be committed before GPU qualification. Do not change them to chase a favorable outcome. A change creates a separate labeled development campaign and new unexposed confirmatory suite.

## Candidate and comparators

Primary experimental proposal: `locality_packet8`, compared with identity. Exact-logical `causal_heavy` is a secondary proposal. Keep old heavy-first and request-reverse as comparator policies and identity-repeat as an A/A control. Do not select the best proposal after test results and call it predeclared.

All candidates are descriptor permutations; fixed Q/K/V/output addresses, output destination maps and split/merge scalars remain unchanged. The algorithm does not decide physical CTA issue order, and eight-/32-descriptor groups are not asserted to be hardware waves. Candidate construction cost must be measured separately; the current GPU probe is run-only, NOT native-plan-plus-run performance.

## Qualification stage

Run a canary on one allocated GPU of each family first; only proceed after reading and validating both receipts. Never mutate shared runtime packages, submit duplicate jobs, or take more allocations simply because an earlier call timed out. Capture job ID, GPU UUID, driver, architecture, clock/power settings, co-tenancy and software hashes. The provided runner records available device/process metadata but does not establish exclusive allocation or locked clocks.

Require the source-pinned wrapper and complete descriptor bijection. Independently sampled FP32 references must pass first. Every policy must preserve full output and LSE exactly in eager and fixed-shape graph execution, including NaN-sentinel overwrites. Stop the campaign on a correctness mismatch; preserve the failure. Qualified in one dtype or split regime does not qualify another.

The automatic planner may not select the abstract collision's fixed tile/chunk. Capture actual plan metadata and keep a separate source-level planner test before calling the constructive family a physical GPU witness. Do not force an invalid reusable plan onto a new shape.

## Formal measurements

The runner supports up to eight disjoint case shards and three process repeats; all shards together must exactly cover the case manifest. Parallelize independent allocations, not two experiments on the same GPU. Record overlapping host/node allocation as a possible confound. Do not infer physical GPU count by summing overlapping partitions.

Cross FP16/BF16, auto/unsplit, eager/graph, warm/128-MiB-write conditions, six randomized blocks and all declared policies. The 128-MiB write is a cache-pressure intervention, not proof of full L2 eviction. Each timer measures one real attention invocation; event and wall times remain separate. Per-policy Python metadata-setup time is separately recorded and must not disappear from any eventual application-cost result.

Save every cell, including failures, identity choices and slowdowns. Report per-geometry ratios, worst case, regret against available policies, and A/A behavior. Use process repeats as clusters; keep geometry families together in any split. Bootstrap intervals conditional on the frozen suite do not establish unseen-hardware guarantees. Profiling calls are diagnostic and cannot enter the unprofiled performance table.

## Decision gates

1. Numerical or descriptor-contract failure: HOLD all performance claims for the affected path.
2. Failed A/A controls: keep raw observations, no clean confirmatory claim. Diagnose before fresh preregistered repeats; do not silently omit noisy cells.
3. Any confirmed fresh-case slowdown above 1%: no unconditional default promotion. Retain the case; a guard trained on it is development only.
4. New candidate unavailable, context changed, incomplete evidence, unknown reuse, or uncertain benefit: identity. This is a fallback rule, not evidence that tuning incurred no cost.
5. Full lifecycle cost must include already incurred probes, candidate generation, metadata transfers, validation, dispatch and invalidation. Establish reuse opportunity before expensive probing; retrospective break-even is not a budget bound.
6. No serving promotion before pristine upstream A/B, native caller integration, paged-cache and dynamic-graph coverage, multiple streams/lifetimes, current-version/current-backend qualification, and full-model TTFT/TPOT/goodput measurements with matching outputs.

## Upstream and novelty boundary

Do not submit a default-on upstream patch on the basis of CPU properties or historical selection alone. First establish whether the current supported backend already covers the proposed ordering dimension. FlashInfer 0.7 replication and its deployment-matched autotuner are required comparators, not silently replaced by 0.6.18. A packet constraint, fallback or bandit label alone is not a new research contribution. The useful target is a reproduced dense-FA2 performance reversal with an independently validated, low-overhead remedy and a stronger representation-collision fixture.
