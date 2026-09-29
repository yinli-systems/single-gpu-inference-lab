# Interpretation and representation limits

The frozen primary test is a controlled permutation contrast, not an optimization leaderboard. Its purpose is to identify which representations discard information, then measure where that discarded information changes actual execution time.

## What the mathematical statement does and does not say
For two states with the same feature vector phi but different target values y_A and y_B, a deterministic predictor f(phi) must have max absolute error at least |y_A-y_B|/2; the equally weighted two-state MAE has the same bound. This follows from the triangle inequality. A dataset's empirical state medians provide empirical targets, not an automatically noise-free or distribution-free latency theorem. Repeated measurements and the order-null control quantify uncertainty.

The statement is about a specified feature map and an equivalence class, not all possible predictors or all workloads. A distribution where pairing is almost determined by the marginals can be well predicted without explicit pairing. This is why the previous shape-campaign functional-form correction must be retained.

The full set of (q,k) pairs is not a proven globally minimal necessary representation. For analytical attention work alone, C=sum(q*k), together with the query-square term, is a compressed sufficient coordinate. Other feature sets could reconstruct C indirectly. For runtime latency, even W may be insufficient because backend execution and ordering also matter. The equal-W and row-order controls are designed to test these limits, not to assume their answer.

The cached-depth multiset is held fixed here. The total per-request KV-length multiset k+q is NOT held fixed. A predictor using those total-length marginals is not automatically covered by this assay's aliasing claim. Published predictors must be classified by their actual features and implementation, not by name or by a simplified substitute.

## Research versus delivery
The new study uses real FlashInfer GPU operations with independent FP32 reference checks. It does not execute a full transformer model. It cannot overturn the previous full-engine-v2 no-promotion result and cannot support TTFT, TPOT, SLO-goodput or production speedup claims.

KernelSight-LM already models per-request/kernel structure and wave quantization; FlashInfer already separates plan/run and supports deployment-matched tuning. Their ideas are prior art, not new inventions here. The intended contribution is the controlled representation test, its exact executable protocol, and positive AND negative boundaries.

References: https://arxiv.org/html/2606.28565v1 ; https://arxiv.org/abs/2501.01005 ; https://flashinfer.ai/2026/09/22/autotuner-v2.html .

## An alternative sufficient coordinate for analytical work

Let L_i=q_i+k_i be total per-request KV length. The elementary identity
`2*C = sum(L_i^2) - sum(q_i^2) - sum(k_i^2)`
means that marginal second moments of q, cached k, and total L, together with sum(q), reconstruct analytical W without storing the full paired set. `representations.py` checks this exactly on every campaign state; 2,000 randomized integer instances are unit-tested. This is algebra, not a novelty claim or GPU latency measurement. It is a necessary control against the overbroad assertion that no marginal-only representation can ever recover coupling.

The registered equal-W pair aliases under these augmented moments but has different paired sets. The row-order control aliases under paired sets but not ordered pairs. Measured latency results determine whether these distinctions matter here; no positive result is assumed.

## Observed environment heterogeneity

Initial formal allocations include RTX4090 processes with drivers580.82.07 and580.105.08, and reported memory totals23028MiB and24564MiB. The registered inference is paired within-process and conditional across three repetitions. Driver/host variation is recorded, not removed selectively, and is not proof of hardware-invariant performance.

## Actual upstream feature-map conformance

Vidur snapshot `abae7f63aa857300f5cdc6f5e0d27860cd24721b`, source blob `a5a96466eb86d94503711afec6d45218bd38d93e`, retains paired request parameters at lines754-780, then constructs the prefill lookup key at lines852-869 as `(sum(rounded cached depth), round(sqrt(sum(chunk^2)))^2)`, with a batch-count multiplier. The preserved marginals therefore alias at THIS lookup, despite the internal request list initially preserving pairs.

`audit_vidur.py` executes those two hash-verified upstream methods on all36states at granularities64/128/256,108calls. Every real-code key matches the reference feature map, and each configuration's states share one key despite differing W. This is source-level conformance, NOT a re-trained Vidur accuracy benchmark and not a claim about all simulator versions. See `vidur-source-audit.json` and the licensed, unmodified reference source.

## Analysis implementation equivalence

The analyzer caches integer weights for the exact registered Random(20260929) hierarchical draws, then computes the same process/block resamples as weighted sums. A regression test compares its95%interval with the literal nested algorithm to11decimal places. This reduces CPU reproduction overhead without changing the resampling design, endpoint, exclusions, or any GPU measurement source. Failed-job markers are additionally rejected even when other files exist.
