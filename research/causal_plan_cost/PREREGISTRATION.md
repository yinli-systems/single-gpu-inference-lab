# Causal execution-plan cost study — 2026-09-28

Status: prospective protocol, not a result. AI-assisted implementation and research; author must independently verify and disclose assistance under applicable assessment rules.

## Question and candidate contribution
Can the batch-global split-KV planner explain equal-attention-work latency cliffs, and can a compact execution-plan representation select a lower-cost split without using test timings? This is a targeted contribution hypothesis, not a claim that split-KV, attention tiling, or load balancing is new.

Prior findings motivating this study: an aggregate normalization baseline matched the earlier M2n predictor on existing partition traces; equal-work shapes on FA2 showed hardware-dependent cliffs. Those data are discovery data, not this study's independent confirmation.

## Primary experiment
Run actual FlashInfer FA2 causal attention on RTX 4090 and RTX 5090. Compare stock auto planning, split disabled, and fixed split sizes 512, 1024, 2048, 4096 tokens (ragged page size 1). Keep tensors identical across planner interventions. Use a generated corpus spanning total query budget, cache depth, batch size, and skew, with whole geometry-family train/test separation. Publish exact corpus, hashes, planned cells, failures, and exclusions before analysis. Include prior equal-work and pairing-swap shapes as a separately labelled diagnostic set, never as held-out test data.

Three process/seed replicates per GPU, two head configurations, random interleaving of candidate policies, warmup excluded, CUDA event timing with repeated blocks. Inner invocations are not independent samples. Planning wall time is recorded separately, and no host-planning cost is hidden behind a kernel-only speed claim. Reference: selected FP32 query/head vectors plus full-output comparison with stock auto (atol 0.005, rtol 0.02), with maximum differences reported. Not bitwise equivalence.

## Models and baselines
Use identical fitting data, regularization and grouped validation. Include normalized marginal attention work, exact joint attention work, flexible joint-feature regression, and plan-aware features. Compare stock auto, each fixed policy, a globally best policy chosen on training families only, and per-case oracle (diagnostic only). No retuning on test outcomes. Report prediction MAE/p95, policy regret, per-shape speedup distribution, number of regressions >5%, and planning overhead. The cost model may depend on installed backend version and hardware; no untested hardware transfer claim.

## Decision gates
1. Reproduce equal-work cliffs and intervene on split size; do not call a correlation a causal mechanism.
2. Require at least 20% held-out MAE improvement over the strongest matched non-plan baseline before claiming a predictive advance.
3. Require positive paired family-cluster confidence interval for policy speedup against the training-selected fixed policy, and report >5% regressions. Otherwise do not recommend learned dispatch.
4. Full-engine and request-level SLO improvements are separate experiments. Operator timing does not count as an LLM serving speedup.
5. An unsupported planner mode is reported as unsupported, not replaced silently. Preserve failure accounting.

## Prior art to distinguish
FlashInfer (arXiv:2501.01005) already provides load-balanced plan/run scheduling and split-KV. Sarathi-Serve (OSDI 2024) already studies chunked prefill. Vidur already uses quadratic chunk statistics. FlashInfer documentation provides fixed_split_size/disable_split_kv and states determinism/graph caveats. Candidate novelty must be the validated representation/decision insight, not these components.

## Publication practice
Use this isolated research branch. Commit protocol, source/tests, then all results including negative results. Do not overwrite historical measurements or expose machine credentials, model weights, or datasets. A dissertation prize has no known public numeric metric threshold; no award or SOTA promise is made.
