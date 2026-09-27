# Frozen analysis addendum — 2026-09-28

This addendum is written before the first resumed canary or full timing run. Preserve the original preregistration and known negative results. All work is AI-assisted; no award or novelty guarantee.

## Recovery
Remote branch: research/causal-plan-cost-20260928 at a935d1e2d7d9d5c4a216f1b0933e4f2413c82ae5. Local isolated worktree: sgi-causal-plan-resume-20260928. Main worktree has unrelated uncommitted Hopper work and must not be altered. ParaCloud initially has no running/queued user jobs. Source was deployed but no canary/result existed.

## Measurements
Keep the existing 72 training, 72 test and 8 discovery shapes; two head shapes, six planner policies and three process replicates per GPU. Canary uses discovery/training geometries only. Do not fit on diagnostics. Time public plan+run+device-completion cycles separately from cached-plan CUDA run timing. Retain each paired block; inner calls are not independent replicates. Planner ABI must match the installed backend. Missing stock reference invalidates candidate comparisons.

## Fitting and selection
Reduce repeated observations to a median per geometry/head/policy/hardware before fitting. Fit each hardware separately, retaining both head shapes. Select regularization/model class with leave-one-training-family-out CV only. Candidate representations are marginal, joint, plan and causal. Every representation receives the same model-class search: standardized Ridge with alpha in {0.01,1,100}, and ExtraTrees with 128 trees, min_samples_leaf in {2,8}, seed 20260928. Predict log positive latency; choose the CV minimum mean absolute log error with deterministic ties. Refit on all training families. Test is opened only after model choices and selected fixed policies have been saved and hashed.

The strongest non-plan baseline is chosen between marginal and joint using training CV. Global fixed policy is selected by minimum mean log cost across training shapes. Report stock auto, all fixed modes, train-selected best fixed, all four learned representations and a labelled hindsight oracle. Selectors have no access to test timings. No learned-policy promotion based on beating stock alone.

## Endpoints and uncertainty
Primary predictive endpoint: test MAE with at least 20% improvement over the training-selected non-plan baseline. Primary dispatch endpoint: family-macro geometric speed ratio against train-selected fixed; paired bootstrap over whole test geometry families (20,000 draws, seed 20260928). Repeats and two heads stay within family clusters. Show both CUDA-run and measured plan+run-cycle outcomes, plus selector overhead and >5% regressions. No cross-hardware generalization claim without a no-refit transfer test.

Fit run-time and cycle-time models separately; compare on their declared endpoint. Measure Python selector overhead using unseen metadata only, and charge it once per new plan. Report a conservative additive cycle+selection estimate, not a measured integrated runtime speedup. A cached-plan reuse count of 32 may be reported only as an amortization sensitivity, not a model-serving measurement. Full vLLM goodput requires a separate registered live experiment after a positive operator gate.

## Causal and novelty scope
Within each shape keep Q/K/V tensors fixed while intervening on public split controls. Compare equal-work contrasts under auto versus common fixed modes, retaining failed/negative contrasts. Planner matching verifies metadata, not the entire physical mechanism. FA2 split-KV, plan/run separation, FLOPs-vs-latency, and LPT/load balancing are prior art. Candidate contribution is a validated, decision-useful causal execution-plan representation; reject it if strong-baseline/cost gates fail.

## Pre-full-run amendment after the canary
Two canaries (1632585/1632586) exposed undersized 128MiB scratch, not numerical failures. Allocate the checked backend-required maximum scratch once per geometry and share it serially across all policies; record policy-required and actual allocated bytes. This is equal reserved workspace, not a serving KV-capacity claim. Correct a missing canary fixture and add a large TRAIN fixture; no test timings selected these changes.

Add a stronger fixed baseline: pick the best fixed policy separately for each hardware/head configuration using TRAIN data only. Primary dispatch ratios and promotion gates must beat this head-conditioned fixed baseline, not only the global policy. Preserve the global selection as a secondary result.
