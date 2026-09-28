# Independent causal-plan study and native CPU milestone

28 September 2026. Companion numerical extract: [summary.json](summary.json). Selection was publicly frozen first in [train-freeze.json](train-freeze.json).

## Result, not a promise

A frozen evaluation of six completed GPU runs supports a **backend-specific predictive improvement**, not a new production-winning scheduler. Causal execution-plan features reduce held-out CUDA-run prediction MAE by **50.70% on RTX 4090 and 44.33% on RTX 5090** against a strong joint-feature baseline chosen using training cross-validation.

We then implemented and actually compiled a native C++ version of the unchanged frozen policy. It preserves **2,432 checked choices** and reduces inclusive selection time from milliseconds to **approximately 11–47 microseconds**. Nevertheless, **none of the four unamortized additive net-performance gates passes** against the training-selected hardware/head-conditioned fixed baseline. The learned dispatcher is not promoted. No operator/selector number is reported as vLLM throughput or request latency.

## Restoration, concurrency and attribution

Original canaries 1632585/1632586 each retained 22 completed and 14 failed policy records due to undersized scratch. Their failures and the earlier normalized-marginal/M2n tie remain on record. This continuation recovered a reviewed repair into an isolated worktree, leaving dirty Hopper work and healthy jobs untouched.

This continuation's composite terminal GPU-preparation/submission attempt was blocked by the tool safety layer; it produced no job ID and was not rerouted. A **separately active execution session** then committed the repair and launched the successful campaign below. We discovered, preserved and waited for those existing jobs rather than submitting duplicates. We independently executed train-only fitting, frozen held-out evaluation, correctness/hash audits, supplemental prediction uncertainty, and native CPU implementation/validation/timing. This is not a claim that we launched the recovered GPU experiments.

Scoped SSH/CPU operations and explicit GitHub writes succeeded. It would be inaccurate to call the blocker missing GitHub write permission or unreachable ParaCloud. Publication here is through GitHub API commits, not a successful terminal git push. Later concurrent wave/native-policy campaign files were visible, but their detailed read was blocked; those results are not incorporated in this independent milestone.

## Frozen design and controls

The corpus has **72 training, 72 test, eight discovery geometries**, two head configurations (16 query / 4 KV and 32 / 8), six policies (auto, no-split, split512/1024/2048/4096), and three process/seed repetitions per GPU model. All **10,944 policy records** completed. This is 152 geometries, not 10,944 independent problems or complete LLM tasks. The head configurations are not separate full models.

Training and test are separated by whole query-budget/cache-scale combinations. Shape-generation patterns and scalar ranges may be shared; this is held-out-combination generalization, not arbitrary production traffic. Each GPU has 144 test geometry/head decision cells clustered in six families.

Every representation receives the same five-estimator search: Ridge alpha 0.01/1/100 or 128-tree ExtraTrees with leaf minimum 2/8. Leave-one-TRAIN-family-out absolute log-error CV selects the estimator. The strongest non-plan representation selected on training is joint for every endpoint. Training also selects fixed options separately by GPU/head: 4090 uses split512 for 16 query heads and no-split for 32; 5090 uses split512 for both.

Models, selected options and hashes were published before this auditor computed test metrics. Nominal confidence intervals resample six whole test families 20,000 times, seed 20260928. They are conditional on the frozen fit/corpus, not hardware-population guarantees; there is no multiple-comparison adjustment. Inner kernel calls are not independent samples. Prediction-error intervals are supplementary descriptive analysis, not another selection rule.

## Prediction: include the failed endpoint

MAE values are milliseconds. Cycle means actual public planning + attention run + device completion, not full-model serving.

| Target | Normalized marginal | Exact joint | Rectangular plan | Causal plan | Causal MAE reduction vs joint; family interval |
|---|---:|---:|---:|---:|---|
| 4090 CUDA run | 0.105603 | 0.086760 | 0.053680 | 0.042771 | 50.70%; [36.31%,57.38%] |
| 4090 cycle | 0.154683 | 0.144900 | 0.152359 | 0.134663 | 7.06%; [-1.78%,35.77%] |
| 5090 CUDA run | 0.083951 | 0.069837 | 0.045509 | 0.038879 | 44.33%; [32.45%,48.27%] |
| 5090 cycle | 0.112411 | 0.102748 | 0.068012 | 0.058495 | 43.07%; [30.94%,45.91%] |

4090 cycle fails the registered 20% improvement gate. Its p95 absolute error worsens from 0.876061 to 0.963921 ms. Do not call prediction universally improved. The name causal refers to the valid work under the causal attention mask; it does not establish a complete microarchitectural causal explanation.

## Decisions: lower MAE does not imply better selection

Ratios are training-selected fixed cost / selected cost, family geometric mean. Above one is favorable. These use measured policy costs **before charging selection itself**.

| Target | Stock auto | Joint | Rectangular plan | Causal; nominal family interval | Hindsight oracle among six options |
|---|---:|---:|---:|---|---:|
| 4090 CUDA | 0.9940 | 1.0003 | 1.0607 | 1.0482 [1.0114,1.0837] | 1.0990 |
| 4090 cycle | 0.9920 | 0.9978 | 1.0299 | 1.0361 [1.0043,1.0675] | 1.0717 |
| 5090 CUDA | 0.8999 | 1.0000 | 1.0156 | 1.0150 [0.9884,1.0439] | 1.0291 |
| 5090 cycle | 0.9271 | 1.0000 | 1.0133 | 1.0099 [0.9935,1.0318] | 1.0248 |

The rectangular-plan ablation has better 4090 CUDA choices despite worse prediction MAE. The causal policy has respectively **17,21,11,3** regressions over 5% among 144 cells. No test case was dropped for an unfavorable outcome.

The 5090 fixed split512 baseline itself has a 1.0786x cycle ratio versus stock [1.0281,1.1338]. This is existing parameter tuning, not a new algorithm or serving speedup. Against this strong baseline, the observed oracle has only about 2.48% cycle-ratio headroom among the six options. It is an optimistic hindsight diagnostic, not a guarantee or a bound over all possible kernels.

The predictor search selected by prediction loss, not policy regret; this is a matched prediction study, not an exhaustive best-adaptive-policy contest.

## Implemented native optimization: decisions unchanged, overhead charged

Native sources are under `research/causal_plan_cost/native_cpu`, fixed at **1e43161a464ebea5f45ca8fe5205e3e84546cf78**. The exporter loads only local hash-checked models produced by this experiment. It preserves Ridge scaling, tree float32 input conversion and model parameters; no refitting or threshold change occurs. Compilation disables fast-math and floating-point contraction.

Four models x 152 geometries x two head shapes x two SM counts = **2,432 exact choice matches**. All **510,720 scalar feature comparisons** match exactly on registered inputs; largest absolute log-cost prediction difference is 3.11e-15. This is not a proof for arbitrary inputs or full-model outputs.

The measured inclusive CPU path includes Python argument construction, FFI, all six feature/model evaluations and result conversion, 21 blocks of five calls per case. Model compilation/loading is offline. CPU measurements are on `ln01`, not an integrated GPU worker; before/after medians are not a paired serving benchmark.

| Frozen predictor | Python median ms | Native inclusive median microseconds | Native selection added to measured GPU cost; family interval |
|---|---:|---:|---|
| 4090 CUDA | 1.239672 | 10.694 | 1.0147 [0.9766,1.0554] |
| 4090 cycle | 8.044053 | 46.692 | 0.9534 [0.9146,0.9962] |
| 5090 CUDA | 1.273576 | 10.752 | 0.9715 [0.9486,1.0011] |
| 5090 cycle | 1.264656 | 10.669 | 0.9817 [0.9644,1.0075] |

These are **additive estimates**, not actual combined selector/GPU timing or LLM serving. All four positive-lower-bound net gates fail. Per-case p95 CPU-cost sensitivity does not reverse the decision. Across-layer amortization or caching could change this trade-off, but neither is established here. Do not claim a large end-to-end speedup from a much faster selector.

## Correctness, causal intervention and transfer

All 10,944 planner metadata checks, 9,120 non-auto full-tensor comparisons and 23,808 selected FP32 reference vectors pass. Auto-self checks are excluded from the non-auto count. Maximum full-tensor difference from auto is 0.0009765625; selected FP32 maximum difference is 0.000200510. Tolerances are atol0.005/rtol0.02, not bitwise equivalence.

The restored discovery intervention rejects a split-only explanation: at 16 query heads on4090, the equal-work shape contrast remains about 1.81x with splitting disabled, versus1.84x under auto. The5090 no-split contrast is near one. Changing split size changes several execution properties; occupancy, wave tails, masking and reuse remain incompletely isolated. These A/B shape ratios are not implemented speedups.

No-refit causal-policy transfer is asymmetric: trained5090 to target4090 raw cycle ratio1.0427 [1.0145,1.0712]; reverse1.0112 [0.9987,1.0322]. Selector cost is excluded. This is not stable bidirectional deployment generalization.

Maximum shared serial workspace is **3,571,875,872 bytes (~3.33GiB)**. Equal reservation isolates this experiment but does not establish production memory efficiency; lost KV capacity is unmeasured. FA2/FP16/D128/no-graph scope must not be generalized to all backends.

## Prior art and actual increment

- FlashInfer already provides query/KV-aware tiled work, split reductions, cost-guided load balancing and plan/run separation: https://arxiv.org/html/2501.01005v2
- FlashAttention-4 already discusses causal/variable-length scheduling and locality trade-offs: https://arxiv.org/html/2603.05451v1
- Tessera already separates logical structure from physical GPU plans in dynamic block-sparse attention: https://arxiv.org/html/2609.25869
- Official controls/workspace/graph qualifications: https://docs.flashinfer.ai/api/attention.html (current docs0.7.0, experiment installed0.6.18).

The supported increment is a **controlled backend-specific representation/decision study**, with stronger joint/fixed baselines, held-out combinations, actual native implementation and explicit boundaries. It is not invention of split-KV, plans, geometry awareness or FLOPs-versus-latency. Priority over all prior systems and a production-winning method have not been established. Earlier normalized-marginal parity and negative real-trace goodput findings remain in force.

## Jobs and exact recovery point

Measurement commit **32120f9b02f842f83dc9ecb0841e80b547524f86**. Train-freeze commit **d8618fa487f3336dedcc17372afc61d05a0db53e**. Native-source commit **1e43161a464ebea5f45ca8fe5205e3e84546cf78**.

Repaired canaries1632609/1632610:60/60 each, COMPLETED0:0. Full4090 array1632615_0/_1/_2 and5090 array1632616_0/_1/_2:1,824/1,824 each, all COMPLETED0:0. Full runs932allocatedGPUseconds; repaired canaries plus full runs1,138seconds/0.3161GPUhours. Earlier failed canaries add210seconds. This is not total project cost. This continuation submitted zero new GPU jobs.

Environment: PyTorch2.13.0+cu130,CUDA13.0,FlashInfer0.6.18,sklearn1.8.0,NumPy2.3.5,GCC11.4.0. A separate backend source-build GitSHA is not asserted.

Campaign:
`/ssd/scxi253/single-gpu-inference-plan-cost-20260928/campaigns/resume-20260928T2348Z`

Independent audit:
`independent-audit-20260928T0000Z/`

All raw paths/hashes and principal audit hashes are in [summary.json](summary.json). Raw measurements are `runs/full-*/measurements.jsonl`, with corresponding summary/manifest files. Native outputs include `native_compiled/export_manifest.json` and `native_validation.json`.

### Reproduction without new GPU allocation

Use a NEW output directory; do not overwrite the frozen audit.

```bash
C=/ssd/scxi253/single-gpu-inference-plan-cost-20260928/campaigns/resume-20260928T2348Z
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312/bin/python
export PYTHONPATH=/ssd/scxi253/single-gpu-inference-plan-cost-20260928/analysis-deps
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "$C/research/causal_plan_cost"
sha256sum -c "$C/source_manifest.sha256"
sha256sum -c "$C/analysis_source_manifest.sha256"
"$E" -m unittest -v test_geometry test_analysis
NEW="$C/audit-reproduction-$(date -u +%Y%m%dT%H%M%SZ)"
"$E" analyze.py fit --root "$C" --out "$NEW"
cat "$NEW/freeze.json"
"$E" analyze.py evaluate --root "$C" --out "$NEW"
# NATIVE is native_cpu/ from commit1e43161, not an unreviewed later revision.
"$E" "$NATIVE/export_native.py" --selection-dir "$NEW" --out "$NEW/native_compiled"
"$E" "$NATIVE/validate_native.py" --selection-dir "$NEW" \
  --native-dir "$NEW/native_compiled" --geometry-root "$C/research/causal_plan_cost" \
  --out "$NEW/native_validation.json"
```

The loaders validate raw hashes, complete coverage, source versions, split membership and numerical checks. Native validation refuses changed choices. CPU timings vary with host load; timestamp-containing selection JSON hashes may differ in reproduction even when fitted decisions match.

**Not completed here:** integrated native/GPU cycles, full vLLM model validation, throughput/TTFT/TPOT/tails/SLOgoodput, graph execution, live KV-capacity effects, complete physical mechanism, and comparison against all prior systems. The candidate is not promoted by this audit. This AI-assisted artifact is not a submitted dissertation or an award guarantee.
