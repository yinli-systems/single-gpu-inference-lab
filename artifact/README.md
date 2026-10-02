# Reproducible, scope-bound resource-sensitive attention research artifact

Run from the repository root:

```sh
make reproduce-v42-paper ARTIFACT_PYTHON=python3
```

Use Python 3.9–3.12. The command creates `artifact/.venv`, installs CPU-only dependencies from an exact-version, distribution-SHA256 lock, then checks every frozen input, independently checks all 1,994 members of the complete private Graph archive and its traced launches, and rebuilds six SVG/PNG figures, three CSV/HTML tables, and full JSON analysis. The name refers to the paper's figure/table evidence; it does not compile the dissertation PDF or certify serving. The first install requires PyPI access; subsequent builds reuse the isolated environment. No CUDA or serving environment is modified. Input archives occupy about 1.4 GB; allow approximately 1 GB additionally for the environment/cache/output. Large archives are already retained in Git evidence.

Existing output is never overwritten. To repeat:

```sh
make reproduce-v42-paper ARTIFACT_PYTHON=python3 ARTIFACT_OUT=artifact/generated-repeat
```

Open the generated `index.html`. `manifest.json` records all input, source and output SHA256 values. Dependency artifacts are accepted only with hashes in `requirements.lock`. Frozen source mismatches or cached raw-member mismatches stop the build. These hashes establish the bytes used, not the scientific validity of an experiment.

## Actual evidence and authority

The current original v4.2 dual canary is **HOLD**. `default_promotion=false`, `serving_promotion=false`, and the original **2/432** token divergence remains unresolved. Original 10 canary cases have been consumed; original 48+12 are not claimed as completed. This artifact consumes no fresh cases and changes no thresholds, frozen decisions or production implementation.

* The retrospective timing population is the original **complete RTX5090** column/arithmetic supplement: two exposed geometries, 48 cells, three actual processes per role, 24 balanced blocks per process and 144 fixed scoring folds. It does not repair the incomplete original RTX4090 campaign or promote dual development.
* The separate private Graph driver has actual dual-GPU functional evidence: 32 cells, 96 metadata epochs, 3,264 traced Resource attention launches at 65,536 bytes of shared memory, and 176 identical SASS instruction pairs per card. These tests establish the specified fixed-capture driver only. They include large workspace copying and Native references, and establish no serving performance guarantee.
* Figure 6 and table 3 explicitly record unavailable HTTP/SLO evidence. They contain no interpolated throughput, latency or SLO values. H100/B200 are also unmeasured. Current public Slurm partitions observed on Paracloud expose RTX4090/RTX5090; this does not establish universal datacenter GPU unavailability.

## Metrology and regret

The hierarchy resamples **processes, then whole paired balanced blocks** in log space. Mirrored within-block windows are preserved together. The 20,000 bootstrap draws never count as new processes. Reported 95% intervals are retrospective marginal intervals, **not simultaneous qualification lower bounds**. Three processes give a limited view of between-process uncertainty. Descriptive process/block/window variances are not unbiased random-effect components. All retained windows are used; none are trimmed.

Four definitions are reported separately: the pre-existing nonnegative available-arm excess; signed independent policy-versus-Native/oracle timing differences; frozen-label lookup against an independent oracle table; and the same lookup restricted to actually measured certified Native+Cap pools. The existing managed oracle falls back to Native for unavailable Resource tactics. Consequently **36/144 folds lack a measured Native+Cap counterfactual**. The available-arm P99 of approximately **0.042468%** is not a global best-forced-cap regret guarantee. Regret summaries include median, mean, P90/P95/P99, worst, negative observations and counts exceeding 0.5%, 1% and 5%. Fractions and percentages are explicitly distinguished.

## Risk and structural prior

The offline risk prototype requires safety checks plus an exact one-sided binomial upper confidence bound. Its failure event is at least one paired block more than 1% slower than Native in one complete process. Bounds assume independent exchangeable processes for a fixed source/environment/geometry/protocol; they are not request-level production guarantees. With zero failures, three processes yield a 95% upper bound of approximately **63.16%**, and five approximately **45.07%**. Supporting a bound strictly below 0.1% requires **2,995** such independent zero-failure trials. Repeated windows or bootstrap draws cannot supply that evidence. The prototype does not issue executable Resource certificates.

The structural-feature logistic prior is a training-only advisory prototype, bound to an environment key with leave-one-geometry-out analysis. It rejects scoring data and other environments. At least five distinct geometries are required; this exposed dataset has only two and produces **INSUFFICIENT_GEOMETRY_SUPPORT**, with no fitted probabilities or validated pruning. It does not yet implement an analytically derived occupancy/cache model. Synthetic tests verify API constraints; they establish no hardware benefit. Actual Resource choice continues to require the existing empirical eligibility/certificate path.

## Remaining experiments

The already dispatched immutable complete qualification keeps its original three-process protocol. A new five-process, 250–500 ms, 24–32-block protocol would be a separately declared experiment, and cannot be substituted into that run. GraphStep/GraphServing, Blackwell 32/128 connections, datacenter GPUs, real serving trace coverage, constrained-policy baselines and SLO Pareto comparisons require new measured evidence. The original token divergence requires the original failing histories/states or a faithful reproduction; new passing histories cannot close it.

Method references: [FlashInfer Autotuner v2](https://flashinfer.ai/2026/09/22/autotuner-v2.html) describes fresh timing against frozen candidate choices; [NVIDIA Graph performance guidance](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/performance-issues.html) describes device-connection tuning. These motivate future comparisons and are not substituted for this project's measurements.

## Retained independent audit

The local persisted reproduction and complete Paracloud CPU archive are included in `persisted-reproduction/`. Verify all remote archive members, extracted bytes, source hashes, output hashes, strict discrete results, float comparisons and decoded figure pixels:

```sh
artifact/.venv/bin/python artifact/verify_reproduction.py
```

Two local builds are byte-identical. Across macOS Python 3.9.6 and Linux Python 3.12.13, all six SVGs are byte-identical and all six PNGs have identical decoded pixels. Of 1,604 floating values, 40 differ by at most 2.220446049250313e-16; discrete values match exactly. PNG compressed bytes may differ across platform encoders. The remote receipt counts 18 hashed outputs; the manifest itself is the nineteenth generated file.

## Separate real Native Graph HTTP artifact, 2026-10-02

The earlier six-figure reproduction remains bound to its historical inputs. A
new separately measured Native HTTP/Graph dataset is now available:
[`native-graph-http-20261002-r2/index.html`](native-graph-http-20261002-r2/index.html).
Run `make reproduce-graph-http GRAPH_HTTP_OUT=artifact/graph-http-repeat` to verify
all499 raw archive members and reproduce its CSV, full JSON, 30-point SLO grids
and HTML page. It uses the existing isolated artifact environment; bootstrap that
environment with `make reproduce-v42-paper` if needed. Outputs are never overwritten.

These are complete Qwen2.5-Coder-1.5B BF16 fixed-workload Native observations from
one allocation per RTX4090/RTX5090, with a separate instrumented Graph observer.
They supply no Resource gain or independent-process confidence interval. Actual
Graph launches and changed GPU payloads were independently verified on both cards;
RTX5090's first Native/observer natural token comparison remains HOLD (9/128
requests,900/7,936 token positions differed). Instrumented timings are invalid.
Ordinary Native restart controls subsequently passed both cards. The full
four-model Resource HTTP/SLO qualification is still pending its formal prerequisites.

Detailed current scope and source-bound advancement rules:
[`GRAPH_SERVING_PROTOCOL.md`](../research/selector_v4/serving/GRAPH_SERVING_PROTOCOL.md).
Historical2/432 divergence and both disabled promotion switches remain unchanged.
