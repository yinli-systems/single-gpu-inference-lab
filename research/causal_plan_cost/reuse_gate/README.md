# Matched reuse milestone after 660d128

Completed 28 September 2026. This is AI-assisted research and implementation, not a submitted dissertation, a priority claim, or a prize guarantee.

## Result

The previous unamortized adaptive dispatcher remains **NOT PROMOTED**. This continuation recovered existing wave/native-policy experiments without duplicating them, derived a strict break-even test, implemented bounded cache and memory-validity contracts, and executed a new direct selector + plan + GPU attention-stack experiment. CPU overhead divided by a layer count is never reported as a measurement.

The registered primary native-cycle policy passes only on **RTX 4090 with 512 MiB scratch**, at 36 distinct layer calls per segment, under unchanged, alternating and LRU-eviction state sequences. Fixed/native elapsed-time ratios are **1.1669, 1.1416 and 1.1477**. Their nominal 95% family-confidence lower bounds are **1.0684, 1.0469 and 1.0512**. All three conditions also beat stock auto, with no greater-than-5% regression among the 18 base geometries in each condition.

**128 MiB and RTX 5090 do not pass the primary gate.** On 5090 with 512 MiB, the policy improves over stock but not reliably over the stronger training-selected fixed comparator. Beating stock alone is not sufficient.

The [machine-readable table](results/direct_reuse_summary.json) contains all primary conditions, reuse-count sensitivity, regressions and raw hashes. The [gate report](RESULT_GATE_20260928.md) includes the full primary table.

## Direct measurement and statistical boundaries

The new corpus contains 54 exact geometries from 18 bases and nine query/cache families. They are disjoint from prior training, testing and discovery geometries. Both GPU models have three process/seed replicates. No fitted weights or decision thresholds were retuned. The bases include two-request and eight-request batches. This is a synthetic held-out geometry experiment, not a production-traffic evaluation.

Each timed episode has four segments. At the primary setting, 36 layers per segment means **144 actual attention calls**. Unchanged geometry reuses one plan. Alternating A/B/A/B and eviction A/B/C/A rebuild one plan per 36-layer segment. Every arm receives the same two-entry decision LRU, active-plan reuse opportunity, data tensors, scratch ceiling and timing boundary. The 36 layers have distinct Q/K/V storage and values.

The experiment excludes model weights, MLPs, a serving scheduler and networking. It is not a complete Qwen model benchmark. The dimensions match the common 36-layer, 32-query-head, eight-KV-head, dimension-128 structure of the referenced Qwen3 configurations, but this assay uses FP16 rather than claiming validation of the original BF16 models.

The fixed comparator is selected per GPU, head configuration, memory budget and reuse count using the old training-only proxy `cycle + (r-1)*run`. It is not claimed to be the empirically optimal fixed choice under the new data-traffic regime. Every reported result is a newly measured direct wall time; stock auto is also evaluated.

On 4090 with 512 MiB, the eviction condition already produces a 1.0740x ratio at one layer per segment, increasing to 1.1251x at four and 1.1477x at 36. Therefore **amortization alone is not established as the cause of reversing the earlier result**. The cache implementation, head configuration, memory feasibility, data traffic and input geometry also differ from 660d128. That earlier negative conclusion is preserved.

Across all settings, 3,888 timing rows and 5,557,248 timed attention invocations completed. Repeated calls are not independent problems. Three process repeats are reduced by cell median; nine whole families supply 20,000 nominal bootstrap draws. No multiple-comparison adjustment or hardware-population guarantee is claimed.

## Correctness and memory

There are 32 passing CPU contract/statistical tests. Before fresh GPU timing, 216 fresh-geometry native choices matched the unchanged fitted models. The direct matrix passed 22,248 non-auto full-tensor comparisons and 25,920 selected FP32-vector checks. Maximum absolute differences were 0.00048828125 versus auto and approximately 0.0001294017 versus FP32. Tolerances were `atol=0.005, rtol=0.02`; this is not bitwise equivalence.

All states and all 36 layers were checked for selected modes before timing. Every timed output was not copied back and rescored. Separately, 15,552 arm/row checks validated actual plan, run, cache-hit, miss and eviction counts.

Scratch ceilings of 128 and 512 MiB are enforced with candidate-to-auto-to-unsplit fallback, not silent allocation growth. Equal serial scratch reservation controls this experiment; it does not establish unchanged production KV capacity. Maximum layer-tensor storage was 7,832,567,808 bytes. Peak PyTorch allocation was 9,326,039,552 bytes, including correctness-check temporaries. The full-serving capacity trade-off remains unverified.

## Recovered mechanism and break-even screening

The existing 864-record wave experiment completed before this continuation. Its four prespecified crossover cases at query totals 1344/2720 and 16/32 query heads all satisfy the directional prediction: the 5090 equal-work shape contrast exceeds 1.1 and exceeds the corresponding 4090 contrast. Equal total queries, cached depth, exact attention work and total-KV-length marginals were checked. Neighboring controls and all policy results remain in the evidence.

These A/B input-shape ratios are **not optimization speedups**. Wave quantization is prior art, and occupancy, locality and masking have not been fully isolated as separate causal factors.

The diagnostic linear condition for positive benefit is:

`r * (U_fixed - U_candidate) > C_selector + (P_candidate - P_fixed)`.

When per-call savings are positive, the first strictly profitable integer is `max(1, floor((C + deltaP) / deltaU) + 1)`. Zero or negative savings may yield no profitable reuse count, or only a finite profitable window. Exact arithmetic contract tests cover these cases.

Under the old measured-cost diagnostic, only 46/144 4090 cycle-target cells and 35/144 5090 cycle-target cells are profitable at 36. Those calculations use the earlier ctypes CPU overhead on ln01 and an approximate cycle-minus-CUDA setup difference. They do not establish the performance of today's direct GPU-worker experiment. See the [recovered mechanism and diagnostic summary](results/recovered_mechanism_and_break_even.json).

## Official implementation and novelty boundaries

The official 22 September Autotuner v2 article requires deployment-matched eager/CUDA Graph measurement and valid runtime/operation identity. FlashInfer v0.7 adds persistent winner reuse. POD and the exact pinned 0.6.18 ragged wrapper already document cross-layer auxiliary-structure reuse. Reuse, cache keys and plan/run separation are **not our inventions**.

The installed 0.6.18 `prefill.py` was hash checked. Its ragged path calls compiled plan/run functions directly and contains no AutoTuner, choose_one, autotune_v2 or MeasurementPolicy invocation. Runtime exports lack the v2 entry points. An official v2 control is therefore not directly applicable without an explicitly identified runner/version integration. Our fixed split is not labelled official autotuning. The [source-level applicability audit](OFFICIAL_PATH_APPLICABILITY.md) records the inspected implementation and references.

The supported increment is a reproducible, resource-bounded, backend-specific study identifying where unchanged learned plan choices have actual net value under realistic reuse, and where they do not. It does not establish first physical-plan selection, new attention mathematics, universal scheduling, or state-of-the-art serving.

## Actual vLLM follow-up and current blocker

Only after the 4090/512 MiB direct gate passed did this continuation implement a process-local paged adapter with page-unit conversion, a 36-layer NHD/HND transition checker, and an actual vLLMEngine driver. Job **1632883** was submitted normally with an isolated vLLM 0.29.0 / FlashInfer 0.6.18 / PyTorch 2.13 / Python 3.13 runtime and cached Qwen3-4B-Instruct-2507 weights.

The paged checker reached its final structured output and the launcher subsequently entered the auto engine arm. At the last successful inspection, no final three-arm completion marker was visible. The subsequent read-only request for `auto.log`, engine receipts, the paged summary and `sacct` was refused with:

> This tool call was blocked by OpenAI's safety checks. Please double check what you are sending.

No more specific reason was given. The same action was not rerouted through another tool, REPL, encoding or account. Full-model throughput, TTFT/TPOT, tail latency, SLO goodput and greedy-output parity remain **unverified**. A particular engine failure or missing GitHub permission is not inferred. The [checkpoint](ENGINE_CHECKPOINT_20260928.md) gives the exact state and minimum authorized recovery.

## Jobs and publication

Canaries 1632714/1632715 completed, using 380 allocated GPU seconds combined. Full arrays 1632759_0/_1/_2 and 1632760_0/_1/_2 completed, using 5,833 GPU seconds. The new reuse experiment totals **6,213 GPU seconds, or 1.7258 GPU hours**. Later engine-job accounting is not verified and is not included. Existing wave/native-policy jobs were not duplicated.

Key commits:

| Milestone | Commit |
|---|---|
| Prospective protocol | `93efc8714919a9a6f9ebbe9f498d73b9cea76bf9` |
| Frozen GPU measurement | `ddfd23a9e3aa5661457f2c4626b7d9e899042963` |
| Canary/parity gate before fresh timing | `6d1c3e732a43dea0034219076c5ff48c17f97915` |
| Primary results and raw hashes | `6da22af22877998cb794ca42b0f59b9bb15fda7b` |
| Paged/full-engine qualification source | `18c51f68396aca9b65bb8005e3f85517e1d32dd3` |
| Exact engine read-blocker checkpoint | `54d6a19ddfe0a5d9dac43bb6726b4696448dcf7c` |

Publication uses the permitted GitHub API on the existing `research/causal-plan-cost-20260928` branch. No successful terminal `git push` is claimed. Main and unrelated dirty worktrees were not overwritten.

## Evidence and reproduction

Remote root:
`/ssd/scxi253/single-gpu-inference-plan-cost-20260928`

Completed reuse raw data and manifests:
`campaigns/reuse-gate-v1/runs/{canary,test}-*/`

Audits:
`artifacts/reuse-gate-audit-v1/`

Archive containing completed reuse raw data, sources and recovered-wave audits, but not unverified engine logs:
`artifacts/reuse-gate-audit-v1/completed_reuse_evidence.tar.gz`

The archive is 872,135 bytes and has SHA256 `14d7ec604e4849677173bb2dc23aa9b8399f7e777b7d9beec8d384d6ba51413d`, covering 51 indexed files. Original wave raw measurements remain separately under `campaigns/wave-threshold-v1/runs/`.

Authoritative full metrics SHA256: `2cd464b70015eeef2089ed427876bc96a29c48079970b4fcdf08e9f9b9cd456a`.

Per-cell data SHA256: `a3e74ea71e69ff88d12657f27d6589c2dbc578fa7694cc7e4319f3dc1981cb07`.

Reproduce the analysis without new GPU allocation, using a fresh output directory:

```bash
P=/ssd/scxi253/single-gpu-inference-plan-cost-20260928
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312/bin/python
S=$P/artifacts/reuse-gate-audit-v1/research/causal_plan_cost/reuse_gate
export PYTHONPATH=$P/analysis-deps:$P/campaigns/native-policy-v1/research/causal_plan_cost:$S
cd "$S"
"$E" -m unittest -v test_contracts test_analyze_reuse
OUT="$P/artifacts/reuse-reproduction-$(date -u +%Y%m%dT%H%M%SZ)"
"$E" analyze_reuse.py --root "$P/campaigns/reuse-gate-v1" --out "$OUT"
sha256sum "$OUT/metrics.json" "$OUT/per_cell.json"
```

A GPU rerun needs a new campaign/output root and frozen source binding; do not resubmit the completed arrays. Engine continuation requires resolving the read-only blocker, not guessing from filenames or substituting operator results.
