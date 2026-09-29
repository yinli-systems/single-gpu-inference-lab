# Result interpretation: pairing matters, but work and unordered pairs are not the whole execution state

**Completed 29 September 2026.** This is a registered FlashInfer kernel experiment, not a full-model or production-serving optimization. Both canaries and all six formal jobs completed with exit `0:0`; no formal rows were removed. The previously rejected full-engine-v2 optimization remains NOT PROMOTED.

## Main result and limiting controls

All 16/16 registered auto/graph same-minus-opposite contrasts have positive conditional 95% intervals. The two states preserve chunk and cached-depth marginals while changing their pairing. However, **10/16 row-order controls fail the registered equivalence criterion, and 10/12 equal-work controls fail it**. These controls are part of the result, not exclusions or inconvenient rows.

Thus the evidence supports cost differences among states aliased by the specified q/cached-k marginal representation. It does NOT support attributing the entire measured difference to scalar attention work, proving that an unordered paired set is sufficient, or claiming an optimization speedup. Row permutation also changes physical concatenation/layout; its effect cannot yet be assigned solely to scheduler order, masking, cache locality, or wave quantization.

## Completed measurement and checking

|Item|Completed value|
|---|---:|
|Formal GPU jobs|6|
|Timing records|20,736|
|State/arm qualification records|864|
|Selected FP32-reference vectors|723,456|
|Exact graph/eager output checks|12,096|
|Maximum selected-vector absolute discrepancy, across FP16/BF16|0.0046712160|
|Allocated GPU time, including two canaries|1.248611 hours|
|CPU contract tests|31|
|Hash-verified upstream feature-map calls|108|

Timing rows are repeated measurements, not 20,736 independent workloads. Reference-vector counts are not full FP32-tensor coverage. FP16/BF16 tolerance checks are not bitwise cross-backend equivalence. Graph/eager exactness applies to the tested attention outputs only.

## Primary timing contrasts

The table shows paired means of same-minus-opposite kernel time, auto split, CUDA-graph replay. **These are milliseconds of shape-dependent cost, not saved application latency.** The complete FP16/BF16 table and all controls are in [RESULTS.md](RESULTS.md) and [summary.json](summary.json).

|Prefills|RTX 4090 FP16, ms [95% CI]|RTX 5090 FP16, ms [95% CI]|
|---:|---:|---:|
|2|0.0555 [0.0544, 0.0567]|0.8455 [0.8421, 0.8495]|
|4|1.8734 [1.8681, 1.8787]|1.3771 [1.2987, 1.4338]|
|8|1.7567 [1.6567, 1.8262]|1.2175 [1.2116, 1.2225]|
|16|4.0386 [4.0250, 4.0515]|2.9061 [2.8971, 2.9152]|

Intervals condition on three process repeats and twelve matched randomized blocks. They are not device-population intervals or multiplicity-adjusted significance claims. The n2 rank correlation is based on only two distinct pairings and is not strong independent evidence.

## Failed controls, with concrete examples

On RTX 4090/FP16/n4, reversing batch rows while preserving the same logical pairs changes measured time by **-551.819 microseconds**,90% CI[-554.604,-548.623], against a 90.950 microsecond equivalence tolerance. The same-order primary state median is 4547.475 microseconds. This is too large to describe that control as harmless measurement noise.

On RTX 5090/FP16/n16, the two distinct pairings with equal analytical W differ by **-306.343 microseconds**,90% CI[-329.495,-281.566], against a 98.747 microsecond tolerance. Thus this assay does not establish W as a sufficient runtime coordinate. The 2/12 equal-work controls that pass are the 4090 n16 FP16/BF16 cells; all other conditions remain visible.

Failing equivalence is not in general proof of a difference. Here the recorded control estimates and intervals must be read against the registered tolerance; the report does not reinterpret a failed equivalence test as an automatic universal theorem.

## A backend-choice diagnostic, not our speedup

For RTX 5090/FP16/n2, the auto/graph state medians are 1703.085 microseconds (same) and 854.461 microseconds (opposite). With split-KV disabled they are 1704.33 and 1638.39 microseconds, respectively. The much larger auto contrast is therefore sensitive to the existing backend option. The paired primary auto contrast is 845.535 microseconds; one cannot explain it merely by a hardware-independent q*k slope.

This comparison is descriptive evidence from the frozen explanatory arms. It is not a newly invented optimization, a deployment recommendation to force a mode, or a fully isolated causal explanation of which internal kernel/plan detail changed.

## Representation and prior-art corrections

Writing L_i=q_i+k_i gives the elementary identity `2*C=sum(L_i^2)-sum(q_i^2)-sum(k_i^2)`. Therefore augmented marginal second moments reconstruct analytical W without storing the entire paired set. We test that identity on every state and 2,000 random integer instances. **The full pair list is not uniquely necessary or proven minimal.** In this experiment, cached-depth marginals are preserved but the per-request total-length multiset q+k is not.

The hash-pinned Microsoft Vidur source first retains per-request parameters, then aggregates the prefill lookup to total rounded cached depth and a rounded chunk-square term. Executing its actual two reviewed feature-construction methods yields 108 matching key checks and aliasing within each configuration. This is conformance of a specific published implementation's key, NOT a retrained Vidur error benchmark or a claim about every simulator version. See [DESIGN_NOTES.md](DESIGN_NOTES.md) and the licensed reference source.

Plan/run separation, CUDA graphs, and kernel-aware/wave-aware cost modeling are existing techniques. No first-invention or state-of-the-art claim is made for them.

## Reproduction and environment boundaries

The raw archive is 2,124,778 bytes, with SHA256 `3e13e63f6bafb4243337e560b02d7b737afb47e341f5f06847f9461f50bde69a`. It contains canary/formal records, validation outputs, telemetry, source binding, launcher, and Slurm receipts. [data-manifest.json](data-manifest.json) records per-file hashes. Unscored warm-up iterations are not individual timing rows; the source and counters record their execution.

A fresh extraction and CPU-only analysis on the Mac regenerated both the canonical JSON and Markdown table **byte for byte** from the ParaCloud raw archive. This is a second-environment analysis reproduction by the same workflow, not an independent third-party GPU replication. No GPU is allocated by `reproduce.py`.

The observed allocations vary in host, driver and reported memory; some repetitions reuse a physical GPU. The paired within-process design and conditional intervals do not erase this heterogeneity or prove freedom from co-node contention. Only two GPU families, explicit FA2,32/8 heads, D128,and four specified geometry configurations are covered. No Llama/MoE/MLA model, H100 native FA3, paged cache, live request admission, TTFT/TPOT, or HTTP performance is tested here.

## Appropriate next scientific question

The next targeted question is why a semantics-preserving row permutation and the existing split option change time on these frozen counterexamples. A separate diagnostic capture can compare actual launch/plan metadata, physical layouts and selected kernels, without mixing profiler timings into these performance measurements. It should precede any new adaptive dispatcher or claim that a compact coordinate is sufficient. The original A100 full-engine permutation campaign remains a distinct unfinished experiment; this artifact does not substitute for it.
