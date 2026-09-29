# Fixed-buffer FA2 plan-order intervention

**Completed:43,200 formal GPU timings plus3,456 native plan+run pilot timings. Default/serving promotion remains OFF.**

[Findings, regressions and limits](INTERPRETATION.md) · [Formal results](formal-results/RESULTS.md) · [Native cycle pilot](native-results/RESULTS.md) · [Protocol](PROTOCOL.md) · [Upstream triage](UPSTREAM_RESEARCH.md)

This work leaves physical Q/K/V/output buffers, plan scalars and output/merge destinations unchanged, while permuting a checked bijection of FA2 work descriptors. It separates metadata ordering from the physical repacking confound in the earlier permutation assay. A source-integrated, default-off native helper was actually compiled and exercised on RTX4090 andRTX5090.

The predeclared heavy-first policy averages about4.50%/2.36% run-only improvement on the frozen4090/5090 holdouts. **It also produces approximately42% slowdown on two5090 holdout cells.** A/A has1/11 failed equivalence controls, all in eager mode. No unfavorable state is omitted. Read the complete table rather than treating the mean as a universal optimization.

Native planning+run is measured separately: held-out gains are under0.5% on4090 and about2.7–2.8% on5090 in a ONE-process pilot. That pilot covers same-pairing states only, not the opposite-pairing regression. Neither experiment is a full-model, paged-cache or serving benchmark.

## Artifacts

`plan_contract.py` enforces source/ABI/ownership/bijection contracts. `measure_order.py` is the immutable real-GPU measurement harness. `native/flashinfer-experimental-order.patch` is the source prototype; `native/measure_native_cycle.py` measures actual plan+run cost. `analysis/` rejects incomplete matrices, altered raw hashes, wrong geometry/descriptors and missing exact-output qualifications.

The original measurement commit is`ad26dac57553cfd7897e8370c23aa089ec06d33a`; native integration is`e1281a9eb9d38166eea01ef9e3dd3c3421afd8568`. Later analysis, receipts and test-runner packaging are not retroactively labelled frozen GPU source. Source manifests refer to the deployed versions retained inside the raw archive.

## CPU-only reproduction

```bash
python3 benchmarks/results/plan-order-mechanism/reproduce.py   --extract-to /tmp/sgi-order-data-NEW   --out /tmp/sgi-order-analysis-NEW
```

Use fresh paths. NoGPU/model download is used. The140-file archive is SHA256 verified against[data-manifest.json](data-manifest.json). Fresh Mac analysis reproduced both tables byte for byte; JSON differs only in three last-bit floating-point values. See[offline-reproduction.json](offline-reproduction.json); do not claim JSON byte identity or independent third-party GPU replication.

All ten allocated GPU jobs completed0:0. Recorded GPU allocation is1.412778hours, including canaries, not a price estimate. See[resource accounting](resource-accounting.json). Local Mac SDK-linker and initial CI failures are retained; final CI applies to its exact pushed commit.
