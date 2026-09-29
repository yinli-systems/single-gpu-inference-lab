# Physical-buffer-fixed FA2 descriptor ordering

**Status:** both real-GPU canaries passed. Formal arrays1638269(4090) and1638270(5090) are submitted, with three repetitions per family and at most two concurrent jobs per family. No formal performance conclusion is recorded at this checkpoint.

Unlike the previous physical row-repacking assay, this intervention leaves Q/K/V and output addresses, plan scalars, split configuration, output destinations and merge destinations unchanged. Only the three work-descriptor arrays are permuted as a complete bijection. Five policies include native identity, an identical A/A control, reversed request order, frozen primary heavy-first ordering and an interleaved explanatory order.

The measured scope is FlashInfer0.6.18 explicitFA2,32/8heads,D128,FP16/BF16,512MiB scratch,eager and fixed-shape CUDA-graph replay. Dynamic/padded graph plans are rejected. This is not a new mathematical attention algorithm or a serving optimization. The private-wrapper adapter is experimental and does not install into a running inference service.

## Evidence boundaries

- GPU measurement source commit: `ad26dac57553cfd7897e8370c23aa089ec06d33a`. `source.sha256` freezes the deployed measurement inputs; later `analysis/`, `native/`, `upstream-audit/` and receipts are explicitly separate, not retroactively preregistered measurement code.
- Canary jobs1638239 and1638241 completed0:0; each has240 timing records and60 state/policy qualifications with exact full output and LSE in eager/graph. Canary timings do not select the primary policy.
- Formal design:30 states(14 old discovery,16 from eight new held-out geometries),two dtypes,two split settings,five policies,two execution modes,six randomized blocks,three processes perGPU. Expected43,200 timing rows if every formal run completes. A partial matrix is not a result.
- Metadata validation/copying is outside the run timer and recorded as setup. Run-only ratios are not net planning/serving gains.
-18 pure contract tests include1,000 random geometries;14 analysis tests reject missing, duplicated, mislabeled or nonfinite data.
- Native C++ is a host-only prototype, not integrated into FlashInfer. Linux differential tests and the retained Mac SDK-linker failure are under `native/`.
- Upstream planner helper equality between0.6.18,0.7.0 and pinnedmain is source evidence only, not newer-runtime GPU qualification.

## Entry points

[Protocol](PROTOCOL.md) · [Real upstream PR triage](UPSTREAM_RESEARCH.md) · [GPU harness](measure_order.py) · [Bijection contract](plan_contract.py) · [Offline analyzer](analysis/analyze.py) · [Native prototype](native/README.md) · [Source audit](upstream-audit/source-audit.json)

CPU analysis takes an already collected campaign root:

```bash
python3 -m unittest discover -s benchmarks/results/plan-order-mechanism -p test_plan_contract.py
python3 -m unittest discover -s benchmarks/results/plan-order-mechanism/analysis -p test_*.py
python3 benchmarks/results/plan-order-mechanism/analysis/analyze.py --root /path/to/campaign --stage test --out /new/analysis/path
```

All six formal jobs, both GPU families and all planned cells must be present. Raw SHA256, source identity, descriptor maps, output/merge destinations, per-policy hashes and exact-output qualification are checked before statistics. Never re-submit completed jobs to improve a number.
