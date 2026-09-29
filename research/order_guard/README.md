# Locality-constrained order proposals and charged admission

**Status (29 September 2026): implemented and CPU-tested; no new GPU run, no production promotion.**
The authorized remote Mac became unreachable and then disconnected during this work. No replacement GPU job was submitted. PR #22 and its failed default-promotion decision remain unchanged.

## Implemented here

`order_guard.py` provides an exact integer **logical causal-pair** counter for a GQA-packed query tile, two order proposals, and a conservative empirical admission screen. It does not predict tensor-core padded work, physical CTA issue order, cache misses or elapsed time.

- `causal_heavy` ranks descriptors by exact logical causal pairs, rather than the previous last-row rectangular approximation. Longest-work-first is prior art, not a new scheduling theory.
- `locality_packet8` keeps the first eight descriptors fixed, preserves order inside eight-descriptor packets, moves later packets only inside 32-descriptor windows, and rejects proposals that increase request/KV-split boundary transitions. This is a new experimental implementation in this repository, **not a demonstrated new-to-the-literature algorithm or performance guarantee**.
- `select` and `admit` separate selection from six confirmation pairs and six A/A pairs, reject reused trial IDs or changed declared execution context, require every confirmation gain to exceed 1%, and charge probing/setup/dispatch against expected reuse. Unknown reuse falls back to identity. Already spent probing remains a cost even after fallback. The gate is an empirical screen; it does not implement C-SquareCB/C-FastCB or prove a worst-case bound.

The context key records declared GPU model, driver, software/source, geometry/plan, dtype, execution mode, cache label and timing boundary. It is not a complete representation of hardware state or proof that two physical GPUs are interchangeable. Clocks, affinity, co-tenancy and device identity must also be audited in future measurements.

## Executed CPU validation

The accompanying receipt records **19 unit tests**, **10,000 random geometries**, **563,326 descriptor checks**, **47,299 independently enumerated tile checks**, and **30,000 proposal-bijection checks**. These are CPU mathematical and software-contract checks, not GPU numerical qualification or performance measurements.

`collisions.py` constructs states preserving three full marginal distributions (query, cached prefix, total attention length) and scalar logical work. Of 1,080 tested fixed-tile/split constructions, 256 have different abstract descriptor counts. One example is:

|Coordinate|A|B|
|---|---|---|
|Query lengths|64, 128, 192|64, 128, 192|
|Cached prefixes|64, 128, 0|128, 0, 64|
|Total KV lengths|128, 256, 192|192, 128, 256|
|Logical work W|49,344|49,344|
|Descriptors, group=4/tile=128/chunk=128|22|20|

Both states have identical sorted values in all three length columns. This is an exact counterexample for the abstract feature map and descriptor contract. **It is not evidence that FlashInfer's auto-planner chooses that split or that measured GPU time differs.** Source-exact claims against named simulators need additional audits.

## Reanalysis of existing GPU measurements

`analyze_archive.py` verifies the original 140-file archive SHA256 and run receipts, and reanalyzes all 43,200 formal rows. It never assigns a measured time to either new proposal.

The known adverse geometry is query `[672,176,96]`, cached prefix `[8192,16896,24064]`. In the old FP16/graph/unsplit data, heavy-first is approximately **1.233x on 4090 but 0.706x on 5090** for the same shape. This is not proof of an SM-wave or cache mechanism.

An offline three-stage selector chooses on process A, confirms on B and scores on C, rotating through the three existing processes. These states were all exposed; the folds are dependent, and the result is development evidence only. On 5090, an **uncharged run-only** diagnostic gives 1.05009x geometric mean with a worst ratio of 0.97998x: avoiding the known 42% failure still leaves a roughly 2% regression. It is not a new GPU speedup or a no-regression result.

The three 4090 processes have differing driver versions; the strict declared-context variant therefore falls back in all three-way folds. The separate driver-ignored result is explicitly an invalid-for-deployment diagnostic.

A cost model using the historical probe recipe's actual invocation counts, wall times and diagnostic setup costs rejects all proposed 5090 selections at assumed reuse budgets of 36 and 1,000. Even at 10,000 future calls, the modeled total-cost ratio is below one when sunk probes on fallback states remain charged. This model excludes JIT, reference checks and warmup and assumes zero dispatcher overhead. It is a **counterfactual for this expensive historical probe recipe**, not a measurement of a new dispatcher, not an unavoidable tuning lower bound, and not a claim that FlashInfer Autotuner v2 cannot amortize startup tuning. A real system should preflight incompatible contexts before probing.

## Reproduce without a GPU

From the repository root, using a new output directory:

```bash
python -m unittest discover -s research/order_guard -p 'test_*.py' -v
mkdir -p /tmp/order-guard-review
python research/order_guard/stress.py --out /tmp/order-guard-review/stress.json
python research/order_guard/collisions.py --out /tmp/order-guard-review/collisions.json
python research/order_guard/analyze_archive.py --out /tmp/order-guard-review/development.json
python research/order_guard/manifest.py --out /tmp/order-guard-review/manifest.json
```

The historical archive is `benchmarks/results/plan-order-mechanism/raw-evidence.tar.gz`, SHA256 `70ba29c0e70e1178cc296996bfbc98cefbc52953744d727aabb4545f0c96410b`.

## Unmeasured GPU qualification entry point

`gpu_probe.py` is real-CUDA-only and requires the original pinned FlashInfer 0.6.18 adapter. It has been syntax-checked and its CLI inspected, but **has not executed on a GPU**. It does not allocate a job or modify a shared installation.

```bash
python research/order_guard/gpu_probe.py --stage canary --out /new/unique/canary-output
```

The manifest generator freezes 57 states: 48 fresh-shape entries, eight constructive-collision entries, and one explicitly exposed regression sentinel. The case SHA256 is `fb6c14b78cc03f8d02b7b16ff8efa9f382ce313b13e46781f5506a78be7cfa37`. Some entries share geometry; do not count entries as independent workloads. See `PROTOCOL.md` for completion and promotion gates.

## Prior art and upstream scope

FlashInfer [Autotuner v2](https://flashinfer.ai/2026/09/22/autotuner-v2.html), released with 0.7, already handles deployment-matched timing, default-path comparison and context-sensitive persistence. Those are not claimed as our inventions. Its [PR #3861](https://github.com/flashinfer-ai/flashinfer/pull/3861) is a design pointer from the official article, not a PR fixed by this work.

The reviewed open [PR #5687](https://github.com/flashinfer-ai/flashinfer/pull/5687), head `ecbee2f0871cb282c96885bdffa7a64dc3ce7491`, implements persistent tail-queue scheduling for SM90 VSA/cake paths. It is a relevant comparator, not the dense FA2 path qualified in PR #22. This work is not its author and has not solved that PR.

[Conservative Contextual Bandits: Beyond Linear Representations](https://arxiv.org/abs/2412.06165) develops C-SquareCB and C-FastCB under statistical assumptions. Its cumulative safety guarantees must not be relabeled as per-kernel worst-case guarantees. No such algorithm or guarantee is claimed for the empirical gate here.
