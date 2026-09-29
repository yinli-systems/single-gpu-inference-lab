# Execution-contract integration (experimental, default off)

An additive read-only validation layer for the existing `resource_generalization`
HTTP collector and result summaries. The frozen collector, manifests, candidate
headers, running jobs and old statistical analysis are not edited.

## What is implemented

- Exact integer representations: aggregate, marginal, joint-work, unordered
  request pairs, ordered request pairs. No rounding-based fake collisions.
- A two-action decision-reversal bound conditional on supplied intervals and a
  common execution context. The elementary bound is not claimed as a new theorem.
- Explicit probing/setup/dispatch/sunk-cost accounting; unknown reuse and
  mismatched contexts fall back. This utility does not dispatch GPU work.
- A reader for actual HTTP artifact schemas that recomputes TTFT, TPOT,
  throughput, all reported quantiles and joint-SLO goodput from request/SSE data.
- Mandatory hash coverage, exact requested IDs and token counts, lifecycle
  record coverage, model-shard manifest consistency and cross-arm source checks.
- A pipeline that runs the original statistical analyzer only after strict
  group validation and request-by-request token parity. Failure is quarantined.
- An exact-feature collision scanner for derived cost summaries. It returns
  candidate witnesses, never a certified GPU discovery. No collision is not a
  proof of sufficiency; unresolved controls and contexts are not pooled.

## Run from the repository root

```bash
python -m unittest discover -s research/execution_contracts -p 'test_*.py' -v
python -m research.execution_contracts.audit --out /tmp/new-http-audit.json \
  http --job /path/to/campaign/serving-runs/JOB_ID --stage smoke
python -m research.execution_contracts.pipeline --root /path/to/campaign \
  --jobs JOB_0 JOB_1 JOB_2 --stage formal --repo . --out /tmp/new-gated-analysis
python -m research.execution_contracts.audit --out /tmp/new-collisions.json \
  scan --manifest /path/to/campaign/source/manifest.json \
  --summary /path/to/complete/summary.json --representation joint_work
```

All output paths must be new. The pipeline additionally requires NumPy for the
existing bootstrap analyzer. Re-review is mandatory if its pinned Git blob
`369441f58807d279aac54b967474fdd097f70c46` changes.

## Boundaries

The source schema is pinned to resource-generalization commit `42a000d`.
No throughput improvement, live dispatch integration, GPU requalification,
novelty guarantee, statistical coverage guarantee or production readiness is
implied. A 64-KiB launch count is not enough to attribute a candidate path.
Model manifests are checked for internal consistency, not independently
re-hashed model bytes. Nominal per-cell intervals are not simultaneous bounds.
Failed integrity or parity prevents the gated speedup table; it does not prove
the resource-only candidate is the cause of nondeterministic model outputs.
SSE coalescing is retained: client TPOT is not a per-token kernel measurement.

The fault-injection tests use synthetic protocol fixtures. Acceptance of a
corrupted fixture by the old validator is not evidence that an existing GPU
artifact was actually corrupted. Validation reports preserve that distinction.

## Prior-art boundary

FlashInfer Autotuner v2 already covers measurement-mode-aware tuning, contextual
identity and invalidation. NVIDIA documents shared-memory occupancy sensitivity
experiments. These primitives are not claimed as new. The next research gate is
a genuine, qualified same-feature/opposite-decision witness and its cost-effective
resolution, rather than relabeling an existing tuning technique.

- https://flashinfer.ai/2026/09/22/autotuner-v2.html
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

## Recorded validation

See [audited results](RESULTS.md). To reproduce the synthetic fault comparison:

```bash
python -m research.execution_contracts.fault_injection --repo . --out /tmp/new-fault-report.json
```
