# Independent full-engine observation audit

Verified snapshot: **2026-09-28 07:01:03 UTC / 11:01:03 Dubai**.

This continuation adds and actually executes an independent evidence auditor. It does not report queued GPU work as measured inference. The frozen measurement source and the already submitted job/dependency graph are unchanged.

## Implemented, tested and deployed

`audit_observations.py` independently rebuilds token-observation timestamps from the per-engine-step event trace and checks them against every request record. It recomputes throughput, admitted TTFT, offered TTFT, per-request mean TPOT, request-latency p50/p95/p99 and both registered SLO counts/goodputs. It rejects missing or duplicate requests, missing final events, nonmonotonic timing, changed lengths, inconsistent summaries and workloads that never executed the target prefill adapter.

Its whole-job check reads every warmup and measured workload in all three arms, validates model/config/source identities, checks actual KV capacity separately from the requested value, and compares generated token IDs rather than trusting a success label. Output mismatches remain visible. Unknown actual capacity fails closed. A complete one-job screen is never a performance-superiority claim.

New tests: **26 observation-contract tests and 10 whole-job file/identity/parity tests**. Combined with the existing 18 measurement tests and eight statistical/completeness tests, **all 62 tests passed on ParaCloud's CPU runtime in 0.361 seconds**. The tests use explicitly synthetic temporary fixtures. They are NOT 62 inference experiments or GPU correctness results.

The new code is in commits:

- `19c4c254182a5f8a1ca6cf932fe0389d85b90896`: independent raw-observation audit.
- `90b95fe65d4aa2256874e6474e3121fed9189945`: observation-contract tests.
- `a3d94d417ca9034e7dbf2847b53c4c347edd19cd`: whole-job evidence tests.

The immutable GPU measurement remains at `7b480e2905456aafce3ef99cb5a6110162270ef5`. No queued source tree was overwritten.

## Existing jobs: checked, not duplicated

| Job | Purpose | State at the verified snapshot |
|---|---|---|
| `1634020` | Complete-model, three-arm qualification | PENDING — Priority |
| `1634042_[0-5%3]` | Six counterbalanced process-order triplets | PENDING — Dependency |

The dependent matrix had already been submitted by the preceding workflow. This continuation did not submit another copy, cancel a healthy job, change its priority, buy compute, or switch to a different GPU without qualification.

At the same snapshot, the normal ParaCloud availability view reported zero free GPUs in each of gpu_4090, gpu_5090, hp_4090 and hp_5090. This is a timestamped allocation bottleneck, not a newly inferred permission failure. The scheduler may change after the snapshot.

A bounded read-only collector inspects these existing jobs and saves `collection_state.json`. It changes no job. If a complete result appears, it applies the independent audit. No throughput, latency, SLO or full-model output-parity result existed at the snapshot above.

## Source and evidence hashes

Remote audit root:

`/ssd/scxi253/single-gpu-inference-plan-cost-20260928/artifacts/full-engine-observation-audit-90b95fe`

| Evidence | SHA256 |
|---|---|
| `all_62_tests.txt` | `ee637e4da557362119beb24ce2d0628815b31fe3670d23274179f24e1d073ab3` |
| `audit_observations.py` | `c379e14b2887d1552155b31bb29f74ff6f79cfe2d261cbe5495881733c0a1eea` |
| `test_audit_observations.py` | `88cab328f123dcece43c3733de52e3f4d6b4e7a739a2c7dfb11a0e83badb05d3` |
| `test_audit_job.py` | `aaef424a7eb8780261cb8bf2266ec1174179df8dbda8f0ea2e3f45230237a910` |

The same new source hashes were checked in the local delivery copy and on ParaCloud. Existing runtime, model and cuRAND prerequisite evidence remains in `campaigns/full-engine-v2/audit/`.

## Official implementation cross-check

The pinned vLLM v0.29.0 `vllm/v1/engine/llm_engine.py` assigns the engine's `vllm_config`, and `vllm/v1/engine/core.py` writes the scheduler-derived block count into `vllm_config.cache_config.num_gpu_blocks` during KV-cache initialization. The driver therefore reads an actual initialized field, but the audit still rejects missing/unequal values instead of assuming the requested capacity was honored.

Official benchmark documentation distinguishes request-mean TPOT from inter-output latency, particularly when one output event contains multiple tokens. This auditor does not fabricate extra zero-duration ITL observations. Our measurements remain in-process engine observations, not HTTP/network latency.

References:
- https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/engine/llm_engine.py
- https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/engine/core.py
- https://docs.vllm.ai/en/latest/benchmarking/cli/
- https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-environment-variables

## Resume

First inspect jobs 1634020 and 1634042 and their existing receipt/complete files. Do not repeat completed reuse experiments or submit duplicate full-model jobs. If the qualification fails, retain its error and dependent-job disposition. If it succeeds, keep the existing dependency-controlled formal matrix. Run both the original analysis and this independent audit in a fresh output directory after the applicable complete matrix exists.

Previous conclusions remain unchanged: the unamortized dispatcher is not promoted; the attention-stack positive scope is RTX4090/512MiB only; no complete-serving speedup, global novelty, or academic award is claimed here.
