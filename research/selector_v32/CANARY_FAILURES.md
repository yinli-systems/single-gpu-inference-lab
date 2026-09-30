# Canary failures retained

## Commit `b09c1c7e5aea9271ef9f80d6d2cb930735c50541`

5090 job `1641737` completed the pristine arm (768 timing rows, 32 qualifications) and then failed before any paired timing because TVM-FFI plan metadata is an immutable `Array`; the harness attempted item assignment. Job `1641736` had not started and was cancelled rather than knowingly reproducing the same implementation failure. No release geometry was executed. The failed campaign remains under `/ssd/scxi253/single-gpu-inference-selector-v32-paired-20260930T090000Z`.

The fix reconstructs the exact original runtime container type from all 16 integer fields after changing only field 15. The selector rule, manifest, release gate and frozen release geometries are unchanged. A new commit and campaign are required; the failed campaign cannot be reused as passing evidence.

## Completed/failed old native-metadata canaries (`672e156`)

5090 job1641803 completed and its frozen canary gate is HOLD: selected Graph16 point worst1.3432355x does not override eager policy worst0.9394821x and disabled-overlay worst0.9441731x. Its 2304 timing rows and complete saved output/LSE hashes are archived.

4090 job1641802 ran without duplication and failed while allocating a512MiB workspace for the final BF16/paged/unsplit near-opposite coordinate. Pristine completed32/32 qualifications and768 timings; paired completed31/32 and1488 timings before CUDA OOM. The log reports23.41GiB in this process and22.76GiB allocated by PyTorch. This is not a recorded tensor-mismatch failure. The partial rows, qualification records, failure JSON and log remain evidence; they cannot be counted as a complete4090 canary.

Revision3.2.1 scopes every case's runtimes/graphs and collects unreachable cycles outside timing; it records post-case CUDA allocation/reservation. This is a candidate repair for retained object state, not a claimed GPU-validated OOM fix until the new canary runs. The geometry and selector are unchanged.
