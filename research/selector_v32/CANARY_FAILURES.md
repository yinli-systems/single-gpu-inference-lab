# Canary failures retained

## Commit `b09c1c7e5aea9271ef9f80d6d2cb930735c50541`

5090 job `1641737` completed the pristine arm (768 timing rows, 32 qualifications) and then failed before any paired timing because TVM-FFI plan metadata is an immutable `Array`; the harness attempted item assignment. Job `1641736` had not started and was cancelled rather than knowingly reproducing the same implementation failure. No release geometry was executed. The failed campaign remains under `/ssd/scxi253/single-gpu-inference-selector-v32-paired-20260930T090000Z`.

The fix reconstructs the exact original runtime container type from all 16 integer fields after changing only field 15. The selector rule, manifest, release gate and frozen release geometries are unchanged. A new commit and campaign are required; the failed campaign cannot be reused as passing evidence.
