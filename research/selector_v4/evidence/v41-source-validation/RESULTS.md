# V4.1 source and CPU contract validation

- ParaCloud pinned environment: 38/38 selector-v4 tests passed; compileall and both Slurm launchers passed shell syntax.
- Official FlashInfer0.7 source: all nine expected source hashes matched before modification.
- Native `BatchPrefillWithRaggedKVCacheKernel` and `BatchPrefillWithPagedKVCacheKernel` source spans remained byte-identical.
- Candidate overlay contains distinct ragged/paged resource kernel clones; legacy plan defaults native; explicit policies are native0/resource-cap1.
- Generated `flashinfer/prefill.py` parses successfully.
- Binding SHA256: `7a8778e8dea2892cdd5d317c0d4a8c47365c36400177aaede46af189a8e68354`.
- Patch SHA256: `bb1d385104b5f0be3ee2ab7552b4e2d5e2c946775d6d7688f25fa5b5f19ecfea`.

This is source/CPU evidence, not compiled-GPU qualification. New dual-GPU v4.1 smoke remains required.
