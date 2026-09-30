# Private-cache v4 smoke r2

Both GPUs completed pristine and v4 ragged/paged CUDA Graph executions with exact full output/LSE hashes. Forced native/cap plan bits are correct; candidate-pool policy selects cap on RTX 4090 and native on RTX 5090 for the frozen 48-descriptor smoke geometry. Campaign-private shared objects contain native and Resource symbols co-resident in the same module (176 instantiations of each family).

Slurm jobs ended nonzero only because the source commit contained an audit tool that looked for kernel definitions in generated JIT wrapper sources; FlashInfer includes the source-bound overlay header instead. The corrected audit was run against the same immutable private `.so` and overlay artifacts and passed. This is retained as `PASS kernel smoke / FAILED post-result tooling`, not rewritten as a clean Slurm success. No release case was consumed.

## Safe-autotune integration note

This evidence was produced by the sibling `kevin/selector-v4-autotune-20260930` source line before v4.1 integration. It proves that the separate native/resource symbols compiled and produced exact pristine-matched outputs in private JIT caches on RTX4090/5090. It is supporting development evidence only: v4.1 has a different qualification revision, stronger identity/cache contracts, native-after-cap sequence, longer windows, and new smoke jobs are required before canary.
