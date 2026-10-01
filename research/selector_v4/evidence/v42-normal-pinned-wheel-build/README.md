# Normal upstream wheel, exact source and vendor headers

The unmodified upstream PEP 517 build hook produced FlashInfer 0.7.1 with genuine Git commit eb4e3e60a6de71f00c0fb558da6f21e4b45d8b55. All 10,126 packaged files were checked byte-for-byte against the build source using the actual upstream setuptools package-dir mapping. Every non-generated package file also matches the SHA256 ledger captured before the build. The stable native Python prefix is unchanged.

CCCL, CUTLASS and spdlog are real Git sparse checkouts at the exact upstream pins. Every present tracked blob was Git-hash verified; all package/native source and runtime compiler-header paths are present. Sparse omissions cover non-build docs, examples and vendor tests. Git objects reused from cached source were admitted only if their blob IDs matched the target tree. The archive retains input ledgers, pins, metadata packs, source preparation logs, normal build log and the wheel/source mapping.

The first post-build checker failed because it assumed two build helper files came from flashinfer/data. Upstream maps flashinfer.data to the repository root. The original rejection is retained; the same unchanged wheel was reverified with that mapping. No wheel rebuild or package edit followed this checker error.

The 65,832,272-byte wheel has SHA256 0337548c78844c10bb5dad6276f76a5c50b45cb753636a6438e03621d26b260d. The full source input archive and wheel remain at receipt-bound remote paths. The compact archive and every member were independently verified after download. Shared Python environment unchanged; hook package installs and optional NVEp builds explicitly disabled through upstream options. No manually overridden version or Git metadata.

This is build/provenance evidence, not GPU, release or HTTP qualification. Dual GPU jobs 1645384/1645385 requalify this exact normal package. Previous d768 results used the recorded older CCCL payload and remain separate evidence.
