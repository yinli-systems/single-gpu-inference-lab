# Selector v4.1.1 contract validation

This revision changes qualification/cache/authorization identity, not workload geometry or the 0.99 safety threshold.

- Qualification, measurement-policy and private binding revision: `4.1.1`.
- Case hash and all dev/canary/release/stress stage hashes remain unchanged.
- Candidate-native/off versus official-pristine is a blocking qualification: point worst >=0.99, simultaneous joint-min95% LCB >=0.99, duplicate-control90% intervals inside reciprocal +/-1%.
- Authorization binds the exact analyzer SHA, manifest SHA, source commit, source archive, overlay, stage, GPU and explicit native-overlay requirements.
- Old v4.1.0 summary/cache/binding records cannot authorize v4.1.1.
- Freshness ledger uses the tracked Linux-case-sensitive `MANIFEST.json` path.

ParaCloud validation environment: Python 3.12.13, NumPy 2.3.5. All 44 tests passed; compileall and both launcher shell syntax checks passed. The USTAR archive contained no PAX headers or AppleDouble entries.

Clean validation archive SHA256: `054401c074135ad4a65e98180db44be6481ff5d5698a4150d7b3e8dd8654ba04` (424 files).

This is CPU/source-contract evidence only. Exact-source dual-GPU smoke and dev qualification remain mandatory before any canary case is consumed. Default and serving promotion remain OFF.
