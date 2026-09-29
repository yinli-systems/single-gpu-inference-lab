# Ada/Blackwell controlled permutation artifact

Status: registered measurement source and strict complete-matrix analyzer. GPU qualification is in progress; no formal performance result is claimed.
See [PROTOCOL.md](PROTOCOL.md) for hypotheses, exclusions and limits. This is a kernel-level companion to the existing prefill-cost research, not a new serving optimization or an A100 full-engine result.

CPU contracts: `python -m unittest -v test_campaign` from this directory.
GPU invocation in a pinned FlashInfer0.6.18/PyTorch2.13+cu130 environment:
`python campaign.py --stage canary --rep 0 --out NEW_OUTPUT_DIRECTORY`
Formal: `python campaign.py --stage test --rep REPEAT_0_TO_2 --out NEW_OUTPUT_DIRECTORY`.
Never reuse an existing output directory or include canary samples in formal estimates.

Analysis contracts: `python -m unittest -v test_campaign test_analysis`.
After all six formal runs complete: `python analyze.py --root RUNS_DIRECTORY --out NEW_ANALYSIS_DIRECTORY`.
The analyzer rejects missing cells/repeats, wrong trial IDs, changed geometries, altered hashes, missing graph checks and mixed runtime identities. Canary data are never part of the formal report.

## Completed qualification

Both registered FP16 n2/n4 canaries completed successfully before formal submission. Each performed9,600selected FP32-vector comparisons and112graph/eager exact-output checks; maximum FP32 absolute discrepancies were0.0002696514 (RTX4090) and0.0004703999 (RTX5090). See [qualification.json](qualification.json). These are numerical/infrastructure checks, not formal latency results.

## Representation audit

[Design notes](DESIGN_NOTES.md) clarify why full pair lists are not the only possible representation. Extra marginal second moments reconstruct analytical work exactly. Run `python representations.py --out NEW_AUDIT.json` and compare to [representation-audit.json](representation-audit.json). This is exact arithmetic, separate from measured runtime cost.

CPU contract suite: `python -m unittest -v test_campaign test_analysis test_representations` (26tests).

## Actual upstream key audit

[Vidur source audit](vidur-source-audit.json) records108calls to hash-verified upstream prefill-feature methods. It confirms invariance of the specific aggregate lookup key, not whole-simulator prediction error. The [licensed reference source](reference_sources/vidur/README.md) is included for offline reproduction.

Run `python reproduce.py` for CPU-only source,contract,representation and upstream-key checks. This command never launches GPU work.
