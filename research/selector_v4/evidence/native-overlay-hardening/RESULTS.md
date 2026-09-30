# Native-overlay qualification hardening

This change is analysis/authorization hardening only. It does not change the kernel, measurement rows, shape families, tactic thresholds, or the 0.99 safety gate.

## Added blocking evidence

- Independent official-pristine versus candidate-native/off ratios are computed from the already frozen process-order protocol.
- Each cell reports point ratio, hierarchical process/block bootstrap CI, raw block worst, and pristine/off duplicate-control CIs.
- Promotion requires native-overlay point worst >= 0.99, simultaneous joint-min 95% LCB >= 0.99, and both control CIs within reciprocal +/-1%.
- Native-overlay evidence is never used to train or select cap; it is qualification-only.
- Cross-repeat identity, eligibility, and runtime cap-support receipts must remain identical.

## Validation attempts retained

1. First Linux validation correctly failed because the new helper divided row objects instead of their `wall_us` values. It also exposed a pre-existing case-sensitive freshness path (`manifest.json` versus tracked `MANIFEST.json`).
2. Both defects were fixed. The freshness-key correction changes no case, case hash, stage hash, selector rule, or threshold.
3. Final ParaCloud validation: Python 3.12.13, NumPy 2.3.5, 42/42 tests passed; compileall and both launchers' shell syntax passed.

Validation archive SHA256: `4e71cc8aa30fda1a5f74e0c2787f67b523c6437e9e6de9e52d3b413b209c3152`.

No GPU performance result is claimed from this CPU validation. The existing dev jobs use immutable source `8ea2169`; their raw evidence may be reanalyzed, but any canary requires a new exact-source smoke of the hardened commit.
