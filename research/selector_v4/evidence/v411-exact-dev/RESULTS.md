# v4.1.1 exact-source dual-GPU development verdict

Both immutable jobs completed with exit 0 and no failure receipts:

- RTX 4090 job `1643854`, runtime 48:03.
- RTX 5090 job `1643855`, runtime 45:53.
- Campaign: `/ssd/scxi253/single-gpu-inference-selector-v411-dev-20260930T211230Z`.
- Source: `50f182dda4e611151a7b799dc2db73ae87b8d7b8`.
- Source archive: `9cbae6dee22dd4fccb7cbb870b83dde6727db5015341f0ad3913f7ca6a083725`.
- Resource binding: `5468256db82ad24d5ff20369576802cc504da5fa96f68d2fe24eec9935789263`.

## RTX 4090 — HOLD

- Selected coverage: 91.0% (131/144).
- Selected geomean: 1.387585x.
- Selected point worst: 1.320191x.
- Selected block worst: 1.286007x.
- Selected joint-min 95% LCB: 1.315340x.
- Candidate-native/pristine geomean: 0.998721x.
- Candidate-native/pristine point worst: 0.995879x.
- Candidate-native/pristine joint-min 95% LCB: 0.992703x.
- Sole failed requirement: `held_out_controls_resolve` (7 selected fold records).

## RTX 5090 — HOLD

- Selected coverage: 75.0% (108/144).
- Selected geomean: 1.400026x.
- Selected point worst: 1.307094x.
- Selected block worst: 1.295103x.
- Selected joint-min 95% LCB: 1.305203x.
- Candidate-native/pristine geomean: 0.999170x.
- Candidate-native/pristine point worst: 0.994911x.
- Candidate-native/pristine joint-min 95% LCB: 0.988980x.
- Sole failed requirement: `native_overlay_joint_min_lcb_at_least_0_99`.

## Interpretation and frozen next step

Kernel-symbol isolation is compiled and the native ragged/paged function spans are byte-identical to official 0.7; cap clones differ only by symbol name before the explicit launch-policy edits. This closes direct `cudaFuncSetAttribute` contamination of the native symbol as the primary explanation. However candidate-native still reaches a modified 16-field plan/resource dispatch, whereas official pristine uses the legacy 15-field plan/run entry. V4.2 must preserve the official native plan and run entries and add a separate resource-only run entry. Control measurement will be lengthened without relaxing the +/-1% or 0.99 gates.

Canary/release/stress remain untouched. No v4.1.1 canary is authorized. Default and serving promotion remain OFF. Historical 2/432 token divergence remains unresolved.
