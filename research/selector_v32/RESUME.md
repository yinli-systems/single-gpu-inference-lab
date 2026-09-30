# Resume selector-v3.2.1 without duplicating experiments

## Frozen execution

- Research branch: `kevin/selector-v32-paired-20260930`.
- GPU source commit: `a849cf4c9894d2d123d66dc27e417e162526e1c6`.
- Campaign: `/ssd/scxi253/single-gpu-inference-selector-v321-native-20260930T100521Z`.
- Source archive SHA256: `4547b66c6d5ed84eb78a1d95acb6d5ae14273ba6b81e9d6bec8ece183787eaa4`.
- Manifest case SHA256: `e6fb15cb7f9d240e1dca6943d3a9c4802af7c239dad2cbc57336aa218ae0df13`.
- RTX4090 canary: **1641933**. RTX5090 canary: **1641934**.
- At2026-09-30 10:10:35UTC both were PENDING(Priority), with no new completion records and no release runs. Read the current queue rather than treating this snapshot as live state.

Actual submission is proven by `evidence/recovery-20260930T0934Z/v321-submission/canary-submit.json`. The earlier `v321-campaign-planned.json` is a historical pre-submit receipt and must NOT trigger resubmission. No cancellations, duplicate jobs, release jobs, or full-model runs were submitted during this recovery.

## First actions on continuation

Read `squeue` and `sacct` for1641933/1641934, then campaign logs and `runs/*/{progress,complete,failure}.json`. Do not overwrite frozen source or resubmit pending/running jobs. Both jobs independently run the four canary geometries and emit a raw-evidence canary analysis after successful measurement. A scheduler exit0 alone is not a performance pass: read `analysis/canary-<partition>-<job>/summary.json`.

Only when both current-revision canaries pass may `authorize_release.py` recompute and authorize release qualification. The thirty v3.2 release geometries remain unexecuted; do not consume them while either canary is missing, failing, or stale. Release data cannot be reused as fresh after selector adjustment. The original v3 release set is already exposed and is not this new set.

## Retained failures and evidence

Old5090 job1641803 completed2304 timing rows and saved exact output/LSE hashes, but its frozen canary is HOLD: eager policy worst0.939482x, disabled-overlay worst0.944173x. SelectedGraph16 point worst1.343236x is not a release result and does not override those failures.

Old4090 job1641802 completed pristine32/32 and paired31/32 before CUDA OOM at the final BF16/paged/unsplit near-opposite case. Preserve its768 completed pristine and1488 partial paired timings. This was an allocation failure, not a reported numerical mismatch. Both old archives and hashes are committed.

## What revision3.2.1 actually changes

The selector rule and manifest are unchanged. Policy selection is performed by the new native C++ entry with explicit OFF/cap/guarded values; legacy entry defaultsOFF. Scored eager calls no longer reconstruct or inspect the FFI plan array. Unscored pristine calibration freezes one common call count across arms/repeats, and scored windows below12ms fail. Graph1 andGraph16 independently poison/check output andLSE. Case-local graph/tensor cleanup and per-case CUDA-memory receipts target the old OOM but require GPU verification.

The analyzer checks artifact/source hashes, ABBA/BAAB execution positions, complete matrices, raw identity payloads, actual plan decisions, fixed references/calibration, and memory records. Absolute and paired worst-case/control gates are both required. None of these CPU contracts proves that new GPU numerics/performance pass.

## Validation and independent blockers

38 CPU tests passed onPython3.12.13/NumPy2.3.5, along with compileall and shell syntax. All four workflows at source commit a849cf4 succeeded: mainCI36700075511, selector contracts36700075692, resource generalization36700075506, andSection636700075519. Nine official source hashes were verified; generated Python syntax and native export structure passed. The CPU-phase record correctly states that CUDA compilation was not yet verified.

PR27 remains a separate draft at `672822f29953c3f466212bdd00de48b1d32f847d`. SGLang#38788/#38850 concern test-harness changes; #38850 already diagnoses delayed cleanup aborting a replacement. Our HTTP production ownership guard must be coordinated, not claimed as the first discovery of the hazard.

The original2/432 full-length token divergence remains unresolved; smoke or forced-history equality does not close it. Default remainsOFF/HOLD. No new fullHTTP result is claimed. Max reasoning GUI selection has not been verified; do not claim it enabled. Report-only commits aftera849cf4 do not change the immutable GPU source snapshot.
