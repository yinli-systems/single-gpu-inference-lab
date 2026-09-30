# Resume selector v4.1 safely

## Branch and state

- Worktree: `/Users/kevin/Projects/sgi-selector-v4-safe-autotune-20260930`
- Branch: `kevin/selector-v4-safe-autotune-20260930`
- Last committed provenance checkpoint before v4.1 integration: `5849c16f5ac6ad78643b8876cab9145491d8f05e`.
- V4.1 integration is not valid GPU evidence until committed source, immutable campaign receipt and job IDs are recorded below.
- No active v4 jobs were present when integration resumed. Do not repeat v3.2.2 96/96 release.

## Persisted evidence to reuse

- Old-overlay safe dev jobs1643638/1643639 completed. Reanalysis from campaign manifest/source is HOLD only because duplicate controls do not resolve; use only to validate cross-fit/cache machinery.
- Sibling isolated-symbol private-cache smoke jobs1643750/1643751 produced exact dual-GPU outputs and compiled co-resident symbols, but Slurm ended nonzero due an obsolete audit path. Copied under `evidence/kernel-isolation-smoke-r2-legacy/`; supporting development evidence only.
- First canary jobs1643723/1643724 used an aliased shared JIT cache and are invalid/retained.
- Sibling jobs1643750/1643751 and all old failures must not be rerun as if fresh.

## V4.1 implementation delta

- separate native/resource ragged+paged kernel symbols;
- legacy plan defaults native; explicit policy supports only native0/cap1;
- runtime requires an isolation marker before applying cached cap;
- unsupported/ineligible cap becomes an explicit second-native null arm;
- native-after-cap exact check per paired cell;
- identity includes binding hash and shared-memory limits;
- 96ms windows, 16 blocks/process, >=32 training blocks, 20k bootstrap draws;
- per-shard compiled binary audit and active-clock telemetry gate;
- all sibling-v4 shapes added to the historical freshness ledger.

## Next admissible action

1. Finish full remote CPU/source validation and commit the exact v4.1 source.
2. Build a new immutable dev-smoke campaign from the official0.7 pristine overlay using `prepare_campaign.py`.
3. Submit exactly one smoke job per GPU using `smoke_v4.sbatch`; use private pristine/candidate JIT roots.
4. Require exact ragged/paged outputs for native-before/cap/native-after and a passing compiled binary audit.
5. Only then run the 2 exposed dev cases. Do not consume the 10 canary shapes until dev controls resolve and all source/binary contracts pass.
6. Canary/release/full HTTP remain OFF/HOLD until their own gates pass. Historical2/432 divergence remains unresolved.

## Reasoning-setting limitation

The UI showed Pro power5/5, but an option explicitly named Max could not be verified. Do not claim Max was enabled.

## Campaign preparation failures and stage-hash fix

Two preparation attempts produced no GPU work. The first failed in the shell before Python because output redirection targeted a root that did not exist. The second generated an overlay but failed before campaign receipt creation on a missing sibling-only `release_hash`; its partial root and receipt are retained. No jobs were submitted and no dev/canary/release/stress case was consumed.

V4.1 now records immutable `stage_hashes` for each family and validates them in the manifest contract. The full38-test contract passes after this fix. Build the next source archive from the commit containing this note; do not submit from `e3121b0`.

## V4.1 dual-GPU smoke PASS

Immutable source `8ea2169dbc2967e02f12e527ea0b304a264df1ab`, campaign `/ssd/scxi253/single-gpu-inference-selector-v41-smoke-20260930T202814Z`, jobs1643827/1643828 both completed0:0. Exact ragged/paged hashes, native0-cap1-native0 plan sequence and compiled symbol co-residence pass on both GPUs. Smoke authorizes exposed dev qualification only; canary remains untouched and unauthorized until dev controls resolve.

## Persistent JIT launcher correction

Before dev submission, audit found that `run_v4.sbatch` placed XDG JIT binaries in a temporary directory deleted on exit while the required binary audit scanned the persistent campaign workspace. No v4.1 dev job had been submitted. The launcher now uses persistent per-mode/per-repeat XDG roots under the campaign cache; a new exact-source smoke is required before dev.

## Native-overlay qualification hardening

A pre-canary audit found that v4.1 cross-fit correctly measures off/cap but treated a native tactic as a mathematical 1.0 policy result. The raw campaign already contains independent official-pristine and candidate-native processes. Analysis now separately gates their performance equivalence: cell worst and simultaneous joint-min95% LCB must both be >=0.99, and pristine/candidate-native duplicate-control90% intervals must fit reciprocal +/-1%. This audit never trains the tactic and does not relax any threshold. Because the analysis source changed, any canary campaign requires a new exact-source smoke after the dev evidence is reanalyzed.

## Linux freshness-ledger correction

The v4.1 manifest previously recorded `research/parity_remediation/evidence/manifest.json`, while the tracked file is `MANIFEST.json`. macOS case-insensitive lookup hid this; ParaCloud Linux validation rejected it. The key is corrected without changing any case geometry, case hash, stage hash, selector rule, threshold, or consumed evidence. A new source commit and exact-source smoke are required before canary.

## Qualification revision 4.1.1

The native-overlay gate changes promotion semantics, so cache/measurement/binding identity is bumped from 4.1.0 to 4.1.1. Authorization now recomputes and binds the exact analyzer SHA, measurement-manifest SHA, campaign source commit, source archive, overlay, stage/gpu identity, and explicit native-overlay requirements. Old 4.1.0 summaries or caches cannot authorize a 4.1.1 canary. The currently running two-case dev campaign remains immutable 8ea2169/4.1.0 development evidence only; after analysis, exact-source 4.1.1 smoke and dev qualification are mandatory before touching canary.
