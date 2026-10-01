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

## V4.1.1 exact-source dev final verdict

Campaign `/ssd/scxi253/single-gpu-inference-selector-v411-dev-20260930T211230Z`, jobs1643854/1643855, source `50f182d`, completed0:0 with12/12 run receipts and no failures. 4090 retains +38.76% selected geomean and +32.02% selected worst, but seven selected folds fail held-out duplicate-control equivalence. 5090 retains +40.00% selected geomean and +30.71% selected worst; all tactic controls pass, but candidate-native/pristine simultaneous joint-min95% LCB is0.988980x. Exact artifacts and remote hashes are under `evidence/v411-exact-dev/`.

Do not submit canary. V4.2 must preserve official native plan/run and put cap behind a separate resource-only run entry; increase measurement resolution without relaxing gates. The10 canary,48 release and12 stress shapes remain untouched.

## V4.2 native-transparent implementation

Continue on `kevin/selector-v42-native-transparent-20261001`. Official 15-field plan,
native Python/C++ host runs, native dispatch and kernel source are audited unchanged.
New opt-in `run_resource` entries are used by the measurement runtime. CPU/source
integration: 48 tests pass, including distinct torch custom/fake op registration,
executed resource eligibility guards and source mutation rejection. Measurement
uses 384ms target/120ms minimum windows and 24 balanced blocks, 3 process repeats.
Native SASS identity and actual policy/pristine performance are blocking gates.
Holdout case geometry/hash is unchanged. Exact-source dual-GPU smoke is next.

## 2026-10-01 exact v4.2 dual-GPU smoke PASS; dev dispatched

Frozen measured source `b769e7c18e93c9d6cfdcd79b58edf76955313859`, archive `60ffb32e73e8e9b2ce0e84f2ca20ad3c4559ecbc651cd86c5e168c03074f3d7e`. Campaign `/ssd/scxi253/single-gpu-inference-selector-v42-20260930T230508Z`. CPU 51 PASS. Smoke 1644061/1644062 COMPLETED 0:0; both native SASS identity true, both dtypes/layouts native/resource/native exact with official 15-field plans. Exposed dev two shards per GPU: 4090 1644068/1644069; 5090 1644070/1644071. Both analyzed PASS required before canary. Holdouts 10/48/12 untouched; all defaults OFF. Exact smoke receipts and compressed raw disassembly archived under `evidence/v42-exact-smoke`. Earlier smoke 1643997/1643998 failed missing nvdisasm and remain supporting numerical evidence, never authorizing dev. NVIDIA cuobjdump/nvdisasm 13.0.85 packages now SHA256 verified.

## Current-main HTTP ownership PASS (independent lifecycle fix)

SGLang ea34a9d: all 49 registered CPU tests passed with full imports. ParaCloud current-main Qwen3-4B full-model job1644085 COMPLETED0:0. Original8/10 expected-complete requests vs candidate10/10; one additional intentional n=2 disconnect per arm. Both original reused-RID replacement requests were aborted by stale cleanup; candidate had zero stale aborts and no owned state left at cleanup. Raw source/SSE/log/metadata archive and SHA manifest: `evidence/v42-http-ownership-main/`. Resource cap disabled; original2/432 token-value divergence remains unresolved.

## V4.2 dual dev PASS; canary consumed and active

Exact sourceb769e7c dual dev passed every requirement after all four jobs1644068–1644071 COMPLETED0:0. Proof: `evidence/v42-exact-dev/`,225 raw files SHA-bound; actual policy/pristine means4090=1.383822,5090=1.285451. Frozen thresholds unchanged. Ten fresh canary shapes consumed at2026-10-01T01:43UTC, jobs1644188–1644207. Release48/stress12 untouched at this checkpoint. Follow finite controller `/ssd/scxi253/single-gpu-inference-selector-v42-20260930T230508Z/logs/kernel-pipeline.log`; no resubmission of consumed stage. On dual canaryPASS it submits release, then only releasePASS opens stress. HTTP serving/performance and historical token attribution remain independent gates and default/serving stayOFF.

## Latest pinned SGLang request-lifetime PASS

Rebased independent patch3825c5a4 on upstreambaae019:49 full-import registered CPU tests PASS. Full Qwen3-4B BF16 real-HTTP job1644258 COMPLETED0:0 in12m56s with isolated official FlashInfer0.7.0.post1; original8/10 and candidate10/10 expected completions, plus one intentional n=2 disconnect per arm. Candidate zero stale-owner aborts and zero remaining owned states. Source/SSE/events/logs archive locally verified: `evidence/v42-http-ownership-current-main/`. This supersedes the earlier dependency startup failure as the ownership verdict; all failure receipts are preserved. Resource tuning/performance and original2/432 attribution are independent and still unqualified.

## Current-main experimental prepared runner: dual GPU PASS

Source d7683dd/native85744da1, jobs1644585/1644586 COMPLETED0:0.49 tests perGPU(37CPU+12GPU), including page1/noncausal, plus source-bound managed-v2 eager/Graph1/Graph16 persistence/reload/capture exact checks. 4090 eager duplicate controls are inconclusive and selectnative; graph gates pass.5090 all three selectresource. Evidence: `evidence/v42-current-main-managed-v2/`. This does not certify dynamic metadata updates or full serving. The metadata-lease bridge270b26c is separately submitted as exposed functional jobs1644734/1644735; no fullHTTP tuning or promotion is authorized yet.

## 2026-10-01T10:56Z durable checkpoint: package and serving geometry gaps repaired

The frozen b769e7c campaign is still dual HOLD.4090 completeness has been recovered with all60 original/resumed processes, but its held-out controls fail;5090 native/pristine also fails0.99. Both full raw archives are locally reverified. Release48/stress12 remain untouched. No new v4.3 protocol, canary source freeze or freshness dispatch exists yet.

Normal eb4e3e60 upstream0.7.1 wheel uses actual CCCL/CUTLASS/spdlog Git pins. Dual53 package tests, prepared managed modes/persistence,352 SASS pairs, dual10 Graph-boundary tests and dual6 real-stride/implicit-page1 tests passed. All are experimental fixed-geometry evidence, not serving/release qualification. Final native-context diagnosis1645468/1645469 is complete:three processes and7,776 durable rows per GPU, all exact, no trimming.5090 has3 unresolved native/native +/-0.5% controls. Evidence is under evidence/v42-native-context-pinned-complete/.

The original full-Qwen geometry observer mistook KVheads8 for implicit page1. Keep its original raw archive and mark paged derived lengths invalid. Corrected observer8f9ab99/job1645581 completed7 samples and54 startup/HTTP observations with raw indptr/tails, page1 and ragged stride6144. All112 native resource-OFF tokens match the original run. This does not close2/432 or provide timing/resource-parity evidence. Actual noncausal prefix Q1024/KV128 exposed a constructor guard gap. FlashInfer13ae45c2 repairs it; normal pinned wheel and dual61 tests/three managed modes passed. Its initial archiver stopped before Slurm completion; separate partial attempt is preserved.

FlashInfer branch codex/flashinfer-native-resource-v42 now HEAD bab48695bbe72ba23c49bad69818b2975f3de6f7. It additionally rejects stale process-module source/compiler envelopes, including different dtype templates, and source changes before first private spec generation. Full CPU patch proof53PASS/16GPU-skip and local29 source/policyPASS are preserved. Final normal wheel SHA f311c2f01a928de03cbe07df55f7afaab1df11fa34d421edf754ef8c86c0784b; actual metadata0.7.1/bab48695;10,126 package files verified against normal build/source ledger. Dual69 jobs1645930/1645931 are active at this checkpoint.

Finite source gate /ssd/scxi253/sgi-module-bound-source-prequalification waits for both final jobs to finish, archives dual69 and normal build, audits fresh native/resource SASS, then dispatches exactly one exposed public setup-cost diagnosis per GPU. It cannot dispatch fresh canary/release/stress or promote. The52d7508 setup harness includes native plan, public runner construction/certificate reload and actual managed run; retains every completed sample, including native fallback when confidence fails. Its32 short observations are descriptive, not release-resolution evidence. Source/JIT/calibration initialization is outside scored calls. Measure this cost before designing deployment integration or claiming kernel gain transfers to serving.

SGLang3825c5a4 ownership fix remains separate,49 CPU and complete-model production-HTTP PASS. Historical2/432 token-value mismatch remains unresolved. No issues/PRs pushed or created; no new chat/automation/goals. Keep defaults OFF and follow source->dual smoke->two exposed dev->new10 fresh canary->release48->stress12->full HTTP/token/Graph boundaries->two independent upstream PRs.
