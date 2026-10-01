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

## 2026-10-01T11:24Z final bab normal source gate complete

Dual69/managed persistence and reload jobs1645930/1645931 COMPLETED0:0; fresh dual SASS352pairs PASS. All compact build/functional/full raw SASS archives locally verified member-by-member and preserved in v42-normal-module-bound-wheel-{build,functional,sass}. Both graph modes select resource; eager staysnative on both.5090 reproduced a12.745ms resource profiling outlier after accepted confidence; cause remainsunknown. The earlier48fits did not establish its cause.

Public fullplan setup jobs1646009/1646010 are active. Partial observations show new-constructor seconds vs native2ms, so prepared kernel gains cannot qualify this per-plan construction path. Keep all32 preregistered observations and analyze with analyze_plan_setup_cost.py before engineering a reuse boundary. No newcanary/release/stress dispatched; default/servingOFF, historical2/432unresolved.

## 2026-10-01T11:34Z public construction cost diagnosis complete

1646009/1646010 bothCOMPLETED0:0;32observations/GPU,alloutputexact. Native/full-public meanwall4090=2.462230/5538.737103ms,5090=2.023024/9798.872406ms. Publicconstruction alone5535.991101/9797.046026ms.4090confidenceinconclusive→native;5090accepted→resource1. Wholepublicconstruction pathcannotbeusedperplan.43members locallySHAverified, evidence/v42-public-plan-setup-complete/.

NewlocalFlashInfer samegeometryeager rebind implementation is uncommitted and unqualified:13 CPU boundarycases and4GPU functionalcasesadded. PrivateCPU patchproof /ssd/scxi253/sgi-eager-rebind-cpu-20261001T1133Z dispatched after5file patch/helper upload completed;66CPU/20GPUskip expected. It uses copiednormalbabpackagewith explicitpatchedsourceledger, neverclaimsnewnormalwheel. NeedCPUPASSbeforecommitanddualGPUfunctional; thennormalnewwheel,completeplan-costdiagnosisandformalnewrevision. No fresh cases consumed.

## 2026-10-01T11:57Z safe eager samegeometry reuse source committed; normal package gates active

FlashInferHEAD d3d080ce36bf074136d4b2093b775a8cb1d7dfc5;13 newCPU rebind-boundary tests+4GPU cases. CPUprivatepatchproof66PASS/20CUDA-skip268.05s. All5 patchedsourcefileSHAs matchthiscommit; GPUexample is the sixth changedfile and outsideCPUproof. Firstlauncherfailedbeforetests duewrongprivatepytest path; originalhelper/logs preserved. Retry usedunchangedownedoverlay and overlay/test-support, noenvinstall. Evidence/v42-eager-rebind-cpu/. Local29source/policytests andfull14fileRuffformat/check PASS.

rebind_same_geometry is explicit eageronly: requires previouslyeligible/preparedmodule and accepted immutablecertificate; verifies fullplanvalues/module/options/stream/compileroptions/tensor-signatures/actualorderedmetadata. Failurepermanentlydisablesresource. Package/compilerfilesmustremainimmutable; no sourcehash/bootstrap/recompilation inside rebind. It doesnotqualifyGraphmetadatawrites orHTTP. Example onlyrebinds anactualmanagedresourcewinner; nativewinner staysdirectnative.

NormalbuilderPID2398314 andfinitewheelgatePID2398527 active, build /ssd/scxi253/sgi-flashinfer-official-build-d3d080ce-20261001, expects unchangedupstreamhook0.7.1 metadata andGitpinnedvendorheaders. Then dual86(66CPU+20GPU)+example+real3managedmodes, freshSASS, then oneexposedfullplan-rebinddiagnosis/GPU. Finite sourcegate /ssd/scxi253/sgi-eager-rebind-source-prequalification controls onlythesequalifiers, nofresh/release/stress/promotion. Helpers hashlocked beforelaunch; do notmutate them afterthischeckpoint.

Researchharness58da590d65abba94d3ffb9df89b40755779170ea, planned /ssd/scxi253/sgi-public-plan-rebind-58da590-d3d080ce-20261001, fixed32shortcalls8ABBA/BAABblocks/GPU, oneprocess, actualplan+metadata-validation/rebind+managedrun. Managedprofiling_repeat fixed256; notcausalproof about12msoutlier. Initialsourceaudit/JIT/calibration/training outsidecalls. Inconclusiveconfidence oractualnativewinner retainsdirectnative. Allcompletedrows durable beforeassert; noformalreleaseauthority.

Frozenv4.2dualHOLDstillstands; no v4.3freshsource/casesfrozen. New10canarynotgenerated/dispatched;48release/12stress untouched; historical2/432unresolved; fullresourceHTTP unqualified; default/servingOFF; twoPRsstilllast.

## 2026-10-01T13:16Z normal d3 dual86/SASS and actual rebind cost complete

Normal d3d080ce wheel65833431bytesSHA6c6c9dfc85cf45924c114cdc5a0b64680d912a9bdf691c2e6ce6e5c227eff76a,10126sourcefilesverified.1646256/1646257COMPLETED0:0 dual86(66CPU+20GPU),example,all3realmanagedmodes.352native/resourceSASSpairsPASS. Allrawbuild/functional/SASSarchives locallySHAverified andevidence/v42-normal-eager-rebind-wheel-*. No thresholdrelaxation/promotion.

Fullplan-rebind diagnostic1646459/1646460COMPLETED0:0,32calls/card,alloutputexact,bothrealcertificateaccepted+actualmanagedresource1,16resource rebindcalls/card. Mean native/rebind4090=2.367619/2.163291ms(1.09445218),5090=1.956714/1.762161ms(1.11040585). Independentofpriornewconstructor seconds; actualplan/metadatavalidation/managedrunincluded. Exposed oneprocessshortcalls only, noformalreleasecontrols/CI. RawarchiveSHA8fe5dc99fae2da6fb78f1fec91dfc2460d4ecbd60cd05196dbba9289e5d3322b,231028bytes, locallyverified. evidence/v42-public-plan-rebind-complete/.

Applying repo self-review skill aftersourcefunctionalcompletion. gitfetchupstreammainnow1cad3816,11commitsafterpinned85744da1. TargetedFA2prefill/scheduler/C++/nativePython/JITcore/compiler/generator/vendorpins/version diffempty. Unrelatedcake kernels/pyproject additions advanced; do notclaimnewupstreamfullpackage qualified bypinnednormalbuild. Needkeepactualbase/hash inreview.

Self-reviewfoundTensorcontroloptions onlyboundshape/stride; nowuncommitted hardening bindsaddress/version,rejectsunknowninferencecontrols early,preservesoutput/LSEscratchwrites, adds3CPU tests+1rebind parametercase. PrivateCPU patchproofPIDpending at /ssd/scxi253/sgi-tensor-control-cpu-20261001T1315Z expects70CPU+20GPUSKIP,readonlydependency symlinkoverlay overnormald3source,3patchedregularfiles. No oldoverlay source mutation. README fourthchangedfile outsideCPU proof; sourcecommitrequiresCPUPASS. Currentnormald3 resultsdonotcovernewsource.

ResearchHEAD updates normal/rebindarchives. No newv4.3protocol orfresh10cases frozen/consumed, original48/12untouched, frozenv4.2dualHOLD/historical2/432unresolved, fullresourceHTTP unqualified, twoPRslast. Nextnormalnewsource+90tests+SASS beforeformal newrevision/development route.

## 2026-10-01 Tensor-control CPU proof committed; normal dual90 pending

75544a17ce0019ca877f50354d95451ee089f859 commits four reviewed files. Private proof70CPU PASS/20GPU skips,3 tested committed files match raw receipt SHA; README separately committed. Evidence/v42-tensor-control-cpu includes original log/XML/patch/helper and committed verification. Normal candidate builderPID858462 and finite gate858475 dispatched only after bundle/helper uploads completed. New wheel must pass dual90(70CPU+20GPU),3 real managed modes and exact native/resource SASS. Previous d3 normal86 remains evidence for d3 only. No new fresh case or release/stress consumption; defaults/serving OFF; old dual HOLD and2/432 remain.

## 2026-10-01T14:24Z exact candidate/pristine normal builds pass; independent public-path probe prepared

75544a17 normal0.7.1 wheel65833543bytes SHA2044882df7413bff1946b5eb78d291c0f6fb715c55263d5e1dd5fa6b28dc94d1,10126files verified. Independent pristine85744da1 wheel65818279bytes SHA2f2daa468b0342cd4d9cd4d68b96c61a8c6a6b07928c78b66f5cf8d15d8c1adc,10121files verified; both current fixed vendor pins and genuine build hooks, no environment install/version override. Dual90 jobs1646876/1646877 active;5090pytest90PASS observed, complete modes/Slurm exits/SASS still required.

Finite strict process-handoff gatePID1267945 waits exact90/SASS, then producer+two independent consumer processes perGPU. Consumers do not calibrate/tune; actual search_cache hit is required including native-1 winners. Old gate1196013 cancelled before any GPU job, original hashlocked helpers/receipt preserved; stats.cache_misses did not prove a hit. Scope is functional prepared eager/Graph1/16, eager rebind and same-pointer Graph payload update, not metadata/fullHTTP/performance qualification.

Researchbb9d764 public_path_prequal.py and public_path_contract.py prepared with7 local contract testsPASS/Ruff/diff checks. Only two exposed dev cases; each GPU/case uses3 genuine-pristine,3 training and3 new policy processes, with384ms target/120ms minimum/24blocks/no trimming. Two other training processes decide each held-out choice, fixed primary artifact rotation; real accepted certificates and v2 resource winners plus separate full-call training gain/control gates. Choice is persisted before scoring; held-out oracle never trains. Actual eager includes plan/rebind/run, native winners go directnative; Graph1/16 are replay-only. No newv4.3protocol/manifest frozen or fresh case consumed. Full source/consumer prerequisite must finish before exposed dispatch. This development probe has no canary/release/default/HTTP authority.

## 2026-10-01T14:53Z exact75544 normal90/SASS proof locally archived

1646876/1646877 both COMPLETED0:0,19m41s/16m55s. Dual90 plus example/real managed modes/persistence/reload,352SASS pairs PASS. Build37members,functional48members and all raw SASS locally member-SHA verified. Independent pristine85744 build37members also locally verified. Evidence/v42-normal-tensor-control-wheel-{build,functional,sass}/ andv42-normal-independent-pristine-wheel-build/.4090prepared eager duplicate controls inconclusive→native-1; bothGraphsresource1;5090all3resource1. Means alone do not authorize resource.

Process-handoff1647026/1647027 producer+two new consumers/card, actual confidence reload and v2 search_cache hit checks;5090finished0:0,4090active at last read. Original unconditioned public-path gate1943718 cancelled before performance dispatch to add fixed30s native-only conditioning after compile/cache hydration, preventing cold pristine/warm candidate comparison. Old helpers/receipt unchanged. New researchc6879fb, scopeEXPOSED_PUBLIC_PATH_PREQUAL_2, privatehashlocked helperroot /ssd/scxi253/public-path-helpers-c6879fb, archiveSHAc8cc4ecb295a78767a5a8313061bc69122e2a17e73b1acb4229376d0bcc75c93, finite gatePID2437314. Still only two exposed dev descriptors and no canary/release/default/HTTP authority.

## 2026-10-01T15:23Z independent handoff proof complete; exposed public-path jobs active

1647026/1647027COMPLETED0:0,17m51s/15m50s. Six genuine processes/card-specific source and GPU/input/native O+LSE hashes match, actual v2 disk hits and stored winners match; consumers nevercalibrate/tune. Eager rebind and Graph1/16 same-pointer payload updates exact. Wholearchive locallySHA reverified, evidence/v42-public-process-handoff-complete/. This is not metadata-update, fullHTTP or performance qualification.

Exposed public_path prequalc6879fb started14:51UTC:4090cases0/1=1647198/1647199,5090=1647200/1647201. Each bounded8h job uses3 pristine+3 training+3 newpolicy processes onone allocatedphysicalGPU, uniqueJITroot/phase, fixed30s nativeconditioning percell/mode afterinit,384ms target/120ms minimum/24balancedblocks. No freshcaseconsumption. Finite completionPID3033512 waits all4jobs, independently compares pristine/native/resource SASS and analyses allrawrows with20kbootstrap andunchanged0.99 floors. It cannot submit any newfresh/48release/12stress/HTTP/PR. Original dualv4.2HOLD and2/432remain; defaults/servingOFF.
