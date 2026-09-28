# Matched reuse milestone after 660d128

Completed28September2026. AI-assisted research and implementation; no prize or priority guarantee.

## What changed

The previous unamortized adaptive dispatcher remains **NOT_PROMOTED**. We recovered existing wave/native-policy experiments without duplicating them, derived a strict break-even test, implemented bounded cache/memory validity contracts, and executed a new direct selector+plan+GPU attention-stack experiment. It does not report CPU-overhead/layer-count as a measurement.

The new registered PRIMARY native-cycle policy passes only on **RTX4090 with512MiB scratch**, at36distinctlayercalls persegment, under unchanged,alternating andLRU-eviction state sequences. Fixed/native elapsed-time ratios are **1.1669,1.1416,1.1477**, with nominal95%familyCI lowerbounds **1.0684,1.0469,1.0512**. All three also beat the default-auto control. There are no >5% regressions among18basegeometries in each of these three conditions.

**128MiB andRTX5090 do not pass the primary gate.** The5090/512MiB policy improves overstock but not reliably over the TRAIN-selected fixed comparator. No headroom is manufactured by using stock alone.

See [complete primary table and sensitivity](results/direct_reuse_summary.json) and [gate interpretation](RESULT_GATE_20260928.md).

## Measurement and inference boundaries

The new corpus contains54exactgeometries from18bases and9query/cachefamilies, all disjoint from prior fit/test/discoveryexactinputs. BothGPUmodels have threeprocess/seedreplicates. No weights or decision thresholds were retuned. The18bases include n2/n8requests; this is a synthetic heldout-geometry test, not productiontraffic.

Each timed episode consists of4segments.36layers means144actualattentioncalls. Unchangedgeometry reusesoneplan; A/B/A/B andA/B/C/A rebuildoneplan per36-layersegment. Everyarm gets the same2-entrydecisionLRU, current-active-plan reuse, datatensors, scratchceiling and timingboundary.36layers have distinctQ/K/Vstorage anddata. No fullmodel weights,MLP,network orservingscheduler are present in this experiment.

The TRAINfixed comparator is chosen perGPU/head/budget/reuse count using oldTRAIN cycle+(r-1)*run as a selectionproxy. It is not claimed to be the empirically optimal fixed policy under newdatatraffic. All reported outcomes are actual newdirectwalltimes, andstockauto is also compared.

At4090/512MiB, the eviction condition already shows1.0740x at one layer persegment, then1.1251x at4 and1.1477x at36. Thus **amortization alone is not established as the cause of reversing660d128**: cache implementation,datareuse,head configuration,memoryfeasibility andinputgeometry all differ from that earlier setting. The original negative result is preserved.

Across all settings,3,888timingrows and5,557,248timedattentioninvocations completed. These invocations are repeated measurements, not independentproblems. Ninewholefamilies supply20,000nominalbootstrapdraws after cell-median reduction of3processrepeats. No multiplecomparison correction or hardware-population inference is claimed.

## Correctness and memory

32CPUcontract/statisticaltests pass;216freshmetadata native decisions match frozen fitted models before freshGPUtiming. The direct matrix passes22,248nonauto fulltensor comparisons and25,920selectedFP32vector checks. Maxabserror0.00048828125 versusauto and0.0001294017 versusFP32,atol0.005/rtol0.02,notbitwise. All36layers andallstates are checked for selectedmodes beforetiming; everytimedoutput is not copiedback andrescored.15,552arm/rowplan-cache-counter checks enforceplan,run,hit,miss andevictioncounts.

Scratchceilings128/512MiB are enforced withcandidate→auto→none fallback, not silent allocationgrowth. Equalserialscratchreservation is forcontrolledcomparison, not a claim that productionKVcapacity is unchanged. Maxlayer-tensorstorage7,832,567,808bytes, peakPyTorchallocated9,326,039,552bytes; referencechecks contribute topeaks. A fullservingcapacity tradeoff remainsunverified.

## Recovered mechanism and algebraic screening

The existing864-recordwaveexperiment completedbeforethiscontinuation. Its4prespecifiedcrossovercases atQ1344/Q2720 and16/32queryheads all match the directionalprediction:5090's equal-workshapecontrast exceeds1.1andthe4090contrast. The pairedtotalQ,totalcachedK,exactW andtotalKVlengthmarginals werechecked. Neighbors andallpoliciesremain inrawdata. These A/Bshape ratios are not optimization speedups. Genericwavequantization remainspriorart, andoccupancy/locality/masking are not fully causally separated.

The diagnostic linear model is:

`r * (U_fixed - U_candidate) > C_selector + (P_candidate - P_fixed)`.

For positive per-call saving, firststrictlyprofitableinteger is `max(1,floor((C+deltaP)/deltaU)+1)`. Zero/negative saving may yield no profitable count or a finitewindow. TestedexactFraction arithmetic handles those cases.

Oldmeasuredcosts indicate only46/1444090cyclecells and35/1445090cyclecells profitable at36; they do not establish directreuseperformance. Those calculations use the earlierctypesCPUcost onln01 andapproximatecycle-minus-CUDAsetup cost, not today's GPUworker/cachepath. See [recoveredmechanism anddiagnostic](results/recovered_mechanism_and_break_even.json).

## Official implementation boundary

The official22September2026Autotunerv2post requiresdeployment-matched eager/CUDAGraph measurement andvalidruntime/operationkeys; v0.7adds persistentwinnerreuse. POD andthepinned0.6.18raggedwrapper already documentcrosslayerauxiliaryreuse. Reuse,cachekeys andplan/run separation are NOT ournovelty.

Installed0.6.18prefill.py was hashchecked; itsraggedpath callscompiledplan/ragged_run directly andcontainsnoAutoTuner,choose_one,autotune_v2orMeasurementPolicy invocation. Runtimeexports lackv2entrypoints. Anofficialv2control isthereforenotapplicable withoutanexplicitnewrunner/versionintegration; we do notmislabelourfixedsplit asofficialautotuning. See [source-level applicabilityaudit](OFFICIAL_PATH_APPLICABILITY.md).

The supportedincrement is a reproducible,resource-bounded,backend-specific evaluation identifying where unchanged learnedplan choices have actualnetvalue underrealreuse andwhere theydo not. This is not aclaimoffirstphysicalplans,newattentionmath,universalscheduling,orstate-of-the-artserving.

## Real vLLM follow-up and current external blocker

Afterthe4090/512MiBgatepassed, weimplemented aprocess-localpagedrouter withpage-unitconversion andseparatephysicalpageplan lifetime, a36layerNHD/HNDtransition checker, andanactualvLLMEngine driver. Job1632883waslaunchednormally usinganisolatedvLLM0.29.0/FlashInfer0.6.18/PyTorch2.13/Python3.13runtime andcachedQwen3-4B-Instruct-2507weights. Thepageroutinesreachedtheirfinalstructuredoutput, then thelauncherenteredtheautoenginearm. No finalthree-armcomplete marker wasvisible atlastsuccessfulinspection.

The subsequentread-onlyrequestforauto.log,engine_receipts.json,pagedsummaryandsacctwasblocked:

`This tool call was blocked by OpenAI's safety checks. Please double check what you are sending.`

No morespecificreasonwasprovided. The sameaction wasnotreroutedthroughanother tool,REPL,encodingoraccount. Full-modelthroughput,TTFT/TPOT,tail,SLOgoodput andgreedyoutputparity remainUNVERIFIED. This is not aninferredGPUfailureorclaimofmissingGitHubpermission. See [exactcheckpoint andminimalauthorizedrecovery](ENGINE_CHECKPOINT_20260928.md).

## Jobs, publication and evidence

Canaries1632714/1632715:completed,380allocatedGPUsecondscombined. Fullarrays1632759_0/_1/_2 and1632760_0/_1/_2:completed,5,833allocatedGPUsecondscombined. Totalnewreuseexperiment6,213GPUseconds=1.7258GPUhours; laterenginejobcostisnotyetverifiedandisnotincluded. Existingwave/nativejobswerenotduplicated.

Importantcommits:
-93efc8714919a9a6f9ebbe9f498d73b9cea76bf9:prospectivereuseprotocol.
-ddfd23a9e3aa5661457f2c4626b7d9e899042963:frozennewGPUmeasurementcode.
-6d1c3e732a43dea0034219076c5ff48c17f97915:canary/paritygatebeforeindependenttest.
-6da22af22877998cb794ca42b0f59b9bb15fda7b:fullprimarydata,sensitivityandrawhashes.
-18c51f68396aca9b65bb8005e3f85517e1d32dd3:actualpaged/fullenginequalificationsource.
-54d6a19ddfe0a5d9dac43bb6726b4696448dcf7c:exactengine-readblockercheckpoint.

AllpublicationusestheallowedGitHubAPIflow onexistingbranchresearch/causal-plan-cost-20260928. No successfulterminalgitpush isclaimed; mainanddirtyotherworktreeswerenotoverwritten.

Remote root:
`/ssd/scxi253/single-gpu-inference-plan-cost-20260928`

Completedrawandmanifests:
`campaigns/reuse-gate-v1/runs/{canary,test}-*/`

Reproducibleaudits:
`artifacts/reuse-gate-audit-v1/`

Completealready-auditedreuse/waveevidencearchive (excludesunverifiedenginelogs):
`artifacts/reuse-gate-audit-v1/completed_reuse_evidence.tar.gz`

Archivebytes872,135;SHA256 `14d7ec604e4849677173bb2dc23aa9b8399f7e777b7d9beec8d384d6ba51413d`;51indexedsource/resultfiles. AuthoritativefullmetricsSHA256 `2cd464b70015eeef2089ed427876bc96a29c48079970b4fcdf08e9f9b9cd456a`;per-cellSHA256 `a3e74ea71e69ff88d12657f27d6589c2dbc578fa7694cc7e4319f3dc1981cb07`.

## Reproduce without new GPU jobs

Use the pinned measurement source andunchangedrawfiles; neveroverwriteoriginalauditresults.

```bash
P=/ssd/scxi253/single-gpu-inference-plan-cost-20260928
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312/bin/python
S=$P/artifacts/reuse-gate-audit-v1/research/causal_plan_cost/reuse_gate
export PYTHONPATH=$P/analysis-deps:$P/campaigns/native-policy-v1/research/causal_plan_cost:$S
cd "$S"
"$E" -m unittest -v test_contracts test_analyze_reuse
OUT="$P/artifacts/reuse-reproduction-$(date -u +%Y%m%dT%H%M%SZ)"
"$E" analyze_reuse.py --root "$P/campaigns/reuse-gate-v1" --out "$OUT"
sha256sum "$OUT/metrics.json" "$OUT/per_cell.json"
```

Re-runningGPUbenchmarksrequiresanewcampaign/outputrootandfreshsourcebinding,notresubmittingcompletedarrays. Enginecontinuationiscurrentlyconditionalonresolvingtheread-onlyblocker,notanunattendedfuturepromise.
