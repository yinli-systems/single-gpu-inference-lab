# Engine checkpoint after the direct reuse milestone

This file reports a current tool boundary, not a missing SSH/GitHub permission or an inferred engine error.

## Completed and independently verified before this checkpoint

Direct reuse: six complete GPU runs,3,888 timing rows;4090/512MiB passes both stock-auto and TRAIN-fixed gates at36 distinct layers per segment under unchanged/alternating/eviction sequences.128MiB and5090 primary routes do not pass. Sourcecommitddfd23a; detailed data in results/direct_reuse_summary.json and RESULT_GATE_20260928.md. No reinterpretation of the earlier660d128 unamortized rejection.

Existing wave/native-policy evidence was collected, not duplicated. Fresh direct canaries1632714/1632715 and full arrays1632759/1632760 completed. Other healthy user jobs were not cancelled or altered.

## Actual follow-up implementation and execution

Full-model qualification protocol was committed in3df5a40de45f30b75ba62c395d87b327f2c7c303 only after the direct scoped gate passed. The process-local paged adapter, cached-page transition checker, actual vLLM engine driver and bounded launcher are committed through18c51f68396aca9b65bb8005e3f85517e1d32dd3.

The ordinary authorized terminal submitted **job1632883** on RTX4090. Its immutable campaign is:
`/ssd/scxi253/single-gpu-inference-plan-cost-20260928/campaigns/reuse-engine-v1`

Run directory:
`runs/job-1632883`

Existing isolated runtime archive was inspected: vLLM0.29.0,FlashInfer0.6.18,PyTorch2.13.0,Python3.13. Cached Qwen3-4B-Instruct-2507 has36layers,32/8heads,D128,threeweightshards totaling8,044,982,000bytes. The screen requests convertedFP16,TP1,eager,512MiBworkspace,2048KVblocks of16tokens. This is distinct from the ragged assay.

Observed progress before the refusal:
- `paged-qualification/` and `paged-qualification.log` exist. The log reached the final structured qualification output, including NHD/HND records and actual mode choices; the final record shows68.2031678seconds inside the checker.
- The launcher subsequently created `arm_order.json`, `auto/`, `auto.log` and `engine_receipts.json`. It only enters the engine loop after reading a passed paged qualification summary.
- The most recent successful squeue read had no active row for1632883. The three-arm `complete.json` file was not present in the visible top-level file listing.
- The final paged numerical summary and full engine receipt/error tail were NOT independently audited before the next read was refused. Do not infer the engine exit code, crash cause, model success rate or throughput from filenames.

## Exact current blocker

A single **read-only** Remote_Desktop_Commander.start_process action requested the tail of `auto.log`, the contents of `engine_receipts.json`, a compact paged summary, and `sacct -j1632883`. It returned exactly:

> This tool call was blocked by OpenAI's safety checks. Please double check what you are sending.

It returned no processID or evidence for that command. The safety layer supplied no more specific reason. We did NOT retry those same reads via the existing REPL,another tool,encoding,account or approval setting. No failed-engine explanation is invented. Previously successful SSH reads,archive deployment,GPU submissions and explicitGitHubAPIwrites remain successful; the refusal does not imply every capability is unavailable.

## Minimal recovery

After that read is authorized, or when the owner directly supplies its output, check only:

```bash
P=/ssd/scxi253/single-gpu-inference-plan-cost-20260928
R="$P/campaigns/reuse-engine-v1/runs/job-1632883"
squeue -j 1632883
sacct -j 1632883 --format=JobID,State,ExitCode,ElapsedRaw -P
cat "$R/engine_receipts.json"
tail -n 60 "$R/auto.log"
cat "$R/paged-qualification/summary.json"
```

Do not resubmit1632883 or rerun completed reuse experiments just to regain context. If an engine task is healthy, preserve it. Classify any real interface/runtime failure from the actual log before patching. Do not promote paged/full-model performance merely because the preceding ragged gate passed.

## Remaining claim boundary

Full-model three-arm comparison,greedyoutputparity,independent engine repeats,TTFT/TPOT,tail latency,throughput,andSLOgoodput remain **unverified**. The directly measured positive result remains the scoped4090/512MiB attention-stack result, not a serving speedup. This AI-assisted work does not assert first invention of reuse or a Distinguished prize guarantee.
