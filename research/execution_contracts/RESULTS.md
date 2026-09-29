# Execution-contract integration: audited results

Base: `42a000d`. New additive branch; frozen source, protocols and GPU jobs unchanged.
This turn submitted zero new GPU jobs. It implemented CPU validation and audited already-running/completed real GPU experiments. No production or serving promotion.

## Validation

75 new tests and 31 existing resource-policy/analysis/serving/numerics tests passed on the remote Linux Python environment. The new suite includes 10,000 synthetic geometry property cases, exact representation checks, lifecycle and HTTP integrity checks, corruption rejection and pipeline failure tests. Synthetic fixtures are not GPU performance data.

28 synthetic fault injections: the byte-pinned old HTTP reader accepted 19 inconsistent fixtures; the new reader accepted zero. This does not imply historical GPU artifacts were corrupted.

## Frozen confirmation results

Each GPU: 48 mode/repeat/shard processes, 147,456 timing records and 2,304 qualification records. Process clusters are not independent GPU populations. All original cells, controls, slowdowns and exclusions are retained.

|GPU|Layout|Calls|Pristine/cap run ratio|95% conditional interval|Worst point ratio|Unresolved controls|
|---|---|---:|---:|---|---:|---:|
|4090|ragged|16|1.137267|[1.136443, 1.138087]|0.935468|9|
|4090|paged|16|1.134747|[1.134080, 1.135337]|0.936823|9|
|5090|ragged|16|1.099887|[1.099263, 1.100509]|0.952200|10|
|5090|paged|16|1.093932|[1.093391, 1.094473]|0.943035|7|

Ratios are native time divided by candidate time, NOT latency-reduction percentages. These are fixed-buffer graph operator measurements, not HTTP speedups. Other one-call and plan-plus-run results are retained in the JSON evidence, not selected away.

Cap passed all 576/576 exact-output qualifications on each GPU. Wide is a separate numerically different candidate and is not promoted.

## Full-model HTTP gate: QUARANTINED

Jobs 1639805/1639806/1639807: 1,296 completed requests. Each per-job artifact passed the strict integrity/derived-metric audit. Cross-mode token parity failed in job 1639805, so the gated statistical speedup table was not produced.

|Request|First different token index (zero-based)|Differing positions|Length|
|---|---:|---:|---:|
|decode-11|74|24|128|
|decode-13|114|13|128|

The same two differences appear within cap between decode block 0 and block 1. This establishes instability in that observed mode, not the first differing operator or the cause. The original logs have no sufficient batch-signature/logit snapshot to establish that attribution. No mismatched sample was deleted or replaced.

Formal GPU contexts also span drivers 580.105.08 and 580.82.07. This is an additional fixed-context generalization limitation, not silently normalized away.

The separate profile records 252 64-KiB launches for cap and zero for pristine/off in job 1639805. The count is retained only as diagnostic evidence; it is not itself a complete causal or per-request eligible-path attribution.

## Exact-feature witness search

Five declared representations were scanned against each fully audited confirmation summary (768 cells per GPU per scan). All ten scans returned zero same-feature/opposite-decision candidates. This is a retrospective search; absence is not a sufficiency proof, and no novel measured decision-reversal result is claimed.

## Evidence and limitations

See `evidence/20260930/validation-summary.json`, `gate.json`, `parity-diagnostic-v2.json`, compressed full derived summaries, scan receipts and five test logs. MANIFEST.sha256 binds the retrieved evidence files. Raw GPU sample arrays remain in the original campaign and are not included here.

This integration is the audit/decision-contract layer, not an activated live scheduler. The next gate is a separately frozen first-divergence experiment with native A/A controls, actual batch signatures, source/driver binding and frozen operator inputs. Diagnosis must not be counted as fresh performance confirmation.
