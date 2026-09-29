# Parity remediation and request-lifetime repair

## Decision

**One scoped HTTP reliability defect reproduced and fixed; the resource-cap inference
optimization remains default OFF and serving HOLD.** No new performance repetitions
were submitted in this campaign. The original two token mismatches and worst-case
resource-policy regressions are not erased or declared solved.

The tested full model is Qwen3-4B-Instruct-2507 through SGLang0.5.20 / FlashInfer0.7.0
on allocated RTX4090 GPUs. The workload is synthetic token input through the real
HTTP model runtime, not a recorded production traffic distribution. Current results
cannot be generalized to all models, GPUs, prompts or concurrent workers.

## 1. A real delayed-cleanup defect and source-level repair

The original `TokenizerManager.create_abort_task` waits two seconds and aborts the
current request bearing an old RID. After a completed stream, a replacement with the
same RID can be registered before the old cleanup fires. The fixed method captures
original request-object and ReqState identities and rechecks ownership after the delay.
A new object or state generation is never canceled by the old cleanup. Original live
requests remain cancelable. No shared SGLang installation was changed.

Job1639861 ran the original and fixed methods in separate owned Qwen HTTP servers,
with an explicitly frozen repeated-RID matrix and unique-RID controls:

|Path|All requests complete|Second reused requests|Unique controls|Observed stale-owner aborts|
|---|---:|---|---:|---:|
|Original|6/8|Aborted at66 and65 of128 tokens|4/4|2|
|Ownership-checked|8/8|Both reached128 tokens normally|4/4|0|

The trace records both the old cleanup-owner object identity and the new registered
target identity at the actual abort call. This is stronger than inferring the problem
from an incomplete response. Twelve CPU tests reproduce the old race and exercise
replacement-before/during-delay, same-object/new-state, live, finished, removed and
batched cases. Remaining production topics include multi-worker and batched n>1
coverage, other cancellation paths, and retaining ReqState buffers during the grace
period. The scoped repair is not yet an upstream-accepted production change.

Related prior work: [SGLang PR38850](https://github.com/sgl-project/sglang/pull/38850)
already describes the RID-reuse problem in scripted-runtime tests. This work does not
claim discovery priority. The inspected production method at upstream commit
`211b1d9784d15845566081cc25c8103ce43cc4f2` matches the installed original. See
`UPSTREAM_REVIEW.md` and `abort_ownership.py` for the patch and review boundaries.

**This truncation defect is NOT the same as the old full-length token-value divergences.**

## 2. Original token parity: denominator and evidence corrected

The previous formal HTTP archive contains432 requests per arm,1296 across all three
arms. Cap differs from pristine on **2/432 cap requests**, at output indices74 and114
(zero-based), both in job1639805/decode-b1. Pristine and disabled-arm outputs are
internally stable across the nine old batches per workload. All original JSON receipt
hashes are checked; all three blocks, including block0, are retained.

The old untyped log values215,750,etc. are tokens/second, not milliseconds. Prior
interpretations treating these as latency were wrong. New validation recomputes
throughput from actual output token counts and elapsed seconds. See
`REPORT_CORRECTIONS.md` for the complete corrections.

## 3. New full-model correctness diagnostics, kept separate by context

|Diagnostic|Paired cap requests|Cross-arm differing requests|Interpretation|
|---|---:|---:|---|
|Original execution configuration,2 process repeats|288|0|Old failure did not reproduce; not resolved|
|Both arms fixed-seed and upstream deterministic mode,lossless wire capture|144|0|Changed configuration; no old speedup transferred|
|Fixed-seed,overlap-disabled sampler observer|144|4|Pristine trajectories also vary; all failures retained|
|Fixed-seed,overlap-disabled layer/KV observer|144|10|Instrumentation changes execution; not a performance trial|

The deterministic mode is an existing upstream feature, not a new algorithm here.
Two earlier diagnostic jobs failed on repeated-ID isolated streams before a cap trace
was obtained. Those failures are retained, and the later transport-correct revisions
are separate experiments, not replacements. Isolated probes use distinct wire IDs;
that workaround alone is not a claimed production fix for RID reuse.

A separate context diagnostic shows pristine decode-11 can choose a different first
token when executed alone rather than batched. At the historical decode-11 common
prefix, a teacher-forced prefill probe reports tied top logprobs for tokens304/315.
Decode-13 does not show the same tie. These probes use a different execution path
and do not establish why the old autoregressive mismatches occurred.

## 4. Sampler and layer/KV evidence

The first completed sampler-observer comparison recorded2210 pristine and2211 cap
steps. Of708 common complete logical batch signatures,677 have identical stable
full-logit hashes;31 show within-context variability. Set comparisons retain candidate-
only variants even when the baseline varies: none appeared in this sampler-only run.
There are392/320 unmatched signatures. Matching histories alone is not matching KV.

The subsequent layer observer is a separate context:2225/2214 sampler steps,712 common
signatures,650 stable identical logit signatures,62 variable signatures, and31 signatures
with cap-only variants. There is no disjoint variant set, but this does NOT exonerate cap.
Its10 cross-arm trajectory differences remain recorded. The observer adds GPU copies
and host synchronization and can change batching, resource behavior and numerical paths.

The layer job captured13 complete snapshots:8 pristine and5 cap, covering36 decoder
layers and the actual request KV prefix (72 K/V tensors per snapshot), physical cache
indices, selected norm/projection/attention/MLP inputs/outputs and logits. Every saved
raw tensor is checked against its metadata hash before analysis.

Only **one** complete execution coordinate matches across arms, at isolated decode-13
before output index114. Its two-by-two repeated snapshots make four dependent pair
comparisons: all718 observed shadow tensors,72 K/V tensors and logits are equal.
This is one matched condition, not four independent workloads and not validation of
both old failures. Cap did not reach the old decode-11 common prefix in this layer run.

Cross-context diagnostics explicitly relax the full-batch match while retaining the
target's complete logical history. For three8-row decode-13 comparisons, current-layer0
Q/K/V inputs match, but the historical layer0 KV prefix already differs. The first
observed current-step shadow difference is layer0 attention output (maximum absolute
row differences0.000244140625–0.00048828125). This does not prove an attention defect:
its historical KV inputs were different. Across8-row versus1-row execution, an earlier
QKV-projection shadow difference appears. These observations narrow the investigation
to the preceding KV/activation generation context; they do not identify the first cause
in the old uninstrumented run or prove a cache race.

## 5. What remains unresolved

- The original cap-only full-length token divergences are not causally attributed.
  Neither a later zero-mismatch replay nor baseline variability clears this gate.
- Existing resource-cap/wide worst-case regressions and unresolved measurement controls
  remain. No new guard or performance result is presented as solving them.
- No complete fresh-family validation, performance test of the deterministic configuration,
  or broad production lifetime qualification was added here.
- No upstream performance PR was submitted and no default was enabled.

The deployed research default remains identity/off. `evidence_contract.release_gate`
requires parity, matching measurement resolution, independently qualified supported
paths, a simultaneous regression bound, positive net benefit and old-failure attribution.
It refuses unknown states rather than silently accepting them. This guard is an
experimental evidence gate, not a theorem about kernel latency.

## Reproduce and inspect

```bash
python -m unittest discover -s research/parity_remediation -p 'test_*.py' -v
python research/parity_remediation/verify_archive.py research/parity_remediation/evidence/raw-evidence.tar.gz
```

All56 CPU tests passed in the recorded remote Python/Torch environment. On the Mac
without Torch,51 passed and five Torch-dependent tests were explicitly skipped; these
five were executed successfully remotely. CPU tests are not GPU performance validation.

The evidence manifest distinguishes archived files from large activation/KV/logit tensor binaries retained
on ParaCloud with exact SHA256 and paths. It includes failed attempts, raw SSE responses,
source snapshots, job accounting, trace metadata, original historical JSON inputs and
all final audits. Diagnostic output timing is never merged into a serving performance table.

## Delivery receipt

Seven bounded one-GPU jobs are terminal: five completed and two failed diagnostic
attempts. All failures are retained. Total allocation was3,761 GPU-seconds
(1.044722 GPU-hours), including the failures; this is not a billing-price estimate.

The public packet contains1,426 source/log/JSON evidence files,13,717,829 compressed
bytes, SHA256 `05506de8b81654281ce632e2de5669ede665f0d7e748601f3b2e35d2970f5499`.
All1426 members were hash-verified on ParaCloud and again on the Mac. Fifty-three
large tensor binary files remain at their recorded ParaCloud paths with SHA256s.
The original62,021,886-byte pre-thinning archive is also retained with SHA256
`cb64d0c94e320540a033c2ef36f3777469336714655fe1fc7e4b3b1cc9a27970`.
No failed request, trace, timing observation or slow case was filtered from the JSON
packet. See `evidence/DELIVERY.json` and `evidence/MANIFEST.json` for exact accounting.

The standalone reviewed cleanup patch and source binding are under `upstream/`.
The review is in this research repository; it is not official SGLang acceptance.
