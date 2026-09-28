# Full-model continuation — execution checkpoint

**Snapshot: 28 September 2026, 06:52:17 UTC / 10:52:17 Dubai.**

The full-model pipeline has been implemented, tested on CPU, deployed and submitted. The new GPU jobs have **not started** at this snapshot. This is a recovery and execution checkpoint, not a new throughput result.

## Actual recovery

The previously refused read was accepted through the same Remote Desktop Commander terminal action under the renewed user request. No tool/channel switch or approval setting change was used.

The original engine job `1632883` had finished with `FAILED`, exit `1:0`, after 225 allocated GPU seconds. The first auto arm failed in FlashInfer sampling JIT because `curand.h` was missing from the compiler search path. This is now an observed failure, not speculation from filenames.

Its preceding paged qualification passed 576 non-auto layer/transition comparisons and 128 selected FP32-vector comparisons. Maximum absolute differences were 0.000244140625 versus auto and 0.000056371092796325684 versus FP32. These are tolerance-based kernel checks, not complete-model output equivalence.

## Implemented common fix and measurement hardening

The required cuRAND headers were already present in the existing NVIDIA CUDA13 package and the existing vLLM runtime archive. The isolated launcher now supplies their include/library paths explicitly. It does not disable FlashInfer sampling, change precision, download new weights or edit a shared environment. An actual CPU-side nvcc header probe passed; the repaired sampling runtime still needs its GPU qualification.

The complete model remains cached Qwen3-4B-Instruct-2507, vLLM 0.29.0, FlashInfer 0.6.18, FP16, eager, single RTX4090. The three arms are auto, the unchanged training-selected fixed policy, and the frozen native-cycle policy. All use the same requested 512MiB scratch and 2048 KV blocks of 16 tokens.

The driver now buffers step traces and writes them after timing. Every model process performs an unscored full-workload warmup. Cumulative output-prefix identity, exact request coverage and exact output length are checked. Offered TTFT includes admission lag; it is separate from admitted TTFT and per-request average TPOT. No HTTP/network latency is claimed.

The two synthetic-token workloads are unchanged. They are not natural-language quality benchmarks. The fixed baseline retains the earlier training-proxy limitation; it is not claimed to be the globally optimal serving policy.

**26 CPU tests passed:** 18 measurement contracts and 8 statistical/completeness contracts. The three model weight shards and config were hashed. None of these CPU tests is counted as GPU performance evidence.

## Actual submitted jobs

| Job | Role | State at snapshot |
|---|---|---|
| `1634020` | Three-arm full-model qualification | PENDING: Priority |
| `1634042_[0-5%3]` | Six counterbalanced process-order repetitions | PENDING: Dependency |

The second job has the verified dependency `afterok:1634020(unfulfilled)`, a three-GPU maximum concurrency and automatic rejection of an impossible dependency. It is submitted, not launched early. The qualification script exits successfully only after three complete arms and exact per-request greedy-token parity. An unsuccessful qualification cannot release the formal repetitions.

No healthy or unrelated job was cancelled. No completed reuse/wave experiment was repeated. Current execution is blocked by scheduler allocation, not the old read refusal or missing GitHub write permission.

## Claims that are deliberately not made

New full-model throughput, TTFT, TPOT, tail latency, SLO goodput and output-parity results are all **unavailable**. No attention-stack ratio is substituted for them. The previous unamortized rejection and the scoped 4090/512MiB attention-stack result remain unchanged.

After actual completion, the analyzer requires complete triplets, source/workload identity and request evidence. The qualification screen receives no inferential confidence interval. The formal comparison uses six whole process-order triplets, not individual tokens as independent samples. Sparse tail quantiles and the two fixed workloads limit generalization.

## Source and evidence

Frozen measurement commit: `7b480e2905456aafce3ef99cb5a6110162270ef5`.

Analysis/test commit: `601976c4080cbc17f829375a40f90977638d74a1`.

Machine-readable recovery, model hashes and submitted-job state: [recovery_and_launch.json](recovery_and_launch.json).

Remote campaign:

```text
/ssd/scxi253/single-gpu-inference-plan-cost-20260928/campaigns/full-engine-v2
```

`audit/recovery.json` records the original failure and raw hashes. `audit/header_probe.json` records the compiler command and header hashes. `audit/model_identity.json` records all weight hashes. `audit/all_cpu_tests.txt` records 26 passing tests. `audit/pipeline_state.json` records the two actual Slurm job records; its snapshot SHA256 is `c772bf1c6d3523198f16e8c82a57e95101331e0f47b7a0b99cccfadf18664934`.

The immutable measurement tree is separate from the later `auditor/` tree. Do not overwrite it while jobs are queued or running.

## Resume without duplicate work

First inspect the existing jobs and outputs:

```bash
P=/ssd/scxi253/single-gpu-inference-plan-cost-20260928
C="$P/campaigns/full-engine-v2"
squeue -j 1634020,1634042
sacct -j 1634020,1634042 --format=JobID,State,ExitCode,ElapsedRaw -P
find "$C/runs" -name complete.json -o -name receipts.json -o -name token_parity.json
```

Only after the relevant complete matrix exists, reproduce its analysis in a new directory:

```bash
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312/bin/python
S="$C/auditor/research/causal_plan_cost/reuse_gate"
export PYTHONPATH="$S/full_engine_v2:$S"
"$E" -m unittest -v test_measure_engine test_analyze_full
"$E" "$S/full_engine_v2/analyze_full.py" \
  --root "$C" --stage screen --out "$C/audit/screen-results-NEW"
# Use --stage test only when all six dependent jobs have complete records.
```

If the screen fails, retain its receipt/log and the dependent cancellation. Diagnose the actual failure before creating another version. Do not rerun healthy or completed jobs merely to recover context, and do not bypass a future tool refusal.
