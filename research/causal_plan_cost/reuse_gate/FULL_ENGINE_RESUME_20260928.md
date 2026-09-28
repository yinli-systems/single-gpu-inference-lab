# Full vLLM continuation — 28 September 2026, 06:40 UTC

The owner requested actual full vLLM execution. A new request through the same normal Remote Desktop Commander terminal action successfully read the previously refused evidence. No alternate read channel or approval change was used.

## Recovered facts

At 06:38:53 UTC the user's ParaCloud queue was empty. Job 1632883 finished FAILED, exit 1:0, 225 allocated GPU seconds. The auto engine process failed during the FlashInfer sampling-module JIT build because `curand.h` was not on the compiler include path. The package header is already present in the existing runtime archive under `nvidia/cu13/include`. This is an environment-discovery failure, not evidence against the attention policy.

Paged qualification did pass: 576 non-auto full-layer/transition comparisons, 128 selected FP32 vectors, max absolute differences 0.000244140625 versus auto and 0.000056371092796325684 versus FP32. NHD/HND and changed physical page identities were checked. Summary SHA256: `90e34cd509e428cbc596f34e0919bf88e5e5ee0e6a3f366b9520577e4eb93b91`.

The prior failed job, source and logs remain immutable. Do not rerun the already completed reuse or wave matrices. The original unamortized rejection, 128 MiB rejection and RTX 5090 primary rejection all remain in force.

## Minimal repair and unchanged experiment

Create an isolated `reuse-engine-resume-v1` campaign. Restore the same runtime archive and append that runtime's already installed NVIDIA include/library directories to compiler discovery. Do not switch samplers, upgrade vLLM/FlashInfer, change model weights, train selectors or modify shared environments. Run a sampling JIT/argmax sentinel before loading the model.

Keep engine_screen.py, paged_router.py and contracts.py byte-identical to c052416 for the first dependency-repaired screen. Reuse the hash-checked successful paged qualification rather than spending another GPU job on identical checks. Actual outputs and path use must still be checked in the complete model.

First run a single finite three-arm screen. If interfaces, complete requests and exact greedy-output comparison pass, obtain the remaining two independent process/job repetitions. For rep0/1/2 use three balanced Latin orders: auto/fixed/native, fixed/native/auto, native/auto/fixed. All positions appear once per arm. Report every failure, not only completed winners. Three repetitions and two workloads support only conditional small-suite evidence; token or request counts are not independent hardware replicates. No early-stop based on an attractive performance point estimate.

Keep the already published full-engine protocol's settings: cached Qwen3-4B-Instruct-2507, FP16 weights/KV, TP1, explicit FA2, eager, 512 MiB workspace, 2048 KV blocks of16 tokens, chunk budget1024, max32sequences. Numeric prompts, arrivals, sampling, seeds, two workloads and SLO thresholds remain unchanged. These are in-process engine metrics, not HTTP response times. A success requires all20requests per arm and equal greedy output tokens. A failure aborts promotion, but all arm results are retained.

For the small screen, compare integrated throughput, offered/admitted TTFT, TPOT, p95/p99 and SLO goodput. Admission slippage and unfinished requests must not be hidden. Do not label a patched-vs-patched auto comparison a stock production-serving win. If the learned plan is never selected or qualified plans are absent, report the interface failure and stop.

Metrics and analysis will be frozen before evaluating timing differences. Any subsequent functional repair will receive a new version and retained failure record; the original experiment will not be overwritten. A performance-positive result requires unchanged complete outputs, a paired uncertainty interval favorable to the candidate, and no degraded SLO goodput under matched capacity. Otherwise the full-system candidate is not promoted.
