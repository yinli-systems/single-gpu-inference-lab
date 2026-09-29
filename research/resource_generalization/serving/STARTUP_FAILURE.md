# Preserved HTTP startup failure and CLI-only repair

Jobs 1639757 and 1639758 passed all pristine/off/cap paged graph, shared-page and independent-stream lifecycle checks, then the PRISTINE server exited during argument parsing. No model request, throughput, TTFT or TPOT sample was produced. The installed SGLang 0.5.20 has separate cuda_graph_max_bs_decode and cuda_graph_max_bs_prefill options, and the legacy --cuda-graph-max-bs is ambiguous. Its declarations in srt/arg_groups/fields/exec_.py were inspected.

The repair uses --cuda-graph-max-bs-decode=16 and leaves prefill defaults unchanged. Candidate C++ source, model, workloads, numerical tolerances and primary policy are unchanged. The original serving-source tree and both failed jobs stay intact; the repaired runner is a separately hashed serving-source-v2 tree and new smoke, not a replacement performance replicate.

## Separate CCCL include failure in repaired CLI smoke

Job 1639770 completed all three lifecycle variants, then the pristine server's fused-RoPE JIT compilation failed with `fatal error: nv/target: No such file or directory`. The child subsequently terminated its server process; the observed -9 exit is not evidence of model OOM or a candidate numerical fault. No HTTP measurement completed.

The existing pinned 0.7.0 package supplies libcudacxx/cub/thrust. A CPU nvcc C++20 compile of CUDA BF16, cuda/std and CUB headers succeeded with explicit include paths, recorded in receipts/sglang-cccl-preflight.json. The v3 launcher sets CPATH and NVCC_PREPEND_FLAGS to these existing directories and directs SGLang JIT/runtime caches to a campaign-private directory. Shared libraries, headers, credentials and old caches are not edited or removed. Source-v1/v2 and all failed jobs remain intact. Candidate/source/workload/numerical policies are unchanged.
