# Preserved HTTP startup failure and CLI-only repair

Jobs 1639757 and 1639758 passed all pristine/off/cap paged graph, shared-page and independent-stream lifecycle checks, then the PRISTINE server exited during argument parsing. No model request, throughput, TTFT or TPOT sample was produced. The installed SGLang 0.5.20 has separate cuda_graph_max_bs_decode and cuda_graph_max_bs_prefill options, and the legacy --cuda-graph-max-bs is ambiguous. Its declarations in srt/arg_groups/fields/exec_.py were inspected.

The repair uses --cuda-graph-max-bs-decode=16 and leaves prefill defaults unchanged. Candidate C++ source, model, workloads, numerical tolerances and primary policy are unchanged. The original serving-source tree and both failed jobs stay intact; the repaired runner is a separately hashed serving-source-v2 tree and new smoke, not a replacement performance replicate.
