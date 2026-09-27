# Prior-art boundary for the resumed causal-plan experiment

Primary sources checked on 2026-09-28. This document is AI-assisted and does not assert priority or award eligibility.

| Source | Already established | What this study must add, rather than rename |
|---|---|---|
| FlashInfer, MLSys 2025, https://arxiv.org/html/2501.01005v2 | Section 3.3 explicitly uses query/KV lengths, tiled work, split reductions, cost-guided load balancing and a plan/run API. Planning is amortizable across layers. | A checked representation and falsifiable intervention result for the installed FA2 implementation, not invention of geometry-aware scheduling or plan/run separation. |
| FlashAttention-3, https://arxiv.org/html/2407.08608v2 | Hardware-aware attention pipelines; experimental protocol explicitly accounts for wave quantization. | Exact equal-work controls and measured policy consequences in this different runtime, not the general statement that FLOPs differ from time. |
| FlashAttention-4, https://arxiv.org/html/2603.05451v1 | Section 3.3 uses classical LPT scheduling for causal/variable-length work, balanced against L2 locality, including Hopper validation. | LPT, head swizzling and causal load balancing are NOT novel contributions of this project. |
| LeanAttention, https://arxiv.org/html/2405.10480 | Decode-oriented hardware-aware work partitioning and scalable split/merge. | This study concerns mixed-length causal prefill planning; a new setting alone does not establish a novel method. |
| PersistentKV, https://arxiv.org/html/2606.26666v2 | Native-paged GQA decode, compact nonempty workqueues and calibrated route selection; includes actual planning/launch overhead. | Cannot claim that calibrated routing, nonempty task queues, or accounting for planning overhead are new. |
| Tessera, submitted 2026-09-22, https://arxiv.org/html/2609.25869 | Dynamic block-sparse attention for video diffusion: logical/physical mapping separation, split-K/task organization, offline regime tables and low-overhead runtime plan selection. | A broad claim of first physical-plan or geometry-aware attention selection is ruled out. Dense causal prefill has a different workload, but our contribution still requires a specific validated mechanism or prediction/decision gain. |
| FlashInfer official API, https://docs.flashinfer.ai/api/attention.html | Public fixed_split_size and disable_split_kv; fixed size has page units (ragged page size one), determinism and CUDA-graph qualifications. | These are interventions supplied by the library, not our new algorithm. Current web docs are 0.7.0; experiment pins installed 0.6.18 and checks its actual planner ABI. |

## Current hypothesis, not a conclusion
At identical numerical attention semantics, a representation of the actual scheduled CTA work and causal iteration distribution may predict planner-option cost better than equally flexible marginal/joint feature models. It is useful only if choices beat a head-conditioned fixed policy selected on training data, after selection and planning costs. Frozen protocol specifies the gates. Failure of these gates is a result, not grounds to change the test split.

## Observation that already rejects an overbroad mechanism
The corrected 4090 canary still exhibits the discovery equal-work cliff with split disabled. Therefore batch-global automatic split choice is not a sufficient explanation. Small fixed chunks mitigate the contrast; this demonstrates sensitivity to the execution plan, but does not isolate occupancy, wave tails, data reuse and masking into a complete physical causal chain. Discovery timings never enter model fitting or held-out evaluation.

## Claims explicitly not made
No new attention mathematics, no invention of split-KV/LPT/physical plans, no universal model from work count alone, no result against full native implementations of all cited systems, and no operator-to-serving speedup extrapolation. Earlier normalized-marginal baseline ties and negative real-trace goodput results remain in force.
