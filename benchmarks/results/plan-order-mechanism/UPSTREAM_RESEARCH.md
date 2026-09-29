# Upstream triage, checked2026-09-29

- vLLM PR52061 (open, head c8061efd61c1651f159f44bb84bb16468a75d890): native opt-in forward-pass metrics, timing scope model_step_cuda, no production-zero-overhead claim. Our complementary deliverable is exact shape/plan counterexamples and conformance fixtures, not a duplicate collector. https://github.com/vllm-project/vllm/pull/52061
- vLLM RFC55259 is an ISSUE, not a merged benchmark interface. Ordered-row manifests are proposed; do not assert support on the installed vLLM0.29 runtime. https://github.com/vllm-project/vllm/issues/55259
- FlashInfer PR5687 (open): persistent queue scheduling for long uniform SM90 VSA. This is prior art and a different path, not our work. Dense FA2 metadata-order isolation on SM89/SM120 is the current experimental scope. https://github.com/flashinfer-ai/flashinfer/pull/5687
- FlashInfer PR5176 is already MERGED. It fixes padded CUDA-graph block masks. Do not re-submit or claim to solve it. https://github.com/flashinfer-ai/flashinfer/pull/5176
- FlashInfer issue4269 concerns NVFP4-vs-FP8 paged prefill. FP16/BF16 ragged tests do not address that issue. https://github.com/flashinfer-ai/flashinfer/issues/4269
- Autotuner v2 is already released; applicability to this exact wrapper must be established from source, not inferred from the library version. https://flashinfer.ai/2026/09/22/autotuner-v2.html

No upstream acceptance, unreported benchmark or universal novelty is claimed. The next submission depends on qualifying the new code and measuring actual net value.
