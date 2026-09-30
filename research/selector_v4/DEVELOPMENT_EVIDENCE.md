# Exposed development evidence

v3.2.2 completed 96/96 dual-GPU runs with exact numerics but remained HOLD. On RTX 4090, selected cells had 1.2851x geomean and 1.1450x worst. On RTX 5090, selected geomean was 1.2037x, but `holdout-v32-opp-04` regressed to 0.9874x. Unselected/fallback tails also fell below 0.99.

The old candidate applied `cudaFuncSetAttribute(..., 65536)` to the same function symbol even when a later call used the native 49152-byte launch. v4 treats this as a plausible function-state contamination mechanism and isolates symbols; this is an engineering hypothesis until GPU canary evidence confirms it.

A 2.4-wave retrospective gate retains all old 4090 selected cells (1.2851x geomean, 1.1450x worst) and reduces the exposed 5090 selected set to 1.1815x geomean, 1.0366x worst. These numbers were used to freeze the candidate pool and must not be reported as fresh v4 results.

Primary design references:
- FlashInfer Autotuner v2: https://flashinfer.ai/2026/09/22/autotuner-v2.html
- NVIDIA CUDA occupancy APIs: https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__OCCUPANCY.html
- PyTorch CUDA Graph semantics: https://docs.pytorch.org/docs/main/notes/cuda.html
