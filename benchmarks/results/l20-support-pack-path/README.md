# Sampling-support output path: bitmap vs compact (L20 microbenchmark)

Measures the whole per-step path a vLLM model runner pays to return the
sampling mask (`--return-sampling-mask`): the GPU pack kernel, the
device-to-host copy, and the host-side construction of the CSR lists the
scheduler consumes. Both paths are fed the same top-k-masked logits and must
produce identical CSR output; the benchmark asserts that on every shape.

| Path | GPU | D2H | Host |
| --- | --- | --- | --- |
| **upstream** (vLLM v0.29.0 `SamplingMaskTensors`) | `_pack_sampling_mask_kernel` → `[B, ceil(V/8)]` uint8 bitmap + counts | bitmap + counts | `np.unpackbits` over the full vocabulary → `np.nonzero` |
| **compact** (`src/l20_stack/ops/triton_support_pack.py`) | `_support_pack_kernel` → `[B, K]` int32 ids + counts (+ logz, sampled logprob) | ids + counts | ragged slice |

The upstream kernel and `tolists` logic are copied verbatim into the benchmark
script so the comparison runs in one process; the copy's source is pinned by
`provenance.upstream_output_py_sha256` (`vllm/v1/worker/gpu/sample/output.py`
at tag `v0.29.0`).

## Environment

- GPU `NVIDIA L20`, host CPU 8 cores; Python 3.12.3, PyTorch `2.13.0+cu130`, Triton `3.7.1`
- Repository commit `6d0bbc2` (clean); kernel source sha256 `e7fbe357…8779d766`
- Inputs: vocab 151936, `top_k = 50` (finite only inside each row's top-k), `K = max_support = 64`
- 5 trials × 50 rounds per path, alternating path order per trial; GPU time from CUDA events,
  host time wall-clock after a stream sync (pinned-buffer copy + CSR build)

## Result (median of trial medians, ms per step)

| Batch | upstream GPU | upstream host | upstream D2H | compact GPU | compact host | compact D2H | total speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.030 | 4.484 | 152 KB | 0.075 | 0.042 | 2.1 KB | **39x** |
| 32 | 0.030 | 17.748 | 608 KB | 0.075 | 0.044 | 8.3 KB | **149x** |
| 128 | 0.036 | 70.830 | 2,431 KB | 0.146 | 0.061 | 33.3 KB | **342x** |
| 256 | 0.226 | 145.103 | 4,863 KB | 0.236 | 0.083 | 66.6 KB | **457x** |

CSR `token_ids`/`offsets` identical on every shape; compact `logz` and sampled
logprob match `torch.logsumexp` within 1e-4.

## Interpretation

- The upstream cost is almost entirely **host** work that scales with `B × V`
  (unpacking ~39 M bits at batch 256), not the GPU kernel. It sits on the
  model runner's output path every decode step, so at batch 256 it caps the
  step rate at roughly 7 steps/s regardless of model size. The serving A/B in
  [`../l20-vllm-sampling-mask-ab/`](../l20-vllm-sampling-mask-ab/README.md)
  shows exactly that floor (~1.6K tok/s for both a 0.5B and a 4B model).
- The compact kernel is slower than the bitmap kernel at small batch on the
  GPU side (0.075 vs 0.030 ms) because it also computes the log-normalizer
  and does a prefix sum; that is 30–150x smaller than the host cost it removes
  and is not the bottleneck at any measured batch.
- This is an operator/path microbenchmark on one host CPU; the absolute host
  numbers depend on the CPU and NumPy build. The ratio is what generalises:
  the bitmap path is `O(V)` per row on the host, the compact path `O(K)`.

## Files

- `raw.json` — per-batch trial medians, protocol, environment, provenance.
- Script: [`scripts/benchmark_support_pack.py`](../../../scripts/benchmark_support_pack.py).
- Kernel tests: [`tests/test_support_pack_gpu.py`](../../../tests/test_support_pack_gpu.py).
