# L20 Fused Top-Logprobs Microbenchmark (post-fix, 2026-09)

This artifact re-measures the two-stage Triton top-logprobs primitive on the
project's primary NVIDIA L20 target **after** the all-masked-tile NaN fix and
tensor-contract hardening landed in commit `8a987d2`. The
[A100 artifact](../a100-fused-top-logprobs/README.md) was measured on the
pre-fix source and has not been remeasured; its `summary.json` carries a dated
`kernel_source_supersessions` entry pointing here.

It is a controlled operator microbenchmark, not a host-latency or serving-speed
claim. The A100 and L20 numbers are not directly comparable to each other; each
is comparable only within its own protocol.

## Why this exists

A vocabulary tile whose every logit is `-inf` (an `allowed_token_ids` or grammar
mask that empties a whole 1024-token tile) made the partial kernel compute
`exp(-inf - (-inf)) = NaN`, which poisoned the row's global log-normalizer.
On this GPU, at default tile sizes, eight live tokens per row produced 8/8 NaN
rows for vocab 32000, 128256 and 151936. The fix shifts by a finite value when
a tile is empty so it contributes the log-sum-exp identity. The regression
suite is [`tests/test_top_logprobs_masked_tiles_gpu.py`](../../../tests/test_top_logprobs_masked_tiles_gpu.py).

The fix adds one `tl.where` per tile and one per row; this artifact checks
that the primitive's cost profile did not change in kind.

## Environment

- GPU: `NVIDIA L20` (`SM89`, 46 GB), driver `580.159.04`, power limit 350 W
- Clocks observed after measurement: SM 2430 MHz (max 2520), memory 9001 MHz, P0
- Python `3.12.3`, PyTorch `2.12.1+cu130`, CUDA runtime `13.0`, Triton `3.7.1`
- Kernel commit: `8a987d25` (clean working tree in every process)
- Shape: vocab `151936`, top-N `5`, temperature `0.8`, FP16 logits

Protocol: identical to the A100 controlled run. Each row aggregates three
independent processes with distinct seeds (`113`, `114`, `115`); each process
runs five paired trials of 30 measured rounds per provider, preconditioned by
an 8192x8192 FP16 GEMM outside the CUDA-event interval (`--clock-policy
steady-state-gemm`). Each speedup is the median of 15 within-trial paired
ratios.

```bash
python scripts/benchmark_l20_top_logprobs.py --batch {1,4} --seed {113,114,115} \
  --clock-policy steady-state-gemm --output raw/b{batch}-r{n}.json
```

## Result

| Batch | Triton median | `log_softmax` + `topk` | Paired speedup (median, min–max) | `logsumexp` + `topk` | Paired speedup (median, min–max) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.012288 ms | 0.100352 ms | 8.250x (8.12–8.91) | 0.099328 ms | 8.083x (7.96–8.73) |
| 4 | 0.015360 ms | 0.112640 ms | 7.333x (7.27–7.86) | 0.113664 ms | 7.400x (7.33–7.93) |

All 60 paired trial comparisons (2 shapes x 2 baselines x 15) beat the
PyTorch baseline. Reported aggregate medians span `7.33x–8.25x`.

Correctness in every process: tie-aware top-N match, strict-above-cutoff
tokens present, exact token order match against `torch.topk(log_softmax)`,
max absolute logprob error `4.77e-7`.

## Interpretation

- The primitive remains roughly an order of magnitude cheaper than composed
  PyTorch on the L20, consistent in character with the A100 result. The lower
  absolute L20 times reflect a faster CUDA/Triton stack and a different GPU,
  not a kernel change; do not read the two artifacts as a cross-GPU scaling
  claim.
- Nothing here changes the serving-level position in
  [`docs/experiment-status.md`](../../../docs/experiment-status.md): the opt-in
  vLLM hook is a path proof with flat total request time, and no end-to-end
  serving win is claimed.

## Files

- `summary.json` — schema 2 aggregate with provenance hashes for the benchmark
  script, kernel source, and each raw file.
- `raw/b{1,4}-r{1,2,3}.json` — one file per process, including nvidia-smi
  snapshots before setup and after measurement.
