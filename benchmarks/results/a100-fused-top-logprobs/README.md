# A100 Fused Top-Logprobs Microbenchmark

This artifact measures a two-stage Triton primitive that selects normalized
top-N logprobs without materializing a full `[batch, vocab]` log-softmax tensor.
It is a controlled operator microbenchmark, not a host-latency or serving-speed
claim.

## Current Result

Environment:

- GPU: `NVIDIA A100-SXM4-80GB` (`SM80`, 80 GB)
- Driver: `570.195.03`
- Python: `3.12.3`
- PyTorch: `2.8.0+cu128`
- CUDA runtime: `12.8`
- Triton: `3.4.0`
- Kernel commit: `5f43f31d9a21d4a69106acbff131665b928560db`
- Shape: vocab `151936`, top-N `5`, temperature `0.8`, FP16

Each row aggregates three independent processes using distinct seeds
(`113`, `114`, and `115`). Every process contains five paired trials with 30
measured rounds per provider, so each speedup below is the median of 15
within-trial ratios.

| Batch | Triton median | `log_softmax` + `topk` | Paired speedup | `logsumexp` + `topk` | Paired speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.021392 ms | 0.186304 ms | 8.781x | 0.177504 ms | 8.391x |
| 4 | 0.023872 ms | 0.224448 ms | 9.445x | 0.221440 ms | 9.290x |

Across the three process medians, Triton ranged from `0.021216–0.021504 ms` at
batch one and `0.023824–0.023968 ms` at batch four. All 30 paired trial
comparisons beat both PyTorch baselines. The full paired speedup range is
`8.27x–9.69x`; the two reported aggregate medians span `8.39x–9.45x`.

Correctness passed for three distinct seeded inputs per shape across all six
timing runs:

| Batch | Tie-aware top-N match | Strict-above-cutoff tokens present | Max absolute logprob error |
| ---: | --- | --- | ---: |
| 1 | yes | yes | `4.768e-7` |
| 4 | yes | yes | `4.768e-7` |

The checker requires unique tokens, non-increasing source logits, every token
strictly above the exact input-dtype cutoff, and reported values that match
full-vocabulary normalization. Only exact cutoff ties may reorder.

Machine-readable aggregate:
[summary.json](summary.json). Raw schema-v2 results:
[controlled-v2/](controlled-v2/).

## Timing Protocol

A100 DVFS materially changes the latency of this 20-microsecond primitive. To
make the active-GPU regime explicit, the benchmark applies the same preallocated
`8192 x 8192` FP16 GEMM immediately before every provider sample:

```text
same CUDA stream:
    identical GEMM clock precondition
    -> start event
    -> measured provider
    -> end event
```

The GEMM is outside the CUDA event interval. All six provider permutations are
cycled for within-round position balance, events are allocated before
measurement, and the stream synchronizes once after all recorded events in each
trial. Speedups are paired within a trial before aggregation.

The comparator is intentional and narrow:

- Triton writes to caller-owned output and workspace tensors.
- The PyTorch baselines include their GPU kernels and full-vocabulary FP32
  temporary materialization.
- CUDA events exclude Python launch and host allocator latency.

This protocol measures steady-state operator latency after GPU compute. It does
not measure an isolated idle call, total host latency, or vLLM serving latency.

## Reproduce

Run each batch in three clean processes:

```bash
for batch in 1 4; do
  for replica in 1 2 3; do
    PYTHONPATH=src python scripts/benchmark_l20_top_logprobs.py \
      --batch "$batch" \
      --vocab 151936 \
      --top-n 5 \
      --temperature 0.8 \
      --warmup 5 \
      --rounds 30 \
      --trials 5 \
      --seed "$((112 + replica))" \
      --clock-policy steady-state-gemm \
      --precondition-size 8192 \
      --precondition-repeats 1 \
      --output "/tmp/top-logprobs-b${batch}-r${replica}.json"
  done
done
```

Each JSON records the git SHA and dirty state, benchmark and kernel source
hashes, command, UTC timestamp, hashed GPU identity and compute capability,
driver, PyTorch/CUDA/Triton/Python versions, protocol, host-side `nvidia-smi`
snapshots, raw samples, paired trial speedups, and correctness details.
Host-specific executable/output paths and the GPU UUID are normalized in the
published copies; each JSON marks that normalization explicitly.

## Historical Artifact and Claim Boundary

The original [b1.json](b1.json) and [b4.json](b4.json) are retained byte-for-byte
for provenance. Their `8.04x–9.17x` headline was sensitive to an unrecorded A100
clock regime, and the published command did not reproduce it reliably from an
idle GPU. Those files are therefore not used for the current headline.

The opt-in Triton hook has separately reached real vLLM `logprobs` requests with
80/80 clean trace hits, but total request time was flat. The current claim is
only:

> `8.39x–9.45x` versus the two composed PyTorch baselines in this repeated,
> steady-state GEMM-conditioned A100 operator protocol.

Do not infer idle-call latency, host latency, vLLM serving speedup, L20
performance, or results on another software stack from this artifact.
