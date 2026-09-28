# Single-GPU Inference Lab

**LLM inference systems research, from request scheduling to GPU execution.**

[![CI](https://github.com/yinli-systems/single-gpu-inference-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/yinli-systems/single-gpu-inference-lab/actions/workflows/ci.yml)

[Technical report](docs/when-token-budgets-lie.md) · [Results](benchmarks/results/README.md) · [Research record](RESEARCH.md) · [Reviewer guide](docs/reviewer-guide.md)

I investigate where inference time is spent, build the smallest change that addresses
the measured bottleneck, and test it against the unmodified runtime. The evidence
covers NVIDIA L20 and A100 systems, with a separate 2× RTX 4090 DP/EP track.

## Selected results

| Work | Finding | Evidence and boundary |
|---|---|---|
| **Prefill cost geometry** | Swapping request-level chunk/KV pairings while preserving aggregate state changes measured step time by **7–55 ms**. Aggregate token counts alone cannot identify that cost. | [Report](docs/when-token-budgets-lie.md) · [pairing-swap experiments](benchmarks/results/prefill-pairing-swap/README.md). Dense Qwen models on the recorded hardware; not a universal latency law. |
| **Sampled-token logprob transport** | On the measured L20/Qwen2.5-0.5B workloads, a flat-output path restores throughput from **0.67× to 0.95–0.96×** of the no-logprobs baseline. | [Serving A/B and exactness checks](benchmarks/results/l20-flat-token-logprobs/README.md) · [vLLM #57442](https://github.com/vllm-project/vllm/pull/57442), **open**. These are workload-specific serving measurements. |
| **When better prediction does not help** | The geometry-aware cost model improves step pricing, but the tested controller does **not** improve real-trace TTFT/TPOT SLO goodput. | [Live trace replay](benchmarks/results/live-trace-replay/README.md). The negative queue-level result is retained alongside the positive cost-model result. |

The implementation, raw measurements, commands, environment records, and failed
hypotheses are linked from each result. The [expanded research record](RESEARCH.md)
contains the remaining kernel, beam-search, KV-prefetch, and multi-GPU studies.

## Upstream work

| Change | Status at the 2026-09-28 review | Entry point |
|---|---|---|
| Sampled-token logprob fast path | Open PR; measured local serving result, not a merged feature | [vLLM #57442](https://github.com/vllm-project/vllm/pull/57442) |
| Integer token IDs for generate logprobs | Open PR; protocol/correctness change, not a speed claim | [vLLM #58181](https://github.com/vllm-project/vllm/pull/58181) |
| Request-free CPU→GPU KV prefetch | Open PR; reserve-gated execution primitive, not a universal admission policy | [vLLM #58198](https://github.com/vllm-project/vllm/pull/58198) |

[All local upstream artifacts](upstream/README.md) · [Current experiment status](docs/experiment-status.md)

## Reproduce and validate

CPU-only validation checks the public artifact index, documentation paths, catalog,
and tests. It does **not** regenerate GPU measurements.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]" numpy
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q
single-gpu-infer artifact-index --strict-warnings
single-gpu-infer doc-links
single-gpu-infer doc-links --file RESEARCH.md
single-gpu-infer artifact-catalog --output /tmp/artifact-catalog.json
cmp benchmarks/results/artifact-catalog.json /tmp/artifact-catalog.json
```

The CPU PyTorch index above is the Linux CI path; macOS users should install the
platform's supported CPU PyTorch wheel instead. GPU reproduction requires the exact
hardware, model, runtime revision, and commands recorded in the selected artifact.
There is no single environment that reproduces every historical campaign.

## Evidence standard

Measured speedups keep their hardware, model, workload, and baseline attached.
Kernel timing, diagnostic traces, full-engine serving, and oracle bounds are
separate evidence categories. Negative results and correctness withdrawals remain
public; stronger presentation does not change the underlying claims.

The prefill study is **not a new scheduler**. KV-prefetch gains under slack do not
establish gains under sustained occupancy. Local patches and open PRs are not
presented as upstream releases. See the [claim policy](RESEARCH.md#claim-policy)
and [hardware scope](docs/hardware-scope.md).

## Repository guide

| Path | Contents |
|---|---|
| `benchmarks/results/` | Raw results, figures, exact commands, and artifact catalog |
| `src/l20_stack/`, `cuda/`, `cpp/` | Kernels, operators, and control implementations |
| `integrations/`, `upstream/` | Runtime integrations and proposed upstream changes |
| `scripts/`, `configs/` | Measurement harnesses, analyzers, and experiment configurations |
| `tests/` | Correctness, contract, artifact, and repository-hygiene checks |
| `docs/`, `RESEARCH.md` | Technical reports, methodology, limitations, and full research record |

The Python namespace remains `l20_stack` for compatibility. Code is available
under the [MIT license](LICENSE).
