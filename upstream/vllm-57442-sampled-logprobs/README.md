# vLLM #57442 — sampled-token logprob fast path

- `0001-sampled-logprob-fast-path.patch` — the PR as it stands on the fork (`flat-token-logprobs`,
  rebased on main 2026-09-22 and ruff-clean; DCO signed).
- `0002-restacked-on-58181-sampled-field.patch` — the same change on top of #58181, carrying the
  floats as `GenerateLogProbs.sampled` (with `content=None` in sampled-only mode) instead of a
  separate `choices[].token_logprobs`, as agreed on #57574. To be pushed once #58181 merges.

Measurements and the layer-by-layer cost decomposition live with research line A:
[`benchmarks/results/l20-flat-token-logprobs/`](../../benchmarks/results/l20-flat-token-logprobs/README.md),
[`benchmarks/results/l20-logprob-engine-decomposition/`](../../benchmarks/results/l20-logprob-engine-decomposition/README.md).
