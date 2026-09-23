# How often do real traces produce the mispriced prefill geometry?

**Question.** [`l20-prefill-cost-geometry/`](../l20-prefill-cost-geometry/README.md) and its A100
replication show that an aggregate step-cost coordinate misprices steps in which several
prefills at different KV depths share the token budget. They also show that on default-FCFS
short-prompt bursts this barely happens (L20 §5c). How common is that geometry under real
request traces, with the vLLM scheduler's own rules?

**Answer (simulation, one A100 running Qwen3-4B).** It depends on the traffic class, not on
the trace family:

- **Short-prompt chat** (BurstGPT, Azure 2023 conversation) at vLLM defaults: steps whose
  geometry error exceeds 5 ms are **0–7%** of prefill steps and **≤ 10%** of prefill time. The
  aggregate coordinate is essentially fine here, as L20 §5c found live.
- **Long prompts and prefix-cached context** (Azure 2023 code; Mooncake conversation, tool-agent,
  synthetic, which carry block hashes): **6–28%** of prefill steps (7–50% of prefill time) at the
  defaults and at budget 8192. With budget 8192 the P95 error on the Mooncake traces is **170–270 ms**, larger than a
  typical step deadline.
- **Capping the per-request chunk** (`long_prefill_token_threshold` 512, a common latency knob)
  raises the share on every trace: to 11–36% of prefill steps outside BurstGPT (BurstGPT:
  0.6% → 1.4%).

This is a **simulation**. The engine was not replayed, and the error is computed from the
measured slope rather than measured per step. It answers "how often", not "how much goodput".

## Method

[`scripts/simulate_geometry_prevalence.py`](../../../scripts/simulate_geometry_prevalence.py)
replays each trace through the vLLM 0.29 V1 scheduler's token-budget rules, read from
`vllm/v1/core/sched/scheduler.py` 0.29.0:

- running requests first, each taking min(remaining, per-request cap if set, budget left);
  decodes take 1 token;
- then waiting requests FCFS while budget remains, capped the same way, up to `max_num_seqs`
  (256, the A100 API-server default);
- admission only if the request's full KV fits (capacity 400k tokens ≈ Qwen3-4B on an 80 GB
  A100 at 0.9 utilization); no preemption.

Prefix caching applies only to the Mooncake traces: an LRU over their 512-token block hashes,
bounded by the KV capacity not held by running requests. Azure and BurstGPT carry no prefix
information, so every request there starts at depth 0. This **under**states geometry for any
deployment that shares system prompts.

**Clock.** Step times come from the M2n form fitted in-sample on every measured A100 Qwen3-4B
prefill step (from [`a100-prefill-cost-geometry/`](../a100-prefill-cost-geometry/README.md)),
plus a linear decode-only model. It is only used to advance simulated time. The measured range
is decode batch ≤ 33 and ≤ 8 prefills. Runs with ≥ 10% of steps outside it are excluded from the
summary below.

**Geometry error of a step.** An aggregate model that is exact on one-prefill steps charges
slope·(Σq)(Σkv) where the engine does slope·Σqᵢkvᵢ. So:

  geo_err = 3.3 ms/M × ((Σq)(Σkv) − Σ qᵢ·kvᵢ) / 10⁶

The 3.3 ms/M is the measured A100 4B slope. The error is zero for one-prefill steps and for
steps whose prefills all start at depth 0, and it is always an **over**-charge. For a
deadline-aware scheduler that means lost prefill throughput (L20 §5: the M0 controller
starved), not violations.

**Checked by hand** on toy traces before any real one was run: tail-plus-head step
(6.76 ms expected and found), prefix hit (776 tokens at depth 1024), equal-depth pair under a
cap (1.73 ms).

**Load.** Each trace is time-compressed or stretched (×0.05 … ×400). The table reports the
highest load with TTFT p50 < 1 s. Every run is in the JSON files.

**Timestamp quantization.** The Mooncake conversation and tool-agent traces record arrivals in
3 s slots: 1,180 distinct timestamps in 59 min, with up to 47 requests on one timestamp.
Replayed as-is, every slot is a simultaneous burst. The results then do not change with load
and TTFT p50 is ~2.5 s at any rate (kept in
[`prevalence-azure-burstgpt-mooncake-asrecorded.json`](prevalence-azure-burstgpt-mooncake-asrecorded.json)).
The summary uses the arrivals spread uniformly within each slot (fixed seed),
[`prevalence-mooncake-jitter3s.json`](prevalence-mooncake-jitter3s.json). The synthetic trace
has distinct timestamps and is used as-is.

## Summary

Highest simulated load with TTFT p50 < 1 s and < 10% of steps outside the measured clock
range ([`summary.md`](summary.md)):

| trace | config | load | TTFT p50 / p99 (s) | prefill steps with ≥2 prefills | geometry error > 5 ms: steps / share of prefill time | geometry error P95 / P99 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Azure 2023 conversation | default (2048, no cap) | ×1 | 0.06 / 0.6 | 14.6% | 6.8% / 9.5% | 8 / 26 |
| Azure 2023 conversation | budget 8192 | ×1 | 0.06 / 0.4 | 9.6% | 0.2% / 0.7% | 0 / 0 |
| Azure 2023 conversation | 2048, cap 512/request | ×1 | 0.11 / 1.5 | 28.5% | 11.3% / 22.8% | 13 / 34 |
| Azure 2023 code | default (2048, no cap) | ×0.5 | 0.22 / 19.1 | 32.7% | 15.9% / 19.1% | 15 / 33 |
| Azure 2023 code | budget 8192 | ×1 | 0.61 / 23.7 | 51.6% | 28.1% / 49.7% | 82 / 152 |
| Azure 2023 code | 2048, cap 512/request | ×0.5 | 0.24 / 19.1 | 40.4% | 29.6% / 51.0% | 48 / 73 |
| BurstGPT (first 20k) | default (2048, no cap) | ×100 | 0.05 / 0.5 | 7.2% | 0.6% / 1.2% | 0 / 3 |
| BurstGPT (first 20k) | budget 8192 | ×100 | 0.05 / 0.3 | 5.7% | 0.0% / 0.3% | 0 / 0 |
| BurstGPT (first 20k) | 2048, cap 512/request | ×100 | 0.07 / 0.7 | 14.8% | 1.4% / 3.7% | 2 / 6 |
| Mooncake conversation (3 s slot spread) | default (2048, no cap) | ×0.2 | 0.66 / 16.4 | 7.6% | 7.1% / 7.5% | 19 / 89 |
| Mooncake conversation (3 s slot spread) | budget 8192 | ×0.2 | 0.46 / 4.8 | 13.3% | 13.2% / 17.7% | 173 / 440 |
| Mooncake conversation (3 s slot spread) | 2048, cap 512/request | ×0.1 | 0.46 / 4.3 | 11.4% | 11.1% / 18.8% | 32 / 73 |
| Mooncake tool-agent (3 s slot spread) | default (2048, no cap) | ×0.2 | 0.24 / 22.8 | 10.7% | 10.2% / 11.3% | 41 / 116 |
| Mooncake tool-agent (3 s slot spread) | budget 8192 | ×0.2 | 0.14 / 5.8 | 13.6% | 13.4% / 22.8% | 170 / 483 |
| Mooncake tool-agent (3 s slot spread) | 2048, cap 512/request | ×0.1 | 0.10 / 3.7 | 11.7% | 11.5% / 19.1% | 30 / 71 |
| Mooncake synthetic | default (2048, no cap) | ×0.25 | 1.00 / 7.1 | 7.9% | 6.4% / 6.6% | 18 / 109 |
| Mooncake synthetic | budget 8192 | ×0.25 | 0.81 / 5.1 | 18.2% | 15.1% / 20.4% | 272 / 547 |
| Mooncake synthetic | 2048, cap 512/request | ×0.25 | 0.55 / 9.7 | 37.0% | 35.8% / 54.5% | 160 / 273 |

Readings:

- **Where the aggregate coordinate is enough:** short-prompt chat. On BurstGPT the
  multi-prefill steps that do occur are mostly short prompts starting at depth 0, which carry no
  geometry error.
- **Where it is not:** long prompts, and above all prefix-cached context (Mooncake). A
  cache-hit request enters at a deep KV depth next to other requests, and that is exactly the
  mixed-depth geometry of the partition cells. This is the traffic of agentic and long-context
  serving.
- **Larger budgets make each mispriced step worse.** At budget 8192 fewer steps are affected on
  chat traffic, but on long-context traffic each affected step is mispriced by hundreds of ms.
- **Load matters.** Every share grows with load (all rows in the JSON files), because
  multi-prefill steps need several requests in prefill at once.

## Limits

- The traces come from other clusters and models. Lengths are replayed as recorded, and the
  arrival rate is rescaled to one A100.
- Prefix sharing is modeled only where the trace carries hashes (Mooncake).
- The clock is a fitted model. The error formula assumes an aggregate model that is exact on
  one-prefill steps; M0 fitted on one-prefill steps is close to that (primary-split MAE 171 ms
  on A100 comes from exactly this term).
- No goodput or SLO claim is made here; that needs the live controller on a replayed trace.

## Data

Public traces, downloaded 2026-09-23 (SHA-256 in [`trace-SHA256SUMS`](trace-SHA256SUMS); not
redistributed here):

- Azure LLM inference trace 2023 (`AzureLLMInferenceTrace_conv.csv`, `_code.csv`),
  github.com/Azure/AzurePublicDataset
- Mooncake FAST'25 traces (`conversation_trace.jsonl`, `toolagent_trace.jsonl`,
  `synthetic_trace.jsonl`), github.com/kvcache-ai/Mooncake
- BurstGPT (`BurstGPT_1.csv`, first 20,000 non-failed requests), github.com/HPMLL/BurstGPT

Reproduce (paths to the downloaded files):

```bash
python scripts/simulate_geometry_prevalence.py --steps benchmarks/results/a100-prefill-cost-geometry/raw/Qwen3-4B/steps.csv --slope 3.3 --kv-capacity 400000 --rate-scales 0.1,0.25,0.5,1,1.5,2 --output out.json --trace azure-conv=azure:AzureLLMInferenceTrace_conv.csv azure-code=azure:AzureLLMInferenceTrace_code.csv mooncake-synthetic=mooncake:synthetic_trace.jsonl
python scripts/simulate_geometry_prevalence.py --steps benchmarks/results/a100-prefill-cost-geometry/raw/Qwen3-4B/steps.csv --slope 3.3 --kv-capacity 400000 --rate-scales 25,50,100,200,400 --output out-burst.json --trace burstgpt=burstgpt:BurstGPT_1.csv
python scripts/simulate_geometry_prevalence.py --steps benchmarks/results/a100-prefill-cost-geometry/raw/Qwen3-4B/steps.csv --slope 3.3 --kv-capacity 400000 --rate-scales 0.05,0.1,0.2,0.3 --output out-mc.json --trace mooncake-conv-jit3s=mooncake:conversation_trace.jsonl@3 mooncake-toolagent-jit3s=mooncake:toolagent_trace.jsonl@3
```
