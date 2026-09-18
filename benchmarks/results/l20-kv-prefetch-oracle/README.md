# Oracle request-free KV prefetch for agentic sessions — upper bound (L20, Qwen3-4B, vLLM 0.29)

Status: **campaigns 26 (oracle upper bound) and 27 (timing uncertainty) done; campaign 28
(capacity-aware admission) pending.**

**Question.** vLLM reloads a session's KV from the CPU tier only when the next request arrives
(reactive). If the resume time were known, a request-free prefetch during the tool wait could take
the CPU→GPU transfer off the critical path. How much is there to gain, and what does it cost the
rest of the engine?

**Answer so far.** With perfect timing the resume TTFT drops 52–57% for 4k–16k prefixes (the load
is 1.2–1.4× the resume's own prefill, at 6.4 µs/token ≈ 23 GB/s); the useful lead time is exactly
the load time (34 / 65 / 104 ms unloaded, ~80–100 ms under 16 concurrent decoders), an unfinished
prefetch is worse than reactive (lead 0: +20%), and once the session does not fit next to the
running decoders (16k prefix into a 24k-token GPU cache with 16 decoders) both reactive load and
prefetch produce multi-second admission stalls — the capacity trade-off is real and is the next
question.

## Setup

`vllm serve Qwen3-4B --max-model-len 20480 --max-num-seqs 64 --kv-offloading-size 7
--num-gpu-blocks-override 1536 --gpu-memory-utilization 0.9` (native CPU offloading; the host has
15 GB RAM, so the CPU tier is 7 GiB ≈ 48k tokens and the GPU cache is capped at 24.5k tokens so
that filler traffic can evict a session; 32k prefixes are out of reach on this host). Harness
[`scripts/measure_kv_prefetch.py`](../../../scripts/measure_kv_prefetch.py): per trial, turn 1
(P random tokens + 32 generated; on finish the blocks are stored to CPU), filler (32k distinct
tokens over 16 requests → evicts the session from the GPU cache), then turn 2 (same prefix + 64
appended tokens, 32 generated; TTFT measured). Arms: **A** retained (no filler; blocks stay in GPU),
**B** reactive (shipped: load triggered by the resume), **C** oracle prefetch emulated at block level
by a prefix-only `max_tokens=1` probe issued `lead` ms before the resume (the probe's decode step
is overhead a real hook would not pay). Arms interleaved per (prefix, repeat); 3 repeats; a second
pass with 16 background decoders (128-token prompts, 2048-token generations) whose per-token
arrival times are recorded. Analysis: [`scripts/analyze_kv_prefetch.py`](../../../scripts/analyze_kv_prefetch.py).

## 1. TTFT_resume(lead), no background (median of 3, ms)

| prefix | A retained | B reactive | C 0 | C 10 | C 25 | C 50 | C 100 | C 250 | C 500 | knee | load | oracle gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4k | 42 | 75 | 91 | 68 | 64 | **35** | 38 | 45 | 50 | 50 ms | 34 ms | 53% |
| 8k | 57 | 121 | 135 | 121 | 97 | 76 | **52** | 62 | 57 | 100 ms | 65 ms | 57% |
| 16k | 91 | 194 | 204 | 200 | 172 | 155 | 101 | **90** | 91 | 250 ms | 104 ms | 54% |

(min–max ranges in `campaign26-analysis.json`; all within ±10 ms except where noted.) The load
costs 6.4 µs per token of prefix; TTFT_reactive = TTFT_retained + load, so the attainable gain is
load / (retained + load), i.e. structurally ~50–60% for this model/PCIe class.

## 2. With 16 concurrent decoders (GPU cache saturated)

| prefix | A | B reactive | C 50 | C 100 | C 250 | knee | load | gain | note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4k | 71 | 152 | 114 | 77 | **67** | 250 ms | 80 ms | 56% | load slows under contention |
| 8k | 81 | 182 | 148 | 105 | **87** | — | 101 ms | 52% | |
| 16k | 91 | 258 … 31,329 | — | 138 … 31,302 | 82 … 30,694 | — | — | — | admission stalls: session + decoders exceed the GPU cache |

Decoder ITL inside prefetch/resume windows vs steady state: p50 16.8 / p95 25.6 ms vs 16.8 / 33.1
— the transfer itself does not disturb decoders; the 16k stalls (ITL max 5–12 s) are block
starvation, not bandwidth.

## 3. Timing uncertainty (campaign27): predictors are fragile, "immediate" is not

Tool waits drawn lognormal (median 600 ms, σ 0.6 → p10 341 / p50 597 / p90 1817 ms), policies
interleaved per (prefix, repeat), 6 repeats; `raw/campaign27/`.

| prefix | reactive | immediate (prefetch at wait start) | fixed lead (knee before the median wait) | EWMA α=0.3 | oracle |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4k | 79 ms | **45 (−43%)** | 50 (−37%), late 2/6 | 63 (−21%), late 2/6 | 43 (−45%) |
| 8k | 120 | **60 (−50%)** | 116 (−3%), late 4/6 | 114 (−5%), late 5/6 | 61 (−49%) |
| 16k | 196 | **81 (−58%)** | 90 (−54%), late 1/6 | 143 (−27%), late 3/6 | 89 (−54%) |
| GPU residency before resume (p50) | 0 | 0.5–1.2 s | 0–0.6 s | 0.1–0.25 s | 50–250 ms |

The loss is asymmetric: a late prefetch gains nothing, an early one pays only residency. A
resume-time predictor would need error well below the load time (34–104 ms) to beat "prefetch
immediately", which realistic tool-latency variance does not permit; immediate prefetch recovers
95–100% of the oracle gain. The decision that matters is therefore not *when* but *whether* a
paused session may hold GPU blocks during its wait — a capacity-aware admission problem
(campaign28).

## Gate

Pre-registered: kill if oracle TTFT gain <5% or the transfer was already overlapped; strong if
resume TTFT ↓ >20% with an optimal lead region. **Strong**: 52–57% with the knee at the load time.

## Limitations

Block-level emulation (probe), not a scheduler hook; artificially small GPU cache and 7 GiB CPU tier
(host RAM); random-token prefixes; PCIe-gen4-class H2D bandwidth; synthetic tool wait; no
multi-session contention yet (campaign28).
