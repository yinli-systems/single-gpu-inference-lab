# Oracle request-free KV prefetch for agentic sessions — upper bound (L20, Qwen3-4B, vLLM 0.29)

Status: **campaigns 26 (oracle upper bound), 27 (timing uncertainty) and 28 (capacity /
admission) done. Research question answered; remaining work is an engineering hook.**

**Question.** vLLM reloads a session's KV from the CPU tier only when the next request arrives
(reactive). If the resume time were known, a request-free prefetch during the tool wait could take
the CPU→GPU transfer off the critical path. How much is there to gain, and what does it cost the
rest of the engine?

**Answer.** (1) With perfect timing the resume TTFT drops 52–57% for 4k–16k prefixes (the load
is 1.2–1.4× the resume's own prefill, at 6.4 µs/token ≈ 23 GB/s); the useful lead time is exactly
the load time (34 / 65 / 104 ms unloaded, ~80–100 ms under 16 concurrent decoders), an unfinished
prefetch is worse than reactive (lead 0: +20%), and once the session does not fit next to the
running decoders (16k prefix into a 24k-token GPU cache with 16 decoders) both reactive load and
prefetch produce multi-second admission stalls. (2) Timing prediction is the fragile part: under
lognormal tool waits, fixed/EWMA predictors fire late in 2–5 of 6 trials and lose most of the gain,
while "prefetch immediately" recovers 95–100% of the oracle. (3) Under sustained occupancy no
admission policy — not even one that knows the resume order — beats reactive on aggregate resume
latency (+10…+57%): a session is evicted only when there is no slack, and a prefetch only pays into
slack, so the two coincide only transiently (capacity that frees up *after* the eviction). The lever
is a slack gate on the block pool, not a value-ranked admission policy.

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

## 3a. Prediction-error frontier (campaign27A)

Prefetch issued at `predicted resume − L*` with a controlled error ε = predicted − true resume
(L* = 50 / 100 ms for 4k / 8k from §1); 5 interleaved repeats per cell; `raw/campaign27a/`.
No catastrophic stalls in any cell (P(TTFT > 1 s) = 0, P(decoder ITL > 500 ms) = 0).

| ε | 4k, no background (floor 42 / reactive 82 ms) | 8k, no background (65 / 123) | 4k, 8 decoders (60 / 145) | early residency | late exposure |
| ---: | ---: | ---: | ---: | ---: | ---: |
| −250 ms | 43 | 57 | 59 | 266–285 ms | 0 |
| −100 | 47 | 58 | 62 | 116–135 | 0 |
| −50 | 47 | 58 | 58 | 66–85 | 0 |
| 0 | 41 | 55 | 95 | 16–35 | 0 |
| +50 | 94 (+15% vs reactive) | 73 | 149 | 0 | 15–34 |
| ≥ +100 | 82 (= reactive) | 123 (= reactive) | 145 (= reactive) | 0 | = load |

Early errors cost only residency, linearly; a late error of one load time forfeits the whole gain,
and a slightly late prefetch is worse than reactive. Under contention the load itself slows (34 →
80 ms at 4k), so the safe region moves earlier than the unloaded knee. Operating rule: issue the
prefetch at least one *loaded* transfer time before the resume — which, given tool-latency
variance, means at the pause.

## 3b. Realizable predictors (campaign27): fragile; "immediate" is not

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

## 4. Capacity and admission (campaign28): prefetch only pays into slack

N paused sessions resume on lognormal waits under 8 background decoders (512-token generations);
no filler — sessions and decoders evict each other from the 24.5k-token GPU cache naturally.
Policies: reactive; immediate-all; oracle-admit (earliest true resume first while Σ prefix ≤ 12k
tokens ≈ nominal free capacity); random-admit and smallest-admit (same cap). 5–6 interleaved
repeats; `raw/campaign28*/`.

| paused-session KV vs free GPU | reactive mean / p50 / p95 | immediate-all | oracle-admit (admitted / non-admitted p50) | smallest-admit | decoder ITL p95 (reactive → immediate) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ≈60% (3×4k) | 63 / 65 / 70 ms | 61 | 60 (60 / —) | — | 16.7 → 17.3 |
| ≈120% (2×4k + 2×8k) | **140 / 131 / 192** | 220 (+57%) | 176 (+26%; 75 / 263) | 154 (+10%; 64 / 171) | 18.6 → 22.8 |
| ≈2× (4×4k + 3×8k) | **257 / 175 / 705** | 374 (+45%) | 299 (+16%; 68 / 331) | 282 (+10%; 76 / 207) | 22.9 → 36.4 |
| 3× with an undersized CPU tier (9 sessions, filler; thrash) | 5,016 | 11,505 | 4,423 (1,389 / 4,975) | 5,201 | — |

At 60% nothing is evicted (reactive is already a GPU hit); above 100% every prefetch displaces live
work: admitted sessions get the full gain, non-admitted ones and the decoders pay more than that.
Only the thrashing extreme (sessions recomputed rather than reloaded) rewards knowing the resume
order. Contrast with campaign26, where filler traffic drained *before* the resume and prefetch
gained 52–58%: the value exists exactly when capacity frees up after the eviction.

## Gate

Pre-registered: kill if oracle TTFT gain <5% or the transfer was already overlapped; strong if
resume TTFT ↓ >20% with an optimal lead region. **Strong**: 52–57% with the knee at the load time.

## Conclusion

Request-free KV prefetch halves agent resume TTFT when the GPU cache has slack (−52…−58% for
4k–16k prefixes, PCIe-class transfer), the practical policy is "prefetch immediately when allowed"
because resume-time prediction cannot beat the load time under realistic tool-latency variance, and
the allow rule is a slack gate on free blocks — not a value-ranked admission — because under
sustained occupancy any prefetch, even oracle-ordered, raises aggregate resume latency by 10–57%.
What remains is engineering: a slack-gated prefetch-on-pause in the offloading connector and a live
A/B on a bursty agent trace.

## Limitations

Block-level emulation (probe), not a scheduler hook; artificially small GPU cache and 7 GiB CPU tier
(host RAM); random-token prefixes; PCIe-gen4-class H2D bandwidth; synthetic tool wait; no
multi-session contention yet (campaign28).
