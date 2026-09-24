# Live trace replay: prevalence confirmed, but a deadline controller does not raise SLO goodput

**Question.** On public request traces replayed against a live vLLM (A100, Qwen3-4B):
1. Does the mispriced multi-prefill geometry occur where the simulator says it does?
2. Does a geometry-aware controller turn that into SLO goodput, against vLLM defaults, fixed
   budgets, the AgentX-style per-request cap and P-PAS?
3. Does it hold up when the workload shifts phase?

**Answers.**
1. **Yes, and the simulator predicted it.**
   - Steps whose aggregate-coordinate mispricing exceeds 5 ms are **12–18%** of prefill steps on
     Azure 2023 code, **4–9%** on Mooncake tool-agent (prefix-cached) and **≤ 0.1%** on BurstGPT.
   - The live values match the simulator within 0.3–3 points in every loaded cell, including four
     robustness windows chosen by a frozen rule and a second jitter seed.
2. **No.**
   - No controller configuration beats the best static arm by more than 1% in any cell (the
     largest margin is 1.87 vs 1.86 req/s, one repeat, Azure-code window 1050 s), and none is
     best under any of nine SLO pairs in the workload shift.
   - With a 65 ms per-step deadline the controllers collapse on agent traffic: 0.06 req/s against
     0.98. Filling every step to the deadline admits less prefill per second than the stock
     2048-token budget, so the queue grows.
   - Setting the deadline equal to the TPOT SLO (100 ms) recovers only to 0.24 req/s.
   - Per-step deadline control is the wrong objective for TTFT + TPOT goodput.
   - Within the same controller, geometry pricing (M2n) is never worse than the aggregate
     coordinate (M0), and is better where it matters (Mooncake ×0.2: 0.24 vs 0.13 req/s at 100 ms;
     workload-shift regret 31% vs 39% at 65 ms). The representation claim survives; the policy
     does not.
3. **The premise did not appear.**
   - In the chat → code → agent → chat run, one fixed budget (1024) is the best stock arm in
     *every* phase (regret 0.0%).
   - The best fixed budget depends on the SLO (512 at a 50 ms TPOT, 1024 at 100 ms, 2048–4096
     looser), not on the phase.
   - P-PAS lands at 4.5–4.8% regret, and the SLO-aligned M2n controller at 5.9%.

Pre-registered in addenda 4, 5, 7, 7b, 8 and 10 of
[`docs/preregistration/2026-09-23-m2-without-aggregate-prefill-kv.md`](../../../docs/preregistration/2026-09-23-m2-without-aggregate-prefill-kv.md).
Every verdict below, including the failed ones, is against those files.

## Setup

- **Hardware and model:** A100-SXM4-80GB (RunPod, two GPUs, one run per GPU at a time), vLLM 0.29.0,
  Qwen3-4B.
- **Server:** prefix caching on, `max-num-seqs` 256, 240 s arrival window per run, then drain.
- **Harness:** [`scripts/replay_trace_serving.py`](../../../scripts/replay_trace_serving.py).
  Synthetic token ids with the trace's lengths. Mooncake block hashes map to fixed 512-token
  blocks, so shared prefixes are real prefix-cache hits. Output length is the trace's, with
  `ignore_eos`.
- **Traces:** Azure 2023 code and conversation; Mooncake tool-agent, with the 3 s timestamp slots
  spread uniformly (seed 0, plus seed 1 for robustness); BurstGPT.

**Arms.**

| arm | configuration |
| --- | --- |
| `default` | 2048, the A100 API default |
| `agentx` | 2048 with a per-request cap of 512 |
| `b512` … `b8192` | fixed `--max-num-batched-tokens` |
| `ppas` | P-PAS as published (arXiv 2608.15171, Alg. 1): N_th 2, B_max 16384, B_cap 2048 |
| `ppas-8k` | P-PAS-style, budget-matched (B_max 8192) |
| `ctl-m0-fcfs`, `ctl-m2n-fcfs` | deadline controller from the live-controller artifacts, fcfs partition, ceiling 8192, D = 65 ms or D = 100 ms (`-D100`) |
| `ctl-m2n-fcfs-cache` | prices waiting requests at their prefix-cache hit depth |
| `ctl-m2n-equal` | equal partition |

**SLOs.** Primary: TTFT ≤ 5 s and TPOT ≤ 100 ms. Goodput = requests meeting both ÷ arrival
window.

**Repeats.** 2 per cell; 4 for the workload shift (2 per GPU).

**Windows** ([`window-rule.json`](window-rule.json)). The rule is frozen in addendum 8 D. The
Mooncake primary windows meet it. The BurstGPT primary window meets it but is not the earliest. No
Azure-code window meets the 20 s gap limit (the trace arrives in minute-long bursts); with the gap
relaxed to 60 s (a reported deviation), the primary t = 180 s window is the earliest.

**Two setup corrections, both committed before the affected runs (addenda 7 and 7b).**
- The first Azure-code windows replayed the same 63 requests, because the trace is empty from 60 s
  to beyond 120 s. They are kept and reported as unsaturated.
- The first BurstGPT window held 86 requests. It was replaced by the 14 h window.

## 1. Where the geometry occurs: simulator vs live

[`sim-vs-live.md`](sim-vs-live.md), `default` arm, 240 s:

| cell | P(geometry error > 5 ms) live / sim | TTFT p50 live / sim (s) |
| --- | ---: | ---: |
| Mooncake ×0.1 / ×0.2 | 3.9% / 4.2% · 5.9% / 6.9% | 0.17 / 0.15 · 0.26 / 0.20 |
| Mooncake ×0.2, windows 60 s / 150 s | 6.8% / 7.9% · 8.5% / 9.9% | 0.22 / 0.16 · 0.26 / 0.20 |
| Mooncake ×0.2, jitter seed 1 | 7.7% / 7.7% | 0.20 / 0.15 |
| Azure code t180 ×0.25 / ×0.5 | 15.2% / 15.7% · 17.6% / 18.9% | 0.22 / 0.18 · 0.35 / 0.32 |
| Azure code ×0.5, windows 540 s / 1050 s | 15.6% / 18.6% · 11.8% / 13.4% | 0.34 / 0.32 · 0.21 / 0.17 |
| BurstGPT ×60, three windows | 0.0–0.1% / 0.0–0.5% | 0.04–0.05 / 0.03–0.04 |

- **V1: held.** Mooncake and Azure code are far above BurstGPT in every window.
- **V2: failed in 1 of 13 cells** (BurstGPT window 64800 s: 0.1% vs 0.5%, both near zero). Two
  more BurstGPT cells are too close to zero for a ratio. It holds in all 10 others.

The simulator used for the prevalence study
([`trace-geometry-prevalence/`](../trace-geometry-prevalence/README.md)) is therefore a faithful
proxy for where the mispriced geometry occurs.

## 2. SLO goodput by arm

[`replay-summary.md`](replay-summary.md): primary goodput in req/s, mean of repeats.

| arm | Mooncake ×0.1 | Mooncake ×0.2 | Azure code t180 ×0.25 | Azure code t180 ×0.5 | BurstGPT t14h ×60 |
| --- | ---: | ---: | ---: | ---: | ---: |
| offered | 0.54 | 1.02 | 2.21 | 2.99 | 4.63 |
| `default` | **0.54** | **0.98** | **2.21** | 2.93 | **4.63** |
| `agentx` | **0.54** | 0.95 | **2.21** | **2.96** | **4.63** |
| `b8192` | **0.54** | **0.98** | 2.13 | 2.26 | **4.63** |
| `ppas` / `ppas-8k` | — | **0.98 / 0.98** | — | — | — |
| `ctl-m0-fcfs` (65 ms) | 0.49 | 0.06 | **2.21** | 2.78 | **4.63** |
| `ctl-m2n-fcfs` (65 ms) | 0.51 | 0.06 | **2.21** | 2.81 | **4.63** |
| `ctl-m2n-fcfs-cache` (65 ms) | — | 0.06 | — | — | — |
| `ctl-m2n-equal` (65 ms) | 0.52 | 0.30 | **2.21** | 2.86 | **4.63** |
| `ctl-m0-fcfs-D100` | — | 0.13 | — | — | — |
| `ctl-m2n-fcfs-D100` | — | 0.24 | — | — | — |
| `ctl-m2n-fcfs-cache-D100` | — | 0.21 | — | — | — |

**Robustness windows** (one repeat each):
- Mooncake 60 s / 150 s: `default` 1.06 / 1.25, `ppas` 1.05 / 1.24, `ctl-m2n-fcfs` 0.30 / 0.05.
- Azure code 540 s / 1050 s: `default` 3.42 / 1.86, `ppas` 2.85 / 1.77, `ctl-m2n-fcfs` 2.63 / 1.87.
- BurstGPT: all arms equal.
- Mooncake jitter seed 1: `default` 0.97, `ppas` 0.97, `ctl-m2n-fcfs` 0.06.

**Pre-registered verdicts.**

| | verdict |
| --- | --- |
| R1 (M2n ≥ M0) | held at ×0.1 (0.51 vs 0.49); tie at ×0.2 |
| R2 (equal partition has worse TTFT p90 than fcfs) | held at ×0.2 (188 s vs 64 s); failed at ×0.1 |
| R3 (no arm ≥ 1.15 × best stock) | held; the controllers lose |
| R4 (cache-aware ≥ plain on Mooncake) | tie at 65 ms (both collapsed); 0.21 vs 0.24 at 100 ms. The limitation it fixes is real (offline check: a 20k-cached waiting request priced at depth 0 gets 1024 tokens, correctly priced 256) but it cannot help a controller whose objective is wrong |
| R5 (M2n at D = 100 reaches 0.9 × best stock) | **failed** (0.24 vs 0.98): the collapse is not only deadline tightness |
| R6 (M2n ≥ M0 at D = 100) | held (0.24 vs 0.13) |

- **P-PAS:** as good as the best static arm on Mooncake and BurstGPT; worse than `default` in both
  Azure-code robustness windows (2.85 vs 3.42, 1.77 vs 1.86).
- **Errors:** 1 client-side `ServerDisconnectedError` in 313 requests (Mooncake window 150 s,
  `default`); none elsewhere.

## 3. Workload shift

One 600 s run: chat (Azure conv ×1) → code (Azure code ×0.5 from 180 s) → agent (Mooncake
tool-agent ×0.2) → chat again. 4 repeats per arm, 2 per GPU; the repeats agree to ±0.02 req/s.
Primary goodput per phase:

| arm | chat | code | agent | chat2 | regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| `b1024` | 4.09 | 3.54 | 1.00 | 4.09 | **0.0%** |
| `agentx` | 4.09 | 3.47 | 0.93 | 4.09 | 1.1% |
| `b2048` (= `default`) | 4.09 | 3.41 | 0.98 | 4.09 | 1.2% |
| `b512` | 4.09 | 3.22 | 0.80 | 4.09 | 4.1% |
| `ppas-8k` / `ppas` | 4.09 | 2.99 / 2.94 | 0.98 / 0.99 | 4.09 | 4.5% / 4.8% |
| `ctl-m2n-fcfs-D100` (/ `-cache-D100`) | 4.09 | 3.54 | 0.26 / 0.24 | 4.08 | 5.9% / 6.0% |
| `ctl-m0-fcfs-D100` | 4.09 | 3.40 | 0.19 | 4.08 | 7.5% |
| `b4096` / `b8192` | 4.09 | 2.39 | 0.99 | 4.09 | 9.1% / 9.2% |
| `ctl-m2n-fcfs` (/ `-cache`) | 4.09 | 3.26 / 3.25 | 0.09 | 1.28 | 31.4% / 31.5% |
| `ctl-m0-fcfs` | 4.09 | 3.20 | 0.09 | 0.44 | 38.5% |

- **W1 failed:** `b1024` is the best stock arm in every phase.
- **W2, W4, W5 failed:** no controller comes close to 0.0% regret, and P-PAS beats every controller.
- **W3 held:** M2n 31.4% < M0 38.5%.
- The 65 ms controllers' agent-phase backlog carries into chat2.

**SLO sensitivity** ([`slo-sensitivity.md`](slo-sensitivity.md); TTFT {2, 5, 10} s × TPOT
{50, 100, 200} ms):
- The best stock arm moves with the SLO: `b512` at 50 ms TPOT, `b1024` / `b2048` at 100 ms,
  `b4096` / `agentx` at 200 ms.
- No controller is best under any of the nine pairs. P-PAS is closer to the best than the
  controllers in eight of nine; the D = 100 controllers edge it only at 10 s / 100 ms.
- M2n ≥ M0 inside the same controller in all nine pairs at both deadlines.

## 4. Scheduler overhead

The controller decision, timed on the A100 host CPU (AMD EPYC 7513) against the installed module
([`overhead/controller-overhead-a100host.json`](overhead/controller-overhead-a100host.json)): p50
28 µs at 1 active request up to 214 µs at 256 (p99 ≤ 231 µs). That is ≤ ~2% of a 10 ms decode step
and ≤ 0.5% of a 40–65 ms prefill step. It is already included in every live measurement above.

## What this changes in the paper story

The representation result stands and is now tested live:
- the pairing swap;
- seven datasets of OOD prediction;
- the learned baseline;
- deadline A/B runs on two GPUs;
- M2n ≥ M0 in every live controller comparison;
- live prevalence matching the simulator.

The systems claim has to be narrowed. On these traces the mispriced geometry is common on code and
agent traffic, but a per-step deadline controller is not the right consumer of better pricing when
the objective is TTFT + TPOT goodput. A static budget chosen for the SLO is already within a few
percent of the per-phase optimum, and P-PAS within 5%. A goodput-oriented scheduler that uses
geometry pricing, for example in admission or TTFT-aware ordering rather than per-step budgeting,
was not built or tested here.

## Reproduce

Per-run JSONs (with every request) and `default`-arm engine traces are in `raw/`, gzipped. Gunzip,
then:

```bash
python scripts/analyze_trace_replay.py --dir raw/<run set> ... --output out.json
python scripts/analyze_slo_sensitivity.py --dir raw/<run set> ... --output slo.json
python scripts/select_replay_windows.py --steps benchmarks/results/a100-prefill-cost-geometry/raw/Qwen3-4B/steps.csv --cell ... --output windows.json
```

Campaign scripts and logs are in [`campaign/`](campaign/). Old arms kept the venv of their earlier
repeats, and new arms (P-PAS, cache-aware) used a copy with the extended controller.
