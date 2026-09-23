# Live deadline controller on the L20 with the pre-registered M2n (Qwen3-4B)

**Question.** The published L20 live run (campaign20) compared M0 and the published M2. The A100
replication added M2n (M2 without the aggregate prefill-KV term). Does M2n change the L20
picture?

**Answer.**
- **M2n beats M0 and M2.** M2n gives **1.59× / 3.46×** M0's safe prefill throughput (N=4 / N=8)
  and +6% / +24% over the published M2, all at 0% violations of the 100 ms deadline.
- **M2n finds the operating point the fixed budgets cannot hold safely.** It runs at 512 tokens
  on most steps and still has **0% violations**, because it drops the budget on the expensive
  steps. fixed-512 reaches about the same throughput but violates the deadline on 5.4% / 10.6% of
  steps.
- **Against the pre-registered fixed arms** (fixed-256 is the best with ≤ 5% violations), M2n is
  **+16% / +32%**. That passes the 15% gate (L3), which I had predicted would fail.
- **Against fixed-384 from the earlier L20 calibration block** (campaign21; 0% violations there,
  and fixed-256 in that block agrees with this one to 0.3%), M2n is +11% / +3%. So with a finer
  fixed grid the gain stays below 15%, as on the A100.

Pre-registration: addendum 4 of
[`docs/preregistration/2026-09-23-m2-without-aggregate-prefill-kv.md`](../../../docs/preregistration/2026-09-23-m2-without-aggregate-prefill-kv.md)
(017bd22, committed before launch).

## Setup

- **Protocol:** as L20 campaign20 ([`l20-prefill-cost-geometry/` §5](../l20-prefill-cost-geometry/README.md)).
  The workload is 8 decoders (4,096 output tokens each) plus N ∈ {4, 8} × 16k prefills injected
  together, with `--max-num-batched-tokens 8192`. Every run uses a fresh server; arms are
  interleaved; 3 repeats. The script is [`campaign/campaign30.sh`](campaign/campaign30.sh); it
  refuses a busy GPU and, unlike campaign20, kills nothing.
- **Controller:** the installed controller with the `m2n` feature set added. The scheduler's
  feature vector was checked against the replay's on 6,000 random steps (0 mismatches). The
  per-step decision tracer was installed, so the budget column comes from the controller's own
  records.
- **Models ([`models/`](models/)):** m0-one and m2-one are the published files (same sha256).
  m2n-one was exported from the same L20 traces (campaigns 18 + 19). A refit of m2-one on those
  traces reproduces the published weights to 3e-11. The q95 margin for M2n is 11.42 ms.

## Results

[`live.md`](live.md): mean (min–max) over 3 repeats; violations are CUDA step times over 100 ms.

| N | arm | violations | prefill step p95 | safe prefill tok/s | budget p50 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | fixed-256 | 0.0% | 62.1 | 5,622 (5,553–5,686) | 256 |
| 4 | fixed-512 | **5.4%** (3.1–7.0) | 100.0 | 6,441 (6,262–6,641) | 512 |
| 4 | M0 | 0.0% | 54.5 | 4,112 (4,065–4,137) | 128 |
| 4 | M2 (published) | 0.0% | 80.0 | 6,154 (6,026–6,219) | 256 |
| 4 | **M2n** | **0.0%** | 90.8 | **6,531 (6,506–6,545)** | 512 |
| 8 | fixed-256 | 0.0% | 84.4 | 4,464 (4,461–4,466) | 256 |
| 8 | fixed-512 | **10.6%** (10.2–11.3) | 103.1 | 5,943 (5,815–6,049) | 512 |
| 8 | M0 | 0.0% | 76.9 | 1,700 (1,693–1,705) | 64 |
| 8 | M2 (published) | 0.0% | 83.1 | 4,760 (4,760–4,761) | 256 |
| 8 | **M2n** | **0.0%** | 92.0 | **5,886 (5,867–5,904)** | 512 |

## Pre-registered predictions

| | N=4 | N=8 | verdict |
| --- | --- | --- | --- |
| L1: M2n ≥ M0, violations ≤ 5% | 1.59×, 0% | 3.46×, 0% | **held** |
| L2: M2n within ±10% of M2 | +6.1% | **+23.7%** | **failed** (M2n better than predicted at N=8) |
| L3: M2n ≥ 1.15 × best fixed arm with ≤ 5% violations (fixed-256) | 1.16× | 1.32× | **held**, although I had predicted it would fail |

**Context for L3** (not part of the pre-registered comparison): the earlier L20 block
([`live-calibration.json`](../l20-prefill-cost-geometry/live-calibration.json), campaign21) also
ran fixed-384.

| N | fixed-256 in campaign21 | fixed-256 here | fixed-384 in campaign21 | M2n here |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 5,619 | 5,622 | 5,887 (0%) | 6,531: **+11%** over fixed-384 |
| 8 | 4,453 | 4,464 | 5,742 (0%) | 5,886: **+3%** over fixed-384 |

The fixed-256 anchor agrees across the two blocks to 0.3%, so the comparison is fair in practice,
but it crosses blocks. The pre-registered gate passes only because the fixed grid was coarse.

## Across GPUs

| | L20 (100 ms) | A100 (65 ms) |
| --- | --- | --- |
| M2n vs M0 | 1.59× / 3.46× | 2.38× / 4.80× |
| M2n vs published M2 | +6% / +24% | +37% / +305% (M2 collapsed at N=8) |
| M2n vs best safe fixed in the same block | +16% / +32% (grid 256/512) | −2% / +2% (grid 128/256/512, 768/1024 unsafe) |
| M2n vs best safe fixed in any block | +11% / +3% (fixed-384) | −2…+7% / +2…+4% |
| M2n's operating point | 512 tokens at 0% violations; fixed-512 violates 5–11% | 512 tokens; fixed-512 is safe on the A100 |

The pattern is the same on both GPUs. M2n removes the aggregate model's starvation and the
published M2's over-pricing. It reaches the best safe fixed budget without tuning. It beats a
budget tuned in hindsight only by the margin that a coarse fixed grid leaves.
