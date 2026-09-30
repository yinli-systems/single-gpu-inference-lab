# Permutation campaign: same marginals, many pairings (L20)

**Question.** The pairing swap showed that re-pairing two chunks with two depths changes the step. Does
that hold at scale, and is the effect set by the coupling C = Σ qᵢkᵢ? Here the chunk multiset and the
depth multiset are fixed and the same n chunks are re-paired to the same n depths in many ways. Every
state has the same prefill count, token sum, KV sum, chunk and depth multisets and decode state; only
the pairing differs.

**Registration.** [`docs/preregistration/2026-09-28-permutation-campaign.md`](../../../docs/preregistration/2026-09-28-permutation-campaign.md),
run on the L20 under addendum 1 (committed at `98a4883` before any L20 data). The A100 registration in
the same file was not executed: the pod became unreachable mid-run and no A100 data was retrieved.

**Setup.** NVIDIA L20 48 GB, the PyPI vLLM 0.29.0 wheel unmodified plus tracer v2, torch 2.13.0+cu130,
Qwen3-4B bf16 (published weights), FLASH_ATTN backend. Configs n2, n4, n8, n16 (depths up to 28k,
chunks 32–1024); per config: `same` (C_max), `opposite` (C_min), 6 random pairings, two pairings with
equal C (n ≥ 4) and an order null (the `same` pairing submitted in reverse order). 12 randomized
blocks, 2 warm-up rounds dropped, layout seeds 0 and 1, cells interleaved. Each state's n requests
are enqueued while scheduling is paused, so the next step holds exactly the n chunks plus one decode
row; every executed step is identified by its (chunks, depths) in the engine trace.
Harness [`measure_pairing_permutations.py`](../../../scripts/measure_pairing_permutations.py),
analysis [`analyze_permutation_campaign.py`](../../../scripts/analyze_permutation_campaign.py),
campaign [`campaign/campaign_l20_permutations.sh`](campaign/campaign_l20_permutations.sh).

## Result

**Every registered prediction held.** [`verdicts.json`](verdicts.json); P6 with the L20 slope
c_X = 6.33 ms/M and band 0.8–1.15 (addendum 1).

| cell | T(same) / T(opposite) (ms) | range | minimax floor | P1 ρ(C, T) | P6 ratio | P2 | P3 | P5 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| n2, seed 0 | 175.8 / 139.3 | 36.4 | 18.2 | — | 0.92 | held | held | — |
| n2, seed 1 | 175.9 / 139.3 | 36.6 | 18.3 | — | 0.92 | held | held | — |
| n4, seed 0 | 358.6 / 217.3 | 141.3 | 70.6 | 0.997 | 0.93 | held | held | held |
| n4, seed 1 | 358.4 / 216.8 | 141.6 | 70.8 | 0.997 | 0.93 | held | held | held |
| n8, seed 0 | 371.5 / 269.0 | 102.5 | 51.3 | 0.997 | 0.94 | held | held | held |
| n8, seed 1 | 371.4 / 268.7 | 102.7 | 51.4 | 0.997 | 0.94 | held | held | held |
| n16, seed 0 | 767.9 / 537.3 | 230.6 | 115.3 | 0.985 | 0.93 | held | held | held |
| n16, seed 1 | 776.5 / 542.6 | 233.9 | 117.0 | 0.985 | 0.94 | held | held | held |

P4 (layout): every state's median agrees across the two layout seeds within ±ε, and the state medians
correlate at ρ = 0.991–1.000 across seeds, in all four configs. 1 of 1056 trials was missing (n16,
seed 0); no other exclusions.

- **Pairing alone moves the step by 36–234 ms.** With every marginal fixed, `same` is 26–65% slower
  than `opposite` (n4: 359 vs 217 ms). Any predictor that sees only the marginals returns one number
  for all these states, so its worst-case error on them is at least the floor, 18–117 ms here.
- **C orders the states.** Spearman ρ between C and the median step time is 0.985–0.997 over the
  distinct pairings, and the range is 0.92–0.94 of c_X·(C_max − C_min), inside the pre-registered band
  and in line with the L20 pairing swap (0.89–1.03).
- **Two pairings with equal C cost almost the same, but not exactly.** The eqC differences are
  +0.9 to +1.6 ms (n4, n16) and −1.3 ms (n8), 0.6–1.3% of the pairing range. That is inside the
  registered ±ε (7–16 ms), so P5 holds, and their 90% intervals also exclude 0. C explains the
  pairing effect to about 99%; per-request structure beyond C is measurable but small on this FA2 path.
- **Submission order does not matter.** The order null's 90% interval against `same` lies within
  ±0.2 ms in the six n ≤ 8 cells; for n16 it is [−0.21, +0.49] ms (seed 0) and [+0.20, +1.15] ms
  (seed 1), at most about 0.15% of a 770 ms step and far inside ±ε.
- **Physical KV placement does not matter.** Both layout seeds give the same state medians.

GPU telemetry over the run (every 0.5 s, including model loads and idle gaps): SM clock 210–2520 MHz,
power 39–358 W, temperature 44–77 °C.

## Reproduce

The analysis reads uncompressed traces, so decompress into a scratch directory first:

```bash
T=$(mktemp -d); cp raw/perm-*.gz raw/*.plan.json $T/; gunzip $T/*.gz
python scripts/measure_pairing_permutations.py --analyze --trace-dir $T --output $T/perm.json
python scripts/analyze_permutation_campaign.py --input $T/perm.json --output $T/verdicts.json --slope 6.33 --p6-band 0.8 1.15
```

Both outputs match the checked-in [`raw/perm.json`](raw/perm.json) and [`verdicts.json`](verdicts.json)
byte for byte. [`raw-sha256.txt`](raw-sha256.txt) lists the raw files; `raw/` also holds the per-cell
logs, `nvidia-smi.txt`, `pip-freeze.txt` and the telemetry.
