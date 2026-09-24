# A learned aggregate model does not recover the missing pairing

**Question.** LPRS-style schedulers fit a small MLP on aggregate latency features. Is the
geometry-OOD failure of M0 a capacity problem that a flexible learned model fixes, or missing
information?

**Answer.** Missing information:
- **Geometry-OOD error.** The MLP is off by **40–232 ms** MAE on the multi-prefill steps, against
  **1.2–3.1 ms** for M2n, across all seven (GPU, model) datasets. On the reverse split it is worse
  than M0 (49–296 ms).
- **Sample efficiency.** M2n reaches ≤ 5 ms with **64** one-prefill training steps. The MLP and M0
  never reach it at any training size up to 2,048, and the MLP sometimes gets *worse* with more
  one-prefill data (L20 4B: 147 → 244 ms).
- **Geometry exposure.** With 25% of its training set drawn from other multi-prefill partitions,
  the MLP still misses the unseen partitions by 10–49 ms, and M0's floor stays at 22–119 ms.
  Together with the [pairing swap](../prefill-pairing-swap/README.md), that floor is the
  identifiability bound seen from the data side.

Pre-registered in addendum 8 B of
[`docs/preregistration/2026-09-23-m2-without-aggregate-prefill-kv.md`](../../../docs/preregistration/2026-09-23-m2-without-aggregate-prefill-kv.md).
LB1 and LB2 held.

## Model

- **Architecture:** scikit-learn `MLPRegressor`, two hidden layers of 64 with ReLU, early
  stopping on a 15% validation split, standardized inputs, 5 seeds.
- **Inputs:** 16 aggregate and marginal features.
  - decode: batch, Σ decode KV, max decode KV, mean decode KV, decode KV variance;
  - prefill: prefill count, Σq, max q, Σq², Σ prefill KV, max prefill KV, mean prefill KV;
  - interactions and execution: Σq·Σk, total tokens, padded tokens, eager flag.

  It sees every aggregate and marginal statistic but not which chunk goes with which depth.
- **Same splits and filters** as `analyze_m2_variants.py`: first iteration and > 10× median
  dropped.

## Results

[`learned-baseline.md`](learned-baseline.md), [`learned-baseline.json`](learned-baseline.json).

**LB1: split MAE (ms).**

| dataset | primary M0 | primary MLP | primary M2n | reverse M0 | reverse MLP | reverse M2n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L20 Qwen3-4B | 339.8 | 232.0 | 3.07 | 36.8 | 241.9 | 2.52 |
| A100 Qwen3-4B | 171.5 | 130.5 | 1.92 | 18.3 | 121.9 | 2.31 |
| A100 Qwen3-8B | 170.0 | 124.2 | 1.66 | 18.6 | 179.2 | 1.72 |
| L20 Qwen2.5-1.5B | 116.4 | 89.7 | 1.75 | 12.7 | 95.0 | 2.16 |
| L20 Qwen2.5-7B | 235.7 | 147.1 | 2.31 | 23.4 | 296.1 | 2.55 |
| A100 Qwen2.5-1.5B | 55.9 | 39.6 | 2.13 | 6.3 | 48.7 | 1.48 |
| A100 Qwen2.5-7B | 122.7 | 89.1 | 1.23 | 13.4 | 149.3 | 1.60 |

**LB2: primary MAE vs one-prefill training steps (mean of 5 draws, ms).**

| dataset | model | 32 | 64 | 256 | 2048 |
| --- | --- | ---: | ---: | ---: | ---: |
| L20 Qwen3-4B | M2n / MLP / M0 | 7.9 / 178 / 309 | 3.2 / 147 / 341 | 3.0 / 168 / 340 | 3.1 / 244 / 340 |
| A100 Qwen3-4B | M2n / MLP / M0 | 5.5 / 71 / 152 | 2.5 / 79 / 167 | 2.1 / 102 / 172 | 2.0 / 126 / 172 |
| A100 Qwen2.5-7B | M2n / MLP / M0 | 7.3 / 74 / 120 | 1.9 / 76 / 122 | 1.4 / 71 / 123 | 1.3 / 88 / 123 |

The other datasets follow the same pattern; see `learned-baseline.md`.

**LB3 (exploratory).** A fraction f of the training set is replaced by multi-prefill steps from the
2×512 and 4×512 cells, and the test uses only the 4×256 and 8×256 cells. At f = 25% the MAE is
MLP 10–49 ms, M0 22–119 ms and M2n 1.4–2.9 ms. Representative geometry helps the learned model
but does not close the gap, and M0 has a floor that no amount of data removes.

## Reproduce

```bash
python scripts/analyze_learned_baseline.py --steps NAME=path/to/steps.csv ... --output learned-baseline.json
```
