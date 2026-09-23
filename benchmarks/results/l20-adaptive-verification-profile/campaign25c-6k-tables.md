## Per cell (mean over repeats): goodput, step/draft ms, chosen budget, cost-prediction error
| actual class | B | profile ctx | goodput | step ms | draft ms | acc/row | budget p50 / max (at max) | pred ms p50 | err p5 / p50 / p95 (actual − pred) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-ctx6000 | 24 | 512 | 2113 | 71.6 | 14.9 | 5.30 | 160 / 168 (25%) | 21.1 | +42.3 / +50.5 / +54.9 |
| code-ctx6000 | 24 | 6144 | 2141 | 71.2 | 15.0 | 5.35 | 160 / 168 (19%) | 28.6 | +36.4 / +42.6 / +48.0 |
| code-ctx6000 | 24 | 8192 | 2141 | 71.6 | 14.9 | 5.39 | 160 / 168 (36%) | 31.6 | +34.1 / +40.0 / +44.9 |
| prose-ctx6000 | 24 | 512 | 1177 | 60.8 | 14.1 | 1.98 | 104 / 168 (0%) | 19.0 | +39.5 / +41.8 / +44.4 |
| prose-ctx6000 | 24 | 6144 | 1141 | 61.1 | 14.3 | 1.90 | 104 / 168 (0%) | 27.1 | +31.4 / +34.0 / +37.3 |
| prose-ctx6000 | 24 | 8192 | 1148 | 61.0 | 14.3 | 1.92 | 104 / 168 (0%) | 30.1 | +28.4 / +30.9 / +34.2 |

## Decision regret vs the matched profile (profile whose context is closest to the actual context)
| actual class | B | matched profile | profile | goodput | regret vs matched | budget disagreement (share of max, + over / − under) | wrong-direction steps (≥1 tier) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prose-ctx6000 | 24 | 6144 | 512 | 1177 | -3.2% | +0% | 0% |
| prose-ctx6000 | 24 | 6144 | 6144 (matched) | 1141 | +0.0% | +0% | 0% |
| prose-ctx6000 | 24 | 6144 | 8192 | 1148 | -0.6% | +0% | 0% |
| code-ctx6000 | 24 | 6144 | 512 | 2113 | +1.3% | +0% | 5% |
| code-ctx6000 | 24 | 6144 | 6144 (matched) | 2141 | +0.0% | +0% | 0% |
| code-ctx6000 | 24 | 6144 | 8192 | 2141 | -0.0% | +0% | 0% |
