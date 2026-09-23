Highest simulated load with TTFT p50 < 1 s and < 10% of steps outside the measured clock range.

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
