# Section-six available-evidence audit

Completed runs: 11/12. Incomplete subgroups are NOT given confirmatory intervals. All data and failures are retained.

|GPU|Version|Regime|Both no-op labels resolve 1%|Cells|
|---|---|---|---:|---:|
|NVIDIA GeForce RTX 4090|0.6.18|eager_one_warm|6|24|
|NVIDIA GeForce RTX 4090|0.6.18|graph_one_warm|8|24|
|NVIDIA GeForce RTX 4090|0.6.18|graph32_steady|21|24|
|NVIDIA GeForce RTX 4090|0.6.18|graph_one_pressure128MiB|9|24|
|NVIDIA GeForce RTX 5090|0.6.18|eager_one_warm|5|24|
|NVIDIA GeForce RTX 5090|0.6.18|graph_one_warm|9|24|
|NVIDIA GeForce RTX 5090|0.6.18|graph32_steady|19|24|
|NVIDIA GeForce RTX 5090|0.6.18|graph_one_pressure128MiB|8|24|
|NVIDIA GeForce RTX 5090|0.7.0|eager_one_warm|6|24|
|NVIDIA GeForce RTX 5090|0.7.0|graph_one_warm|10|24|
|NVIDIA GeForce RTX 5090|0.7.0|graph32_steady|22|24|
|NVIDIA GeForce RTX 5090|0.7.0|graph_one_pressure128MiB|8|24|

## Two-request reversal (BF16 / unsplit)

|GPU|Version|Regime|Native/causal-heavy [95% interval]|1% control resolution|
|---|---|---|---|---|
|NVIDIA GeForce RTX 4090|0.6.18|eager_one_warm|0.580424 [0.580125, 0.580681]|True|
|NVIDIA GeForce RTX 4090|0.6.18|graph_one_warm|0.571924 [0.571694, 0.572160]|True|
|NVIDIA GeForce RTX 4090|0.6.18|graph32_steady|0.570404 [0.570328, 0.570475]|True|
|NVIDIA GeForce RTX 4090|0.6.18|graph_one_pressure128MiB|0.572259 [0.571858, 0.572736]|True|
|NVIDIA GeForce RTX 5090|0.6.18|eager_one_warm|0.570612 [0.569355, 0.571991]|True|
|NVIDIA GeForce RTX 5090|0.6.18|graph_one_warm|0.563234 [0.562839, 0.563733]|True|
|NVIDIA GeForce RTX 5090|0.6.18|graph32_steady|0.561657 [0.561558, 0.561745]|True|
|NVIDIA GeForce RTX 5090|0.6.18|graph_one_pressure128MiB|0.564088 [0.563816, 0.564360]|True|
|NVIDIA GeForce RTX 5090|0.7.0|eager_one_warm|0.567857 [0.566927, 0.568974]|True|
|NVIDIA GeForce RTX 5090|0.7.0|graph_one_warm|0.560199 [0.559884, 0.560512]|True|
|NVIDIA GeForce RTX 5090|0.7.0|graph32_steady|0.559000 [0.558908, 0.559080]|True|
|NVIDIA GeForce RTX 5090|0.7.0|graph_one_pressure128MiB|0.561332 [0.561060, 0.561622]|True|

Single-call and graph32 are different reuse/cache regimes. Intervals are conditional on this exposed suite and not adjusted for multiple comparisons. No current source, clock lock, or engine-wide performance guarantee follows from a passing cell.
