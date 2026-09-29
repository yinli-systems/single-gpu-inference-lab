# Metadata-only ordering results

Real FlashInfer runs. Run-only ratios exclude metadata setup; these are not serving speedups.
All declared cells, failed controls and process repeats are retained.

|GPU|set|mode|policy|native/candidate ratio [95% CI]|
|---|---|---|---|---|
|NVIDIA GeForce RTX 4090|discovery|eager|identity_repeat|0.99961 [0.99881, 1.00043]|
|NVIDIA GeForce RTX 4090|discovery|eager|request_reverse|1.00085 [0.99736, 1.00344]|
|NVIDIA GeForce RTX 4090|discovery|eager|heavy_first|1.01394 [1.01119, 1.01618]|
|NVIDIA GeForce RTX 4090|discovery|eager|interleave|0.98355 [0.98138, 0.98564]|
|NVIDIA GeForce RTX 4090|discovery|graph|identity_repeat|1.00009 [0.99959, 1.00065]|
|NVIDIA GeForce RTX 4090|discovery|graph|request_reverse|1.00483 [1.00344, 1.00585]|
|NVIDIA GeForce RTX 4090|discovery|graph|heavy_first|1.01346 [1.01237, 1.01440]|
|NVIDIA GeForce RTX 4090|discovery|graph|interleave|0.98431 [0.98321, 0.98534]|
|NVIDIA GeForce RTX 4090|holdout|eager|identity_repeat|1.00013 [0.99976, 1.00051]|
|NVIDIA GeForce RTX 4090|holdout|eager|request_reverse|1.04945 [1.04824, 1.05068]|
|NVIDIA GeForce RTX 4090|holdout|eager|heavy_first|1.04584 [1.04445, 1.04755]|
|NVIDIA GeForce RTX 4090|holdout|eager|interleave|0.97529 [0.97454, 0.97617]|
|NVIDIA GeForce RTX 4090|holdout|graph|identity_repeat|0.99996 [0.99973, 1.00021]|
|NVIDIA GeForce RTX 4090|holdout|graph|request_reverse|1.04865 [1.04718, 1.04950]|
|NVIDIA GeForce RTX 4090|holdout|graph|heavy_first|1.04499 [1.04290, 1.04628]|
|NVIDIA GeForce RTX 4090|holdout|graph|interleave|0.97432 [0.97385, 0.97476]|

A/A failures on NVIDIA GeForce RTX 4090: 1. Median prototype metadata setup 797.3 us (NOT included in run-only ratios).

|NVIDIA GeForce RTX 5090|discovery|eager|identity_repeat|0.99962 [0.99748, 1.00191]|
|NVIDIA GeForce RTX 5090|discovery|eager|request_reverse|1.04301 [1.04161, 1.04444]|
|NVIDIA GeForce RTX 5090|discovery|eager|heavy_first|1.03685 [1.03310, 1.03983]|
|NVIDIA GeForce RTX 5090|discovery|eager|interleave|1.00717 [1.00544, 1.00879]|
|NVIDIA GeForce RTX 5090|discovery|graph|identity_repeat|1.00020 [0.99952, 1.00086]|
|NVIDIA GeForce RTX 5090|discovery|graph|request_reverse|1.03979 [1.03853, 1.04087]|
|NVIDIA GeForce RTX 5090|discovery|graph|heavy_first|1.03489 [1.02893, 1.03846]|
|NVIDIA GeForce RTX 5090|discovery|graph|interleave|1.00426 [1.00320, 1.00534]|
|NVIDIA GeForce RTX 5090|holdout|eager|identity_repeat|1.00011 [0.99977, 1.00044]|
|NVIDIA GeForce RTX 5090|holdout|eager|request_reverse|1.01941 [1.01891, 1.01988]|
|NVIDIA GeForce RTX 5090|holdout|eager|heavy_first|1.02305 [1.02125, 1.02433]|
|NVIDIA GeForce RTX 5090|holdout|eager|interleave|1.00568 [1.00460, 1.00662]|
|NVIDIA GeForce RTX 5090|holdout|graph|identity_repeat|0.99993 [0.99971, 1.00010]|
|NVIDIA GeForce RTX 5090|holdout|graph|request_reverse|1.01950 [1.01916, 1.01972]|
|NVIDIA GeForce RTX 5090|holdout|graph|heavy_first|1.02360 [1.02266, 1.02423]|
|NVIDIA GeForce RTX 5090|holdout|graph|interleave|1.00612 [1.00510, 1.00701]|

A/A failures on NVIDIA GeForce RTX 5090: 11. Median prototype metadata setup 498.6 us (NOT included in run-only ratios).

