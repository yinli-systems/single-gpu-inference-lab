# Controlled permutation results

Kernel-only results; shape contrasts are not optimization speedups. All formal cells retained.
Intervals are conditional on three process repeats and twelve blocks; no device-population or familywise claim.

|GPU|dtype|n|same-opposite, us [95% CI]|rho(W,time)|order-null equivalent|equal-W equivalent|
|---|---|---|---|---|---|---|
|NVIDIA GeForce RTX 4090|bfloat16|n16|4031.968 [4020.555, 4044.064]|0.960|False|True|
|NVIDIA GeForce RTX 4090|bfloat16|n2|54.777 [54.181, 55.628]|1.000|True|N/A|
|NVIDIA GeForce RTX 4090|bfloat16|n4|1876.405 [1870.047, 1883.134]|0.829|False|False|
|NVIDIA GeForce RTX 4090|bfloat16|n8|1760.087 [1675.100, 1814.897]|0.924|False|False|
|NVIDIA GeForce RTX 4090|float16|n16|4038.560 [4025.012, 4051.528]|0.960|False|True|
|NVIDIA GeForce RTX 4090|float16|n2|55.464 [54.403, 56.729]|1.000|True|N/A|
|NVIDIA GeForce RTX 4090|float16|n4|1873.423 [1868.119, 1878.650]|0.848|False|False|
|NVIDIA GeForce RTX 4090|float16|n8|1756.698 [1656.741, 1826.219]|0.924|False|False|
|NVIDIA GeForce RTX 5090|bfloat16|n16|2935.356 [2910.434, 2958.822]|0.936|False|False|
|NVIDIA GeForce RTX 5090|bfloat16|n2|841.797 [838.512, 844.684]|1.000|True|N/A|
|NVIDIA GeForce RTX 5090|bfloat16|n4|1411.733 [1346.616, 1460.392]|0.921|True|False|
|NVIDIA GeForce RTX 5090|bfloat16|n8|1215.422 [1211.836, 1219.885]|0.809|False|False|
|NVIDIA GeForce RTX 5090|float16|n16|2906.062 [2897.079, 2915.221]|0.930|False|False|
|NVIDIA GeForce RTX 5090|float16|n2|845.535 [842.111, 849.454]|1.000|True|N/A|
|NVIDIA GeForce RTX 5090|float16|n4|1377.079 [1298.676, 1433.782]|0.902|True|False|
|NVIDIA GeForce RTX 5090|float16|n8|1217.532 [1211.568, 1222.468]|0.809|False|False|
