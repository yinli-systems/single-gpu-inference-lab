# Native plan-plus-run result

Pristine official0.7.0 versus disabled and enabled isolated native source. Single exposed witness. Three processes per mode; independent block resampling across modes.

|GPU|Actual calls per plan|Pristine/disabled [95% CI]|Pristine/candidate [95% CI]|All mode A/A controls resolve1%|
|---|---:|---|---|---|
|NVIDIA GeForce RTX 4090|1|0.999355 [0.996397, 1.002170]|1.008597 [1.003758, 1.014097]|True|
|NVIDIA GeForce RTX 4090|32|1.000689 [0.999806, 1.002426]|1.011484 [1.007225, 1.019333]|True|
|NVIDIA GeForce RTX 5090|1|1.000179 [0.998276, 1.002103]|1.017900 [1.016035, 1.019839]|True|
|NVIDIA GeForce RTX 5090|32|1.000024 [0.999750, 1.000359]|1.020943 [1.019619, 1.021756]|True|

No unconditional promotion. Exact-witness support is intentionally narrow. Paged cache, dynamic graph plans, threaded/multistream ownership, new shapes and real full-model serving are not qualified. A1% result is unresolved when its matching A/A controls fail; graph32 does not stand in for32 different transformer layers.
