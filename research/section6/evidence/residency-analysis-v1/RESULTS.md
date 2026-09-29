# Residency-pressure diagnostic

Unprofiled paired CUDA-event measurements. One exposed BF16/unsplit/D128 shape. Native is the disabled path of the same experimental host source, not a pristine full-engine comparator.

|GPU|Regime|Native48/heavy48|Native48/native64|Native48/heavy64|Native64/heavy64|A/A resolves 1%|
|---|---|---|---|---|---|---|
|NVIDIA GeForce RTX 4090|graph_one_warm|0.57208 [0.57177, 0.57236]|1.00246 [1.00070, 1.00476]|1.00772 [1.00717, 1.00832]|1.00525 [1.00316, 1.00676]|True|
|NVIDIA GeForce RTX 4090|graph32_steady|0.57012 [0.56995, 0.57025]|1.00304 [1.00267, 1.00335]|1.00718 [1.00703, 1.00736]|1.00413 [1.00374, 1.00453]|True|
|NVIDIA GeForce RTX 4090|graph_one_pressure128MiB|0.57262 [0.57211, 0.57302]|1.00109 [1.00056, 1.00160]|1.00775 [1.00718, 1.00826]|1.00665 [1.00592, 1.00735]|True|
|NVIDIA GeForce RTX 5090|graph_one_warm|0.56021 [0.55992, 0.56058]|1.01298 [1.01171, 1.01395]|1.02098 [1.02030, 1.02158]|1.00789 [1.00674, 1.00934]|True|
|NVIDIA GeForce RTX 5090|graph32_steady|0.55902 [0.55895, 0.55909]|1.01350 [1.01339, 1.01362]|1.02149 [1.02131, 1.02170]|1.00789 [1.00767, 1.00814]|True|
|NVIDIA GeForce RTX 5090|graph_one_pressure128MiB|0.56155 [0.56124, 0.56189]|1.01311 [1.01253, 1.01357]|1.02115 [1.02067, 1.02162]|1.00793 [1.00728, 1.00862]|True|

Shared-memory padding changes a resource-residency constraint and may also change cache/resource behavior. It does not directly measure CTA-to-SM issue order. Removal of a harmful reorder is not automatically a speedup over native48. D/E require a separate independently qualified remedy, pristine lifecycle A/B and supported callers.
