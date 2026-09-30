# Plan-aware resource selector v2

This campaign was frozen after the first release guard failed worst-case safety. It does not rename that exposed release as holdout. The selector keeps the original long-KV/query-skew gate but additionally requires an unsplit actual plan whose grid has more blocks than the native two-CTA-per-SM capacity: `padded_batch_size * num_kv_heads > 2 * num_sms`. SM count is queried from the allocated CUDA device during planning.

The 24 release cases are new and densely cover descriptor counts 31–34 around the RTX 4090 boundary, 41–44 around the RTX 5090 boundary, far-above points, balanced and 8191/8192-KV controls. FP16/BF16, ragged/paged, auto/unsplit, eager qualification and 1/16-call CUDA Graph timing remain crossed. The primary gate is Graph16 run-device: every numerically qualified, resolved selected cell must be >=0.99; selection must have positive net aggregate benefit. No serving performance run is allowed before both GPU families pass.
