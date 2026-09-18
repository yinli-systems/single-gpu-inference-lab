### qwen25-05b-c64

Model `Qwen2.5-0.5B-Instruct`, 128 prompts x 256 tokens (ignore_eos), concurrency 64, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e median | e2e p99 | mask mean size | FlashInfer sampler |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mask_upstream` | `gen` | 16,450 (14,872–16,647) | 0.925x | 975 ms | 1,009 ms | 15.0 | no |
| `mask_nobitmap` | `gen` | 16,398 (14,449–16,503) | 0.922x | 968 ms | 1,031 ms | 15.1 | no |
| `native` | `gen` | 17,784 (17,514–18,668) | 1.000x | 861 ms | 893 ms | — | yes |

### qwen25-05b-c256

Model `Qwen2.5-0.5B-Instruct`, 512 prompts x 256 tokens (ignore_eos), concurrency 256, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e median | e2e p99 | mask mean size | FlashInfer sampler |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mask_upstream` | `gen` | 19,767 (18,556–20,203) | 0.865x | 2,681 ms | 5,185 ms | 15.2 | no |
| `mask_nobitmap` | `gen` | 19,641 (19,494–20,739) | 0.860x | 2,647 ms | 5,192 ms | 15.2 | no |
| `native` | `gen` | 22,847 (21,759–23,260) | 1.000x | 2,070 ms | 4,130 ms | — | yes |

### qwen3-4b-c64

Model `Qwen3-4B`, 128 prompts x 256 tokens (ignore_eos), concurrency 64, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e median | e2e p99 | mask mean size | FlashInfer sampler |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mask_upstream` | `gen` | 3,511 (3,220–3,521) | 1.009x | 4,623 ms | 4,714 ms | 3.2 | no |
| `mask_nobitmap` | `gen` | 3,491 (3,222–3,516) | 1.003x | 4,676 ms | 4,714 ms | 3.1 | no |
| `native` | `gen` | 3,480 (3,241–3,524) | 1.000x | 4,643 ms | 4,667 ms | — | yes |

### qwen3-4b-c256

Model `Qwen3-4B`, 512 prompts x 256 tokens (ignore_eos), concurrency 256, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e median | e2e p99 | mask mean size | FlashInfer sampler |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mask_upstream` | `gen` | 4,278 (3,899–4,285) | 1.000x | 10,977 ms | 22,060 ms | 3.2 | no |
| `mask_nobitmap` | `gen` | 4,305 (3,890–4,308) | 1.007x | 10,865 ms | 21,926 ms | 3.2 | no |
| `native` | `gen` | 4,277 (3,918–4,342) | 1.000x | 10,860 ms | 21,669 ms | — | yes |

