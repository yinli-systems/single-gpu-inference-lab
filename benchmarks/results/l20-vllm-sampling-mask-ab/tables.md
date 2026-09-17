### qwen25-05b-generate

Model `Qwen2.5-0.5B-Instruct`, 128 prompts x 256 tokens (ignore_eos), concurrency 64, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e median | e2e p99 | mask mean size | FlashInfer sampler |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `native` | `gen` | 18,566 (17,886–18,684) | 1.000x | 865 ms | 897 ms | — | yes |
| `native` | `logprobs` | 13,403 (12,846–13,458) | 0.722x | 1,203 ms | 1,235 ms | — | yes |
| `mask_bitmap` | `gen` | 1,582 (1,578–1,586) | 0.085x | 10,252 ms | 10,395 ms | 14.7 | no |
| `mask_bitmap` | `logprobs` | 1,497 (1,486–1,509) | 0.081x | 10,838 ms | 11,050 ms | 15.0 | no |
| `mask_compact` | `gen` | 14,323 (14,120–14,649) | 0.771x | 1,098 ms | 1,158 ms | 14.7 | no |
| `mask_compact` | `logprobs` | 10,076 (9,928–10,379) | 0.543x | 1,561 ms | 1,672 ms | 14.7 | no |

### qwen3-4b-generate

Model `Qwen3-4B`, 128 prompts x 256 tokens (ignore_eos), concurrency 64, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e median | e2e p99 | mask mean size | FlashInfer sampler |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `native` | `gen` | 3,538 (3,250–3,540) | 1.000x | 4,608 ms | 4,660 ms | — | yes |
| `native` | `logprobs` | 3,338 (3,323–3,345) | 0.944x | 4,859 ms | 4,948 ms | — | yes |
| `mask_bitmap` | `gen` | 1,583 (1,526–1,585) | 0.447x | 10,282 ms | 10,415 ms | 3.2 | no |
| `mask_bitmap` | `logprobs` | 1,484 (1,460–1,493) | 0.419x | 10,906 ms | 11,170 ms | 3.2 | no |
| `mask_compact` | `gen` | 3,418 (3,177–3,476) | 0.966x | 4,680 ms | 4,898 ms | 3.1 | no |
| `mask_compact` | `logprobs` | 3,238 (3,211–3,266) | 0.915x | 4,969 ms | 5,102 ms | 3.2 | no |

### qwen25-05b-completions

Model `Qwen2.5-0.5B-Instruct`, 128 prompts x 256 tokens (ignore_eos), concurrency 64, API `completions`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | ITL median | ITL mean | tokens/chunk | e2e median | FlashInfer sampler |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `native` | `gen` | 17,954 (17,425–18,118) | 1.000x | 3.21 ms | 3.27 ms | 1.00 | 904 ms | yes |
| `native` | `logprobs` | 14,948 (14,714–14,971) | 0.833x | 34.74 ms | 33.93 ms | 8.37 | 1,088 ms | yes |
| `mask_bitmap` | `gen` | 1,591 (1,588–1,594) | 0.089x | 39.61 ms | 39.55 ms | 1.00 | 10,254 ms | no |
| `mask_bitmap` | `logprobs` | 1,534 (1,530–1,541) | 0.085x | 40.95 ms | 41.08 ms | 1.00 | 10,661 ms | no |
| `mask_compact` | `gen` | 14,882 (14,698–14,922) | 0.829x | 3.56 ms | 3.97 ms | 1.01 | 1,087 ms | no |
| `mask_compact` | `logprobs` | 12,117 (12,087–12,135) | 0.675x | 26.07 ms | 28.02 ms | 5.54 | 1,344 ms | no |

### qwen25-05b-decomposition

Model `Qwen2.5-0.5B-Instruct`, 128 prompts x 256 tokens (ignore_eos), concurrency 64, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e median | e2e p99 | mask mean size | FlashInfer sampler |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `native` | `gen` | 18,360 (17,764–18,495) | 1.000x | 881 ms | 904 ms | — | yes |
| `native` | `logprobs` | 13,128 (13,052–13,174) | 0.715x | 1,225 ms | 1,255 ms | — | yes |
| `native_fi_off` | `gen` | 18,599 (17,751–18,611) | 1.013x | 868 ms | 895 ms | — | no |
| `native_fi_off` | `logprobs` | 13,311 (13,208–13,315) | 0.725x | 1,210 ms | 1,246 ms | — | no |
| `native_processed` | `gen` | 18,656 (17,875–18,664) | 1.016x | 870 ms | 888 ms | — | yes |
| `native_processed` | `logprobs` | 13,348 (13,347–13,493) | 0.727x | 1,210 ms | 1,235 ms | — | yes |

### qwen25-05b-batch-invariant

Model `Qwen2.5-0.5B-Instruct`, 128 prompts x 256 tokens (ignore_eos), concurrency 64, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 1 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e median | e2e p99 | mask mean size | FlashInfer sampler |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mask_bitmap` | `gen` | 1,329 (1,329–1,329) | nanx | 12,251 ms | 14,203 ms | 14.8 | no |
| `mask_bitmap` | `logprobs` | 1,486 (1,486–1,486) | nanx | 10,958 ms | 11,075 ms | 14.8 | no |
| `mask_compact` | `gen` | 7,795 (7,795–7,795) | nanx | 2,084 ms | 2,124 ms | 14.8 | no |
| `mask_compact` | `logprobs` | 6,640 (6,640–6,640) | nanx | 2,374 ms | 2,532 ms | 14.8 | no |

