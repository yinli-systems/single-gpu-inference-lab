### pr2-qwen25-05b-c256

Model `Qwen2.5-0.5B-Instruct`, 512 prompts x 256 tokens (ignore_eos), concurrency 256, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e p50 | e2e p99 | API CPU | core CPU | bytes/req | mask mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `native` | `gen` | 23,038 (21,515–23,285) | 1.000x | 2,055 ms | 4,086 ms | 0.37 | 0.97 | 1,634 | — |
| `native` | `logprobs0` | 15,451 (14,997–15,529) | 0.671x | 3,105 ms | 6,212 ms | 0.78 | 0.87 | 41,310 | — |
| `native` | `token_logprobs` | 21,846 (21,283–22,069) | 0.948x | 2,202 ms | 4,362 ms | 0.39 | 0.99 | 6,831 | — |
| `mask_upstream` | `gen` | 20,313 (19,551–20,329) | 0.882x | 2,492 ms | 5,072 ms | 0.43 | 0.99 | 20,948 | 15.1 |
| `mask_upstream` | `logprobs0` | 13,088 (12,694–13,830) | 0.568x | 3,810 ms | 7,536 ms | 0.78 | 0.86 | 58,718 | 15.1 |
| `mask_upstream` | `token_logprobs` | 19,132 (18,773–19,144) | 0.830x | 2,691 ms | 5,291 ms | 0.43 | 1.00 | 25,304 | 15.2 |

### pr2-qwen25-05b-c64

Model `Qwen2.5-0.5B-Instruct`, 128 prompts x 256 tokens (ignore_eos), concurrency 64, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e p50 | e2e p99 | API CPU | core CPU | bytes/req | mask mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `native` | `gen` | 18,592 (17,862–18,639) | 1.000x | 866 ms | 895 ms | 0.23 | 0.92 | 1,630 | — |
| `native` | `logprobs0` | 13,495 (13,164–13,991) | 0.726x | 1,159 ms | 1,258 ms | 0.66 | 0.84 | 41,346 | — |
| `native` | `token_logprobs` | 17,868 (16,655–17,927) | 0.961x | 907 ms | 926 ms | 0.35 | 0.95 | 6,832 | — |
| `mask_upstream` | `gen` | 16,452 (15,800–16,658) | 0.885x | 969 ms | 1,016 ms | 0.31 | 0.94 | 20,489 | 14.7 |
| `mask_upstream` | `logprobs0` | 11,106 (11,001–11,315) | 0.597x | 1,381 ms | 1,513 ms | 0.65 | 0.80 | 58,510 | 14.9 |
| `mask_upstream` | `token_logprobs` | 15,453 (15,310–15,456) | 0.831x | 1,038 ms | 1,091 ms | 0.36 | 0.93 | 25,334 | 15.1 |

### pr2-qwen3-4b-c64

Model `Qwen3-4B`, 128 prompts x 256 tokens (ignore_eos), concurrency 64, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e p50 | e2e p99 | API CPU | core CPU | bytes/req | mask mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `native` | `gen` | 3,524 (3,237–3,535) | 1.000x | 4,638 ms | 4,663 ms | 0.12 | 0.37 | 1,626 | — |
| `native` | `logprobs0` | 3,351 (3,348–3,370) | 0.951x | 4,844 ms | 4,919 ms | 0.30 | 0.44 | 41,395 | — |
| `native` | `token_logprobs` | 3,503 (3,489–3,515) | 0.994x | 4,646 ms | 4,707 ms | 0.12 | 0.39 | 6,893 | — |
| `mask_upstream` | `gen` | 3,442 (3,213–3,501) | 0.977x | 4,717 ms | 4,842 ms | 0.13 | 0.36 | 6,045 | 3.1 |
| `mask_upstream` | `logprobs0` | 3,320 (3,311–3,332) | 0.942x | 4,876 ms | 4,977 ms | 0.35 | 0.44 | 42,206 | 3.2 |
| `mask_upstream` | `token_logprobs` | 3,443 (3,436–3,477) | 0.977x | 4,658 ms | 4,764 ms | 0.15 | 0.42 | 9,466 | 3.1 |

