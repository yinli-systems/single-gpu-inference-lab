### qwen25-05b-native-ladder

Model `Qwen2.5-0.5B-Instruct`, 128 prompts x 256 tokens (ignore_eos), concurrency 64, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e median | e2e p99 | mask mean size | FlashInfer sampler |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `native` | `gen` | 18,613 (17,922–18,682) | 1.000x | 870 ms | 892 ms | — | yes |
| `native` | `logprobs` | 12,939 (12,641–13,298) | 0.695x | 1,215 ms | 1,248 ms | — | yes |
| `native` | `logprobs0` | 13,718 (13,663–13,841) | 0.737x | 1,175 ms | 1,208 ms | — | yes |
| `native` | `logprobs0_flat` | 13,490 (13,326–13,653) | 0.725x | 1,184 ms | 1,236 ms | — | yes |
| `native_flat` | `gen` | 18,555 (17,910–18,566) | 0.997x | 872 ms | 890 ms | — | yes |
| `native_flat` | `logprobs` | 13,338 (13,189–13,472) | 0.717x | 1,205 ms | 1,242 ms | — | yes |
| `native_flat` | `logprobs0` | 13,911 (13,691–13,944) | 0.747x | 1,155 ms | 1,191 ms | — | yes |
| `native_flat` | `logprobs0_flat` | 16,596 (16,042–16,656) | 0.892x | 973 ms | 995 ms | — | yes |

### qwen25-05b-mask-ladder

Model `Qwen2.5-0.5B-Instruct`, 128 prompts x 256 tokens (ignore_eos), concurrency 64, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e median | e2e p99 | mask mean size | FlashInfer sampler |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mask_upstream` | `gen` | 16,372 (15,299–16,472) | nanx | 982 ms | 1,005 ms | 15.3 | no |
| `mask_upstream` | `logprobs0_flat` | 11,756 (11,257–11,808) | nanx | 1,361 ms | 1,411 ms | 14.8 | no |
| `mask_upstream_flat` | `gen` | 15,991 (15,501–16,600) | nanx | 976 ms | 1,048 ms | 15.3 | no |
| `mask_upstream_flat` | `logprobs0_flat` | 14,045 (13,377–14,049) | nanx | 1,144 ms | 1,177 ms | 14.7 | no |

### qwen3-4b-mask-ladder

Model `Qwen3-4B`, 128 prompts x 256 tokens (ignore_eos), concurrency 64, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e median | e2e p99 | mask mean size | FlashInfer sampler |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mask_upstream` | `gen` | 3,486 (3,227–3,500) | nanx | 4,672 ms | 4,719 ms | 3.2 | no |
| `mask_upstream` | `logprobs` | 3,262 (3,239–3,286) | nanx | 4,966 ms | 5,027 ms | 3.2 | no |
| `mask_upstream` | `logprobs0_flat` | 3,308 (3,296–3,343) | nanx | 4,829 ms | 4,979 ms | 3.1 | no |
| `mask_upstream_flat` | `gen` | 3,468 (3,207–3,497) | nanx | 4,661 ms | 4,836 ms | 3.2 | no |
| `mask_upstream_flat` | `logprobs` | 3,301 (3,293–3,302) | nanx | 4,892 ms | 5,023 ms | 3.2 | no |
| `mask_upstream_flat` | `logprobs0_flat` | 3,446 (3,424–3,458) | nanx | 4,720 ms | 4,754 ms | 3.1 | no |

