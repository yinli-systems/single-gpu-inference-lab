### qwen25-05b-skip-tokenizer

Model `Qwen2.5-0.5B-Instruct`, 128 prompts x 256 tokens (ignore_eos), concurrency 64, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e median | e2e p99 | mask mean size | FlashInfer sampler |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `native` | `gen` | 18,544 (17,745–18,560) | 1.000x | 874 ms | 894 ms | — | yes |
| `native` | `logprobs` | 13,188 (12,819–13,203) | 0.711x | 1,223 ms | 1,255 ms | — | yes |
| `native_skip_tokenizer` | `gen` | 18,612 (17,959–18,618) | 1.004x | 869 ms | 890 ms | — | yes |
| `native_skip_tokenizer` | `logprobs` | 13,379 (12,823–13,499) | 0.721x | 1,185 ms | 1,242 ms | — | yes |
| `mask_upstream` | `gen` | 15,532 (15,280–16,418) | 0.838x | 992 ms | 1,103 ms | 15.2 | no |
| `mask_upstream` | `logprobs` | 11,301 (10,880–11,462) | 0.609x | 1,403 ms | 1,457 ms | 14.9 | no |
| `mask_upstream_skip_tokenizer` | `gen` | 16,272 (15,108–16,412) | 0.877x | 988 ms | 1,020 ms | 14.8 | no |
| `mask_upstream_skip_tokenizer` | `logprobs` | 11,253 (10,828–11,370) | 0.607x | 1,420 ms | 1,457 ms | 14.6 | no |

