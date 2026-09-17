### engine-skips-qwen25-05b-c256

Model `Qwen2.5-0.5B-Instruct`, 512 prompts x 256 tokens (ignore_eos), concurrency 256, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e p50 | e2e p99 | API CPU | core CPU | bytes/req | mask mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `native` | `gen` | 23,301 (21,740–23,387) | 1.000x | 2,057 ms | 4,091 ms | 0.33 | 0.98 | 1,632 | — |
| `native` | `token_logprobs` | 18,516 (18,021–19,102) | 0.795x | 2,496 ms | 4,978 ms | 0.70 | 1.01 | 6,821 | — |
| `native_skip_ranks` | `gen` | 22,708 (21,191–23,141) | 0.975x | 2,168 ms | 4,128 ms | 0.38 | 0.97 | 1,634 | — |
| `native_skip_ranks` | `token_logprobs` | 18,798 (18,718–19,358) | 0.807x | 2,507 ms | 5,019 ms | 0.72 | 0.99 | 6,824 | — |
| `native_skip_logprob_engine` | `gen` | 22,837 (21,458–23,025) | 0.980x | 2,070 ms | 4,174 ms | 0.36 | 0.99 | 1,634 | — |
| `native_skip_logprob_engine` | `token_logprobs` | 19,473 (19,231–19,535) | 0.836x | 2,472 ms | 4,945 ms | 0.71 | 1.00 | 2,650 | — |

### engine-skips-qwen25-05b-c64

Model `Qwen2.5-0.5B-Instruct`, 128 prompts x 256 tokens (ignore_eos), concurrency 64, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e p50 | e2e p99 | API CPU | core CPU | bytes/req | mask mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `native` | `gen` | 18,467 (17,599–18,488) | 1.000x | 870 ms | 906 ms | 0.38 | 0.94 | 1,636 | — |
| `native` | `token_logprobs` | 16,373 (16,367–16,502) | 0.887x | 985 ms | 1,002 ms | 0.60 | 0.99 | 6,833 | — |
| `native_skip_ranks` | `gen` | 18,635 (17,735–18,703) | 1.009x | 870 ms | 885 ms | 0.27 | 0.94 | 1,636 | — |
| `native_skip_ranks` | `token_logprobs` | 16,002 (14,607–16,501) | 0.866x | 1,002 ms | 1,054 ms | 0.58 | 0.99 | 6,833 | — |
| `native_skip_logprob_engine` | `gen` | 18,540 (17,743–18,578) | 1.004x | 872 ms | 900 ms | 0.25 | 0.95 | 1,636 | — |
| `native_skip_logprob_engine` | `token_logprobs` | 16,632 (16,593–16,679) | 0.901x | 972 ms | 992 ms | 0.62 | 0.99 | 2,650 | — |

### engine-skips-qwen3-4b-c64

Model `Qwen3-4B`, 128 prompts x 256 tokens (ignore_eos), concurrency 64, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e p50 | e2e p99 | API CPU | core CPU | bytes/req | mask mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `native` | `gen` | 3,527 (3,238–3,535) | 1.000x | 4,617 ms | 4,677 ms | 0.10 | 0.35 | 1,624 | — |
| `native` | `token_logprobs` | 3,491 (3,491–3,494) | 0.990x | 4,678 ms | 4,696 ms | 0.30 | 0.41 | 6,894 | — |
| `native_skip_ranks` | `gen` | 3,531 (3,263–3,551) | 1.001x | 4,612 ms | 4,662 ms | 0.12 | 0.37 | 1,626 | — |
| `native_skip_ranks` | `token_logprobs` | 3,493 (3,492–3,494) | 0.991x | 4,671 ms | 4,703 ms | 0.30 | 0.40 | 6,882 | — |
| `native_skip_logprob_engine` | `gen` | 3,523 (3,240–3,530) | 0.999x | 4,636 ms | 4,664 ms | 0.11 | 0.34 | 1,626 | — |
| `native_skip_logprob_engine` | `token_logprobs` | 3,527 (3,512–3,535) | 1.000x | 4,624 ms | 4,678 ms | 0.26 | 0.40 | 2,644 | — |

### skip-tokenizer-qwen25-05b-c256

Model `Qwen2.5-0.5B-Instruct`, 512 prompts x 256 tokens (ignore_eos), concurrency 256, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e p50 | e2e p99 | API CPU | core CPU | bytes/req | mask mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `native` | `gen` | 22,968 (22,208–23,381) | 1.000x | 2,061 ms | 4,111 ms | 0.36 | 0.97 | 1,635 | — |
| `native` | `token_logprobs` | 19,314 (18,981–19,615) | 0.841x | 2,504 ms | 4,994 ms | 0.73 | 1.00 | 6,831 | — |
| `native_skip_tokenizer` | `gen` | 23,337 (21,839–23,349) | 1.016x | 2,038 ms | 4,055 ms | 0.24 | 0.97 | 1,629 | — |
| `native_skip_tokenizer` | `token_logprobs` | 19,367 (18,977–19,837) | 0.843x | 2,484 ms | 4,987 ms | 0.39 | 1.01 | 6,818 | — |

### skip-tokenizer-qwen25-05b-c64

Model `Qwen2.5-0.5B-Instruct`, 128 prompts x 256 tokens (ignore_eos), concurrency 64, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e p50 | e2e p99 | API CPU | core CPU | bytes/req | mask mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `native` | `gen` | 18,477 (17,833–18,593) | 1.000x | 874 ms | 900 ms | 0.28 | 0.93 | 1,631 | — |
| `native` | `token_logprobs` | 16,665 (15,809–16,776) | 0.902x | 974 ms | 990 ms | 0.60 | 0.98 | 6,832 | — |
| `native_skip_tokenizer` | `gen` | 18,740 (17,871–18,754) | 1.014x | 866 ms | 882 ms | 0.37 | 0.94 | 1,632 | — |
| `native_skip_tokenizer` | `token_logprobs` | 16,402 (15,598–16,521) | 0.888x | 983 ms | 996 ms | 0.38 | 0.98 | 6,829 | — |

