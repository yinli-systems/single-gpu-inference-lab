### sweep-Qwen2.5-0.5B-Instruct-c1

Model `Qwen2.5-0.5B-Instruct`, 4 prompts x 256 tokens (ignore_eos), concurrency 1, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e p50 | e2e p99 | API CPU | core CPU | bytes/req | mask mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `native` | `gen` | 430 (430–430) | 1.000x | 595 ms | 596 ms | 0.09 | 0.93 | 1,636 | — |
| `native` | `logprobs0` | 407 (407–409) | 0.947x | 626 ms | 630 ms | 0.10 | 0.98 | 41,163 | — |
| `native` | `logprobs0_flat` | 408 (407–409) | 0.948x | 626 ms | 629 ms | 0.09 | 0.96 | 41,163 | — |
| `native` | `token_logprobs` | 411 (411–412) | 0.955x | 623 ms | 624 ms | 0.09 | 0.96 | 6,762 | — |
| `mask_upstream` | `gen` | 417 (417–418) | 0.969x | 614 ms | 614 ms | 0.06 | 0.96 | 18,877 | 13.2 |
| `mask_upstream` | `logprobs0` | 395 (395–395) | 0.919x | 647 ms | 647 ms | 0.08 | 0.98 | 63,268 | 17.9 |
| `mask_upstream` | `logprobs0_flat` | 395 (395–395) | 0.918x | 647 ms | 647 ms | 0.09 | 0.98 | 63,268 | 17.9 |
| `mask_upstream` | `token_logprobs` | 397 (397–397) | 0.923x | 644 ms | 644 ms | 0.08 | 0.98 | 29,480 | 17.9 |

### sweep-Qwen2.5-0.5B-Instruct-c256

Model `Qwen2.5-0.5B-Instruct`, 512 prompts x 256 tokens (ignore_eos), concurrency 256, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e p50 | e2e p99 | API CPU | core CPU | bytes/req | mask mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `native` | `gen` | 22,942 (21,677–23,151) | 1.000x | 2,160 ms | 4,090 ms | 0.37 | 0.96 | 1,635 | — |
| `native` | `logprobs0` | 15,322 (14,417–15,571) | 0.668x | 3,089 ms | 6,157 ms | 0.78 | 0.81 | 41,302 | — |
| `native` | `logprobs0_flat` | 15,438 (15,330–15,483) | 0.673x | 3,105 ms | 6,173 ms | 0.77 | 0.81 | 41,291 | — |
| `native` | `token_logprobs` | 19,168 (18,952–19,398) | 0.835x | 2,516 ms | 5,044 ms | 0.72 | 1.02 | 6,818 | — |
| `mask_upstream` | `gen` | 19,817 (19,356–20,422) | 0.864x | 2,579 ms | 5,029 ms | 0.37 | 1.01 | 20,546 | 14.7 |
| `mask_upstream` | `logprobs0` | 12,976 (12,612–13,948) | 0.566x | 3,836 ms | 7,600 ms | 0.77 | 0.84 | 58,200 | 14.7 |
| `mask_upstream` | `logprobs0_flat` | 12,934 (12,665–13,273) | 0.564x | 3,645 ms | 7,468 ms | 0.75 | 0.80 | 58,074 | 14.6 |
| `mask_upstream` | `token_logprobs` | 16,410 (15,938–16,619) | 0.715x | 3,166 ms | 6,253 ms | 0.73 | 1.00 | 25,035 | 15.0 |

### sweep-Qwen2.5-0.5B-Instruct-c512

Model `Qwen2.5-0.5B-Instruct`, 1024 prompts x 256 tokens (ignore_eos), concurrency 512, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e p50 | e2e p99 | API CPU | core CPU | bytes/req | mask mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `native` | `gen` | 23,945 (22,297–24,067) | 1.000x | 5,097 ms | 8,228 ms | 0.33 | 1.01 | 1,634 | — |
| `native` | `logprobs0` | 15,787 (15,703–15,864) | 0.659x | 7,811 ms | 12,623 ms | 0.79 | 0.82 | 41,317 | — |
| `native` | `logprobs0_flat` | 15,583 (15,334–15,642) | 0.651x | 7,691 ms | 12,731 ms | 0.77 | 0.83 | 41,307 | — |
| `native` | `token_logprobs` | 19,264 (18,856–19,907) | 0.805x | 6,339 ms | 10,228 ms | 0.71 | 1.02 | 6,831 | — |
| `mask_upstream` | `gen` | 20,730 (19,173–20,993) | 0.866x | 5,916 ms | 9,878 ms | 0.44 | 1.01 | 20,867 | 15.0 |
| `mask_upstream` | `logprobs0` | 13,728 (13,635–13,790) | 0.573x | 9,240 ms | 14,862 ms | 0.80 | 0.83 | 58,579 | 15.0 |
| `mask_upstream` | `logprobs0_flat` | 13,767 (13,595–14,252) | 0.575x | 9,079 ms | 14,841 ms | 0.78 | 0.84 | 58,315 | 14.8 |
| `mask_upstream` | `token_logprobs` | 16,984 (16,877–17,221) | 0.709x | 7,261 ms | 11,915 ms | 0.75 | 1.01 | 24,807 | 14.8 |

### sweep-Qwen2.5-0.5B-Instruct-c64

Model `Qwen2.5-0.5B-Instruct`, 128 prompts x 256 tokens (ignore_eos), concurrency 64, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e p50 | e2e p99 | API CPU | core CPU | bytes/req | mask mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `native` | `gen` | 18,645 (17,798–18,654) | 1.000x | 873 ms | 887 ms | 0.32 | 0.96 | 1,635 | — |
| `native` | `logprobs0` | 13,583 (11,910–13,722) | 0.728x | 1,186 ms | 1,220 ms | 0.66 | 0.84 | 41,318 | — |
| `native` | `logprobs0_flat` | 13,023 (12,808–13,369) | 0.698x | 1,201 ms | 1,250 ms | 0.64 | 0.83 | 41,323 | — |
| `native` | `token_logprobs` | 16,146 (16,113–16,247) | 0.866x | 1,004 ms | 1,027 ms | 0.58 | 0.98 | 6,832 | — |
| `mask_upstream` | `gen` | 16,103 (15,579–16,521) | 0.864x | 976 ms | 1,035 ms | 0.42 | 0.94 | 20,695 | 14.8 |
| `mask_upstream` | `logprobs0` | 11,652 (11,207–11,817) | 0.625x | 1,375 ms | 1,403 ms | 0.67 | 0.80 | 58,219 | 14.7 |
| `mask_upstream` | `logprobs0_flat` | 11,177 (11,021–11,812) | 0.599x | 1,370 ms | 1,514 ms | 0.65 | 0.80 | 58,477 | 14.9 |
| `mask_upstream` | `token_logprobs` | 13,686 (13,270–13,923) | 0.734x | 1,150 ms | 1,206 ms | 0.59 | 0.95 | 24,816 | 14.8 |

### sweep-Qwen3-4B-c1

Model `Qwen3-4B`, 4 prompts x 256 tokens (ignore_eos), concurrency 1, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e p50 | e2e p99 | API CPU | core CPU | bytes/req | mask mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `native` | `gen` | 82 (82–82) | 1.000x | 3,134 ms | 3,136 ms | 0.03 | 0.36 | 1,627 | — |
| `native` | `logprobs0` | 81 (81–81) | 0.989x | 3,166 ms | 3,169 ms | 0.05 | 0.34 | 41,544 | — |
| `native` | `logprobs0_flat` | 81 (81–81) | 0.989x | 3,167 ms | 3,171 ms | 0.04 | 0.34 | 41,544 | — |
| `native` | `token_logprobs` | 81 (81–81) | 0.991x | 3,160 ms | 3,163 ms | 0.05 | 0.36 | 6,943 | — |
| `mask_upstream` | `gen` | 81 (81–81) | 0.994x | 3,153 ms | 3,154 ms | 0.04 | 0.34 | 6,456 | 3.5 |
| `mask_upstream` | `logprobs0` | 80 (80–80) | 0.983x | 3,185 ms | 3,190 ms | 0.05 | 0.31 | 42,430 | 3.4 |
| `mask_upstream` | `logprobs0_flat` | 80 (80–80) | 0.982x | 3,189 ms | 3,193 ms | 0.05 | 0.36 | 42,430 | 3.4 |
| `mask_upstream` | `token_logprobs` | 80 (80–80) | 0.985x | 3,181 ms | 3,184 ms | 0.04 | 0.31 | 9,739 | 3.4 |

### sweep-Qwen3-4B-c256

Model `Qwen3-4B`, 512 prompts x 256 tokens (ignore_eos), concurrency 256, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e p50 | e2e p99 | API CPU | core CPU | bytes/req | mask mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `native` | `gen` | 4,324 (3,913–4,336) | 1.000x | 10,773 ms | 21,518 ms | 0.13 | 0.38 | 1,624 | — |
| `native` | `logprobs0` | 4,236 (4,071–4,241) | 0.980x | 11,463 ms | 22,854 ms | 0.36 | 0.39 | 41,413 | — |
| `native` | `logprobs0_flat` | 4,213 (4,046–4,241) | 0.974x | 11,360 ms | 22,942 ms | 0.39 | 0.38 | 41,414 | — |
| `native` | `token_logprobs` | 4,274 (4,231–4,286) | 0.988x | 10,834 ms | 21,863 ms | 0.33 | 0.43 | 6,882 | — |
| `mask_upstream` | `gen` | 4,285 (3,907–4,293) | 0.991x | 10,952 ms | 22,048 ms | 0.17 | 0.42 | 6,078 | 3.2 |
| `mask_upstream` | `logprobs0` | 4,181 (4,166–4,202) | 0.967x | 11,766 ms | 23,488 ms | 0.39 | 0.43 | 42,245 | 3.1 |
| `mask_upstream` | `logprobs0_flat` | 4,015 (4,009–4,186) | 0.928x | 11,621 ms | 23,328 ms | 0.33 | 0.43 | 42,233 | 3.2 |
| `mask_upstream` | `token_logprobs` | 4,240 (4,192–4,265) | 0.980x | 11,144 ms | 22,197 ms | 0.30 | 0.44 | 9,531 | 3.2 |

### sweep-Qwen3-4B-c512

Model `Qwen3-4B`, 1024 prompts x 256 tokens (ignore_eos), concurrency 512, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e p50 | e2e p99 | API CPU | core CPU | bytes/req | mask mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `native` | `gen` | 4,130 (4,090–4,133) | 1.000x | 29,530 ms | 47,492 ms | 0.12 | 0.33 | 1,626 | — |
| `native` | `logprobs0` | 3,922 (3,920–3,933) | 0.950x | 30,967 ms | 50,003 ms | 0.37 | 0.40 | 41,408 | — |
| `native` | `logprobs0_flat` | 3,917 (3,903–3,919) | 0.948x | 31,026 ms | 50,140 ms | 0.36 | 0.39 | 41,410 | — |
| `native` | `token_logprobs` | 4,096 (4,079–4,103) | 0.992x | 29,695 ms | 47,841 ms | 0.33 | 0.40 | 6,875 | — |
| `mask_upstream` | `gen` | 4,076 (4,059–4,103) | 0.987x | 29,730 ms | 48,181 ms | 0.16 | 0.38 | 6,077 | 3.2 |
| `mask_upstream` | `logprobs0` | 3,877 (3,864–3,877) | 0.939x | 31,305 ms | 50,663 ms | 0.39 | 0.41 | 42,313 | 3.2 |
| `mask_upstream` | `logprobs0_flat` | 3,867 (3,859–3,868) | 0.936x | 31,408 ms | 50,827 ms | 0.38 | 0.42 | 42,289 | 3.2 |
| `mask_upstream` | `token_logprobs` | 4,059 (4,041–4,060) | 0.983x | 30,019 ms | 48,462 ms | 0.34 | 0.43 | 9,492 | 3.1 |

### sweep-Qwen3-4B-c64

Model `Qwen3-4B`, 128 prompts x 256 tokens (ignore_eos), concurrency 64, API `generate`, sampling {'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}, 3 interleaved rounds.

| Server | Request | tok/s (median, min–max) | vs native/gen | e2e p50 | e2e p99 | API CPU | core CPU | bytes/req | mask mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `native` | `gen` | 3,517 (3,239–3,538) | 1.000x | 4,644 ms | 4,667 ms | 0.10 | 0.38 | 1,625 | — |
| `native` | `logprobs0` | 3,361 (3,349–3,375) | 0.956x | 4,833 ms | 4,913 ms | 0.34 | 0.43 | 41,433 | — |
| `native` | `logprobs0_flat` | 3,342 (3,337–3,358) | 0.950x | 4,850 ms | 4,923 ms | 0.33 | 0.39 | 41,388 | — |
| `native` | `token_logprobs` | 3,498 (3,493–3,515) | 0.994x | 4,657 ms | 4,712 ms | 0.31 | 0.37 | 6,894 | — |
| `mask_upstream` | `gen` | 3,479 (3,216–3,481) | 0.989x | 4,688 ms | 4,724 ms | 0.14 | 0.35 | 6,089 | 3.2 |
| `mask_upstream` | `logprobs0` | 3,287 (3,270–3,312) | 0.935x | 4,888 ms | 5,104 ms | 0.34 | 0.39 | 42,207 | 3.1 |
| `mask_upstream` | `logprobs0_flat` | 3,330 (3,305–3,334) | 0.947x | 4,850 ms | 4,977 ms | 0.35 | 0.35 | 42,203 | 3.2 |
| `mask_upstream` | `token_logprobs` | 3,419 (3,418–3,444) | 0.972x | 4,738 ms | 4,811 ms | 0.30 | 0.39 | 9,472 | 3.2 |

