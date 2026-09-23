## L20-Qwen3-4B
  skew-bal-chunk512    decode 19.58/19.70 (thirds 19.58, 19.58, 19.58) | 512@8k 90.8/99.3 (n=12) | 512@12k 104.0/109.0 (n=12)
  skew-skew-chunk512   decode 19.48/19.62 (thirds 19.47, 19.47, 19.49) | 512@8k 91.0/96.0 (n=12) | 512@12k 104.5/108.9 (n=12)
  decode skewed/balanced p50: 0.995
## A100-Qwen3-4B
  skew-bal-chunk512    decode 10.38/10.47 (thirds 10.37, 10.38, 10.38) | 512@8k 49.9/54.4 (n=12) | 512@12k 56.1/59.5 (n=12)
  skew-skew-chunk512   decode 11.27/11.33 (thirds 11.26, 11.26, 11.27) | 512@8k 47.1/51.6 (n=12) | 512@12k 55.0/57.3 (n=12)
  decode skewed/balanced p50: 1.086
## A100-Qwen3-8B
  skew-bal-chunk512    decode 14.32/14.39 (thirds 14.31, 14.32, 14.32) | 512@8k 64.2/68.7 (n=12) | 512@12k 70.1/73.4 (n=12)
  skew-skew-chunk512   decode 15.09/15.15 (thirds 15.08, 15.09, 15.10) | 512@8k 61.2/65.6 (n=12) | 512@12k 68.8/71.1 (n=12)
  decode skewed/balanced p50: 1.054
