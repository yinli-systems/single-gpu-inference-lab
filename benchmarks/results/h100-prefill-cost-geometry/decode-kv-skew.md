## H100-Qwen3-4B
  skew-bal-chunk512    decode 6.08/6.18 (thirds 6.08, 6.08, 6.08) | 512@8k 15.6/16.5 (n=12) | 512@12k 18.4/20.4 (n=12)
  skew-skew-chunk512   decode 6.46/6.57 (thirds 6.46, 6.46, 6.46) | 512@8k 16.5/16.7 (n=12) | 512@12k 19.2/20.3 (n=12)
  decode skewed/balanced p50: 1.063
## H100-Qwen3-8B
  skew-bal-chunk512    decode 8.72/8.79 (thirds 8.71, 8.72, 8.72) | 512@8k 21.0/22.3 (n=12) | 512@12k 23.5/25.2 (n=12)
  skew-skew-chunk512   decode 8.68/8.78 (thirds 8.68, 8.68, 8.68) | 512@8k 21.5/21.8 (n=12) | 512@12k 24.1/25.2 (n=12)
  decode skewed/balanced p50: 0.996
## H100-Qwen2.5-1.5B-Instruct
  skew-bal-chunk512    decode 2.82/2.88 (thirds 2.81, 2.82, 2.82) | 512@8k 8.6/8.9 (n=12) | 512@12k 8.6/8.7 (n=12)
  skew-skew-chunk512   decode 2.82/2.87 (thirds 2.82, 2.82, 2.81) | 512@8k 8.7/9.0 (n=12) | 512@12k 8.7/9.0 (n=12)
  decode skewed/balanced p50: 1.000
## H100-Qwen2.5-7B-Instruct
  skew-bal-chunk512    decode 7.09/7.15 (thirds 7.09, 7.08, 7.09) | 512@8k 18.0/18.8 (n=12) | 512@12k 19.1/19.2 (n=12)
  skew-skew-chunk512   decode 7.08/7.11 (thirds 7.08, 7.08, 7.08) | 512@8k 17.5/19.2 (n=12) | 512@12k 18.8/19.6 (n=12)
  decode skewed/balanced p50: 0.999
