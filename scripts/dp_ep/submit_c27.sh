#!/bin/bash
# Submit the task-27 (spec-decode acceptance skew) job: n-gram speculation, K=4, graph mode, multiport LB, q=512.
cd /data/run01/scxi253/inference
sbatch lab-scripts/sbatch_c27.sh Qwen1.5-MoE-A2.7B-Chat graph multiport 512 '--speculative-config {"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
