#!/bin/bash
# 27A: prediction-error frontier. leads = L* - eps for eps in {-250,-100,-50,0,+50,+100,+250}, L* = 50 (4k) / 100 (8k); lead<0 -> reactive.
source ~/inference/vllm_env.sh
cd ~/inference/single-gpu-inference-lab
R=~/inference/results/campaign27a-68f13c7; mkdir -p $R
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 2
python scripts/measure_kv_prefetch.py --model ~/inference/models/Qwen3-4B --vllm-bin vllm --output $R/nobg.json --prefixes 4096,8192 --leads-ms 0,50,100,150,200,300,350 --repeats 5 --seed 51 > $R/nobg.log 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 3
python scripts/measure_kv_prefetch.py --model ~/inference/models/Qwen3-4B --vllm-bin vllm --output $R/bg8.json --prefixes 4096 --leads-ms 0,50,100,150,200,300 --repeats 5 --seed 53 --background 8 --background-tokens 512 > $R/bg8.log 2>&1
echo CAMPAIGN27A_DONE
