#!/bin/bash
source ~/inference/vllm_env.sh
cd ~/inference/single-gpu-inference-lab
# harness file shipped by scp = commit bf372ad (GitHub unreachable from L20 at run time)
R=~/inference/results/campaign28-bf372ad; mkdir -p $R
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 2
python scripts/measure_kv_prefetch.py --model ~/inference/models/Qwen3-4B --vllm-bin vllm --output $R/contention-bg8.json --contention-mode --background 8 --background-tokens 512 --repeats 5 --seed 31 > $R/contention-bg8.log 2>&1
echo CAMPAIGN28_DONE
