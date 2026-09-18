#!/bin/bash
# harness file shipped by scp = commit cb62e1e
source ~/inference/vllm_env.sh
cd ~/inference/single-gpu-inference-lab
R=~/inference/results/campaign28b-cb62e1e; mkdir -p $R
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 2
python scripts/measure_kv_prefetch.py --model ~/inference/models/Qwen3-4B --vllm-bin vllm --output $R/contention-bg8.json --contention-mode --background 8 --background-tokens 512 --repeats 6 --seed 37 --session-sizes 4096,4096,4096,4096,8192,8192,8192 --cap-tokens 12288 --filler-tokens 0 > $R/contention-bg8.log 2>&1
echo CAMPAIGN28B_DONE
