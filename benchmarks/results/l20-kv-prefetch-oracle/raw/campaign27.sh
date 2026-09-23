#!/bin/bash
source ~/inference/vllm_env.sh
cd ~/inference/single-gpu-inference-lab
git fetch -q origin kv-prefetch && git checkout -q -- . && git clean -fdq && git checkout -q -B kv-prefetch origin/kv-prefetch
R=~/inference/results/campaign27-$(git rev-parse --short HEAD); mkdir -p $R
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 2
python scripts/measure_kv_prefetch.py --model ~/inference/models/Qwen3-4B --vllm-bin vllm --output $R/policies.json --policy-mode --repeats 6 --seed 23 > $R/policies.log 2>&1
echo CAMPAIGN27_DONE
