#!/bin/bash
# Campaign26: oracle request-free KV prefetch upper bound. (a) no background; (b) 16 background decoders.
source ~/inference/vllm_env.sh
cd ~/inference/single-gpu-inference-lab
git fetch -q origin kv-prefetch && git checkout -q -- . && git clean -fdq && git checkout -q -B kv-prefetch origin/kv-prefetch
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
R=~/inference/results/campaign26-$(git rev-parse --short HEAD); mkdir -p $R
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 2
python scripts/measure_kv_prefetch.py --model ~/inference/models/Qwen3-4B --vllm-bin vllm --output $R/nobg.json --repeats 3 > $R/nobg.log 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 3
python scripts/measure_kv_prefetch.py --model ~/inference/models/Qwen3-4B --vllm-bin vllm --output $R/bg16.json --repeats 3 --background 16 > $R/bg16.log 2>&1
echo CAMPAIGN26_DONE
