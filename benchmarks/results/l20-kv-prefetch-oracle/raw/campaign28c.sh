#!/bin/bash
# harness = commit cb62e1e (scp). Occupancy sweep: paused-session KV as a fraction of free GPU capacity (~19k tokens with 8 decoders).
source ~/inference/vllm_env.sh
cd ~/inference/single-gpu-inference-lab
R=~/inference/results/campaign28c-cb62e1e; mkdir -p $R
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 2
python scripts/measure_kv_prefetch.py --model ~/inference/models/Qwen3-4B --vllm-bin vllm --output $R/occ60.json --contention-mode --background 8 --background-tokens 512 --repeats 5 --seed 41 --session-sizes 4096,4096,4096 --cap-tokens 12288 --filler-tokens 0 --contention-policies reactive,immediate-all,oracle-admit > $R/occ60.log 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 3
python scripts/measure_kv_prefetch.py --model ~/inference/models/Qwen3-4B --vllm-bin vllm --output $R/occ120.json --contention-mode --background 8 --background-tokens 512 --repeats 5 --seed 43 --session-sizes 4096,4096,8192,8192 --cap-tokens 12288 --filler-tokens 0 --contention-policies reactive,immediate-all,oracle-admit,smallest-admit > $R/occ120.log 2>&1
echo CAMPAIGN28C_DONE
