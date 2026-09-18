#!/bin/bash
# Default-FCFS short-prompt bursts: 16 x 2048 and 32 x 1024 prompts arriving together into 8 decoders.
# Fixed budgets are native vLLM (--max-num-batched-tokens, no patch active); controllers price the
# FCFS partition (VLLM_EXP_PARTITION=fcfs). Fresh server per run, interleaved, 3 repeats, D=100 ms.
source ~/inference/vllm_env.sh
cd ~/inference/single-gpu-inference-lab
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
R=~/inference/results/campaign22-$(git rev-parse --short HEAD); mkdir -p $R $R/trace
M=~/inference/models/Qwen3-4B
GRID=64,128,192,256,384,512,768,1024,1536,2048,3072,4096,8192
run() { local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 2
  env "${envs[@]}" python scripts/measure_prefill_interference.py --model $M --vllm-bin vllm --repeats 1 --trace-dir $R/trace --output $R/$name.json "$@" > $R/$name.log 2>&1; }
set -x
for rep in 1 2 3; do
  for W in "16x2048" "32x1024"; do
    N=${W%x*}; P=${W#*x}
    B="--background 8 --background-tokens 4096 --inject $N --long-tokens $P"
    for C in 256 512 1024; do run B$W-fixed$C-r$rep -- --chunk-budgets $C $B; done
    run B$W-m2static-r$rep VLLM_EXP_DEADLINE_MS=100 VLLM_EXP_COST_MODEL=$HOME/inference/m2-one.json VLLM_EXP_CANDIDATES=$GRID VLLM_EXP_PARTITION=fcfs -- --chunk-budgets 8192 $B
    run B$W-m2online-r$rep VLLM_EXP_DEADLINE_MS=100 VLLM_EXP_COST_MODEL=$HOME/inference/m2-one.json VLLM_EXP_CANDIDATES=$GRID VLLM_EXP_PARTITION=fcfs VLLM_EXP_ONLINE_MARGIN=1 -- --chunk-budgets 8192 $B
    run B$W-m0online-r$rep VLLM_EXP_DEADLINE_MS=100 VLLM_EXP_COST_MODEL=$HOME/inference/m0-one.json VLLM_EXP_CANDIDATES=$GRID VLLM_EXP_PARTITION=fcfs VLLM_EXP_ONLINE_MARGIN=1 -- --chunk-budgets 8192 $B
  done
done
echo CAMPAIGN22_DONE
