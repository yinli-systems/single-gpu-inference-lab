#!/bin/bash
# Calibration round: finer budget grid for everyone, online per-bucket residual quantile margin,
# M0+online control, and the M2-fit-on-multi calibration oracle. Fresh server per run, interleaved, 3 repeats, D=100 ms.
source ~/inference/vllm_env.sh
cd ~/inference/single-gpu-inference-lab
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
R=~/inference/results/campaign21-$(git rev-parse --short HEAD); mkdir -p $R $R/trace
M=~/inference/models/Qwen3-4B
GRID=64,128,192,256,384,512,768,1024,1536,2048,3072,4096,8192
run() { local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 2
  env "${envs[@]}" python scripts/measure_prefill_interference.py --model $M --vllm-bin vllm --repeats 1 --trace-dir $R/trace --output $R/$name.json "$@" > $R/$name.log 2>&1; }
set -x
for rep in 1 2 3; do
  for N in 4 8; do
    W="--chunk-budgets 8192 --background 8 --background-tokens 4096 --inject $N --long-tokens 16384"
    run N$N-fixed256-r$rep VLLM_EXP_FIXED_BUDGET=256 -- $W
    run N$N-fixed384-r$rep VLLM_EXP_FIXED_BUDGET=384 -- $W
    run N$N-m2static-r$rep VLLM_EXP_DEADLINE_MS=100 VLLM_EXP_COST_MODEL=$HOME/inference/m2-one.json VLLM_EXP_CANDIDATES=$GRID -- $W
    run N$N-m2online-r$rep VLLM_EXP_DEADLINE_MS=100 VLLM_EXP_COST_MODEL=$HOME/inference/m2-one.json VLLM_EXP_CANDIDATES=$GRID VLLM_EXP_ONLINE_MARGIN=1 -- $W
    run N$N-m0online-r$rep VLLM_EXP_DEADLINE_MS=100 VLLM_EXP_COST_MODEL=$HOME/inference/m0-one.json VLLM_EXP_CANDIDATES=$GRID VLLM_EXP_ONLINE_MARGIN=1 -- $W
    run N$N-m2oracle-r$rep VLLM_EXP_DEADLINE_MS=100 VLLM_EXP_COST_MODEL=$HOME/inference/m2-multi.json VLLM_EXP_CANDIDATES=$GRID -- $W
  done
done
echo CAMPAIGN21_DONE
