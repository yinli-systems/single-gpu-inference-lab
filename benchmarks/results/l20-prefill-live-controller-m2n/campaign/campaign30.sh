#!/bin/bash
# L20 live controller with the pre-registered M2n arm (addendum 4). Protocol of campaign20; refuses a
# busy GPU or a dirty tree; never kills anything.
source ~/inference/vllm_env.sh
export VLLM_NO_USAGE_STATS=1
cd ~/inference/single-gpu-inference-lab
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
R=~/inference/results/campaign30-$(git rev-parse --short HEAD); mkdir -p $R/trace
M=~/inference/models/Qwen3-4B; CM=~/inference/ctl-models-l20; D=100
cp $CM/*.json $R/; nvidia-smi > $R/nvidia-smi.txt
run() { local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -gt 500 ] && { echo "GPU busy (${used} MiB) before $name - stopping"; exit 2; }
  env "${envs[@]}" python scripts/measure_prefill_interference.py --model $M --vllm-bin vllm --repeats 1 --trace-dir $R/trace --output $R/$name.json "$@" > $R/$name.log 2>&1
  local rc=$?; echo "$(date +%H:%M:%S) $name exit=$rc"; [ $rc -ne 0 ] && { echo "RUN_FAILED $name"; exit 3; }; }
for rep in 1 2 3; do
  for N in 4 8; do
    W="--chunk-budgets 8192 --background 8 --background-tokens 4096 --inject $N --long-tokens 16384"
    run N$N-fixed256-r$rep VLLM_EXP_FIXED_BUDGET=256 -- $W
    run N$N-fixed512-r$rep VLLM_EXP_FIXED_BUDGET=512 -- $W
    run N$N-m0-r$rep VLLM_EXP_DEADLINE_MS=$D VLLM_EXP_COST_MODEL=$CM/m0-one.json -- $W
    run N$N-m2-r$rep VLLM_EXP_DEADLINE_MS=$D VLLM_EXP_COST_MODEL=$CM/m2-one.json -- $W
    run N$N-m2n-r$rep VLLM_EXP_DEADLINE_MS=$D VLLM_EXP_COST_MODEL=$CM/m2n-one.json -- $W
  done
done
echo CAMPAIGN30_DONE $R
