#!/bin/bash
# Live A/B of the env-gated deadline controller (installed in site-packages via apply_deadline_controller.py).
# Fresh server per run; conditions interleaved; 3 repeats. Deadline 100 ms for prefill-containing steps.
source ~/inference/vllm_env.sh
cd ~/inference/single-gpu-inference-lab
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
R=~/inference/results/campaign20-$(git rev-parse --short HEAD); mkdir -p $R $R/trace
M=~/inference/models/Qwen3-4B
run() { # name, env-assignments..., -- harness args
  local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 2
  env "${envs[@]}" python scripts/measure_prefill_interference.py --model $M --vllm-bin vllm --repeats 1 --trace-dir $R/trace --output $R/$name.json "$@" > $R/$name.log 2>&1
}
set -x
for rep in 1 2 3; do
  for N in 4 8; do
    W="--chunk-budgets 8192 --background 8 --background-tokens 4096 --inject $N --long-tokens 16384"
    run N$N-fixed128-r$rep VLLM_EXP_FIXED_BUDGET=128 -- $W
    run N$N-fixed256-r$rep VLLM_EXP_FIXED_BUDGET=256 -- $W
    run N$N-fixed512-r$rep VLLM_EXP_FIXED_BUDGET=512 -- $W
    run N$N-m0-r$rep VLLM_EXP_DEADLINE_MS=100 VLLM_EXP_COST_MODEL=$HOME/inference/m0-one.json -- $W
    run N$N-m2-r$rep VLLM_EXP_DEADLINE_MS=100 VLLM_EXP_COST_MODEL=$HOME/inference/m2-one.json -- $W
  done
done
echo CAMPAIGN20_DONE
