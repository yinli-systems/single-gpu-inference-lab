#!/bin/bash
# A100 replication of l20-prefill-cost-geometry section 5 (live env-gated deadline controller), per
# docs/preregistration/2026-09-23-m2-without-aggregate-prefill-kv.md addendum 2. GPU 0 only, port 8124,
# controller installed in a copy of the venv (venv-ctl); refuses a busy GPU; never kills anything.
set -u
export CUDA_VISIBLE_DEVICES=0 VLLM_NO_USAGE_STATS=1 HF_HOME=/root/lab/hf HF_HUB_OFFLINE=1
export PATH=/root/lab/venv-ctl/bin:$PATH
cd /root/lab/lab-ctl
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
R=/root/lab/results/a100-live-Qwen3-4B-$(git rev-parse --short HEAD); mkdir -p $R/trace
M=/root/lab/models/Qwen3-4B; CM=/root/lab/ctl-models; D=65
cp $CM/*.json $R/; nvidia-smi > $R/nvidia-smi.txt
run() { local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local used=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -gt 500 ] && { echo "GPU 0 busy (${used} MiB) before $name - stopping"; exit 2; }
  env "${envs[@]}" python scripts/measure_prefill_interference.py --model $M --vllm-bin /root/lab/venv-ctl/bin/vllm --port 8124 --repeats 1 --trace-dir $R/trace --output $R/$name.json "$@" > $R/$name.log 2>&1
  local rc=$?
  echo "$(date +%H:%M:%S) $name exit=$rc"
  [ $rc -ne 0 ] && { echo "RUN_FAILED $name"; exit 3; }; }
for rep in 1 2 3; do
  for N in 4 8; do
    W="--chunk-budgets 8192 --background 8 --background-tokens 4096 --inject $N --long-tokens 16384"
    run N$N-fixed128-r$rep VLLM_EXP_FIXED_BUDGET=128 -- $W
    run N$N-fixed256-r$rep VLLM_EXP_FIXED_BUDGET=256 -- $W
    run N$N-fixed512-r$rep VLLM_EXP_FIXED_BUDGET=512 -- $W
    run N$N-m0-r$rep VLLM_EXP_DEADLINE_MS=$D VLLM_EXP_COST_MODEL=$CM/m0-one.json -- $W
    run N$N-m2-r$rep VLLM_EXP_DEADLINE_MS=$D VLLM_EXP_COST_MODEL=$CM/m2-one.json -- $W
    run N$N-m2n-r$rep VLLM_EXP_DEADLINE_MS=$D VLLM_EXP_COST_MODEL=$CM/m2n-one.json -- $W
  done
done
echo "A100_LIVE_DONE $R"
