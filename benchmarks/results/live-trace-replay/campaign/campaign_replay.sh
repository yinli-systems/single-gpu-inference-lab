#!/bin/bash
# Live trace-replay goodput (pre-registration addendum 4). Usage: campaign_replay.sh GPU PORT NAME TRACE "SCALES"
set -u
GPU=$1; PORT=$2; NAME=$3; TRACE=$4; SCALES=$5
export CUDA_VISIBLE_DEVICES=$GPU VLLM_NO_USAGE_STATS=1 HF_HOME=/root/lab/hf HF_HUB_OFFLINE=1 PATH=/root/lab/venv-ctl/bin:$PATH
cd /root/lab/lab-ctl
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
R=/root/lab/results/replay-$NAME-$(git rev-parse --short HEAD); mkdir -p $R/trace
M=/root/lab/models/Qwen3-4B; CM=/root/lab/ctl-models; D=65
nvidia-smi > $R/nvidia-smi.txt
run() { local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -gt 500 ] && { echo "GPU $GPU busy (${used} MiB) before $name - stopping"; exit 2; }
  env "${envs[@]}" python scripts/replay_trace_serving.py --model $M --vllm-bin /root/lab/venv-ctl/bin/vllm --port $PORT --trace $TRACE --window-s 240 --trace-dir $R/trace --output $R/$name.json "$@" > $R/$name.log 2>&1
  local rc=$?; echo "$(date +%H:%M:%S) $name exit=$rc $(tail -1 $R/$name.log | cut -c1-200)"; [ $rc -ne 0 ] && { echo "RUN_FAILED $name"; exit 3; }; }
for rep in 1 2; do
  for S in $SCALES; do
    run x$S-default-r$rep       NOOP=1 -- --rate-scale $S --max-num-batched-tokens 2048
    run x$S-agentx-r$rep        NOOP=1 -- --rate-scale $S --max-num-batched-tokens 2048 --long-prefill-token-threshold 512
    run x$S-b8192-r$rep         NOOP=1 -- --rate-scale $S --max-num-batched-tokens 8192
    run x$S-ctl-m2n-fcfs-r$rep  VLLM_EXP_DEADLINE_MS=$D VLLM_EXP_COST_MODEL=$CM/m2n-one.json VLLM_EXP_PARTITION=fcfs -- --rate-scale $S --max-num-batched-tokens 8192
    run x$S-ctl-m0-fcfs-r$rep   VLLM_EXP_DEADLINE_MS=$D VLLM_EXP_COST_MODEL=$CM/m0-one.json VLLM_EXP_PARTITION=fcfs -- --rate-scale $S --max-num-batched-tokens 8192
    run x$S-ctl-m2n-equal-r$rep VLLM_EXP_DEADLINE_MS=$D VLLM_EXP_COST_MODEL=$CM/m2n-one.json VLLM_EXP_PARTITION=equal -- --rate-scale $S --max-num-batched-tokens 8192
  done
done
echo "REPLAY_DONE $NAME $R"
