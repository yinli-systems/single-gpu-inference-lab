#!/bin/bash
# Stage 2 on one A100 GPU (pre-registration addendum 5): wait for this GPU's stage-1 replay, then one
# repeat of the workload-shift run (8 arms, interleaved), then one repeat of the BurstGPT control.
# Usage: campaign_stage2.sh GPU PORT REP STAGE1_LOG
set -u
GPU=$1; PORT=$2; REP=$3; S1=$4
until grep -q "REPLAY_DONE\|RUN_FAILED\|busy\|DIRTY" $S1; do sleep 30; done
grep -q REPLAY_DONE $S1 || { echo "stage 1 did not finish cleanly - not starting"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPU VLLM_NO_USAGE_STATS=1 HF_HOME=/root/lab/hf HF_HUB_OFFLINE=1 PATH=/root/lab/venv-ctl/bin:$PATH
cd /root/lab/lab-ctl2
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
H=$(git rev-parse --short HEAD); M=/root/lab/models/Qwen3-4B; CM=/root/lab/ctl-models; D=65; TR=/root/lab/traces
PH="chat=azure:$TR/AzureLLMInferenceTrace_conv.csv:1:0:150;code=azure:$TR/AzureLLMInferenceTrace_code.csv:0.5:180:150;agent=mooncake:$TR/toolagent_trace.jsonl@3:0.2:0:150;chat2=azure:$TR/AzureLLMInferenceTrace_conv.csv:1:0:150"
run() { local R=$1 name=$2; shift 2
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  mkdir -p $R/trace
  local used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -gt 500 ] && { echo "GPU $GPU busy (${used} MiB) before $name - stopping"; exit 2; }
  env "${envs[@]}" python scripts/replay_trace_serving.py --model $M --vllm-bin /root/lab/venv-ctl/bin/vllm --port $PORT --trace-dir $R/trace --output $R/$name.json "$@" > $R/$name.log 2>&1
  local rc=$?; echo "$(date +%H:%M:%S) $name exit=$rc $(tail -1 $R/$name.log | cut -c1-160)"; [ $rc -ne 0 ] && { echo "RUN_FAILED $name"; exit 3; }; }
RS=/root/lab/results/shift-$H
for B in 512 1024 2048 4096 8192; do run $RS b$B-r$REP NOOP=1 -- --phases "$PH" --max-num-batched-tokens $B; done
run $RS agentx-r$REP NOOP=1 -- --phases "$PH" --max-num-batched-tokens 2048 --long-prefill-token-threshold 512
run $RS ctl-m2n-fcfs-r$REP VLLM_EXP_DEADLINE_MS=$D VLLM_EXP_COST_MODEL=$CM/m2n-one.json VLLM_EXP_PARTITION=fcfs -- --phases "$PH" --max-num-batched-tokens 8192
run $RS ctl-m0-fcfs-r$REP VLLM_EXP_DEADLINE_MS=$D VLLM_EXP_COST_MODEL=$CM/m0-one.json VLLM_EXP_PARTITION=fcfs -- --phases "$PH" --max-num-batched-tokens 8192
RB=/root/lab/results/replay-burstgpt-t14h-$H; W="--phases chat=burstgpt:$TR/BurstGPT_1.csv:60:50400:240"
run $RB x60-default-r$REP NOOP=1 -- $W --max-num-batched-tokens 2048
run $RB x60-agentx-r$REP NOOP=1 -- $W --max-num-batched-tokens 2048 --long-prefill-token-threshold 512
run $RB x60-b8192-r$REP NOOP=1 -- $W --max-num-batched-tokens 8192
run $RB x60-ctl-m2n-fcfs-r$REP VLLM_EXP_DEADLINE_MS=$D VLLM_EXP_COST_MODEL=$CM/m2n-one.json VLLM_EXP_PARTITION=fcfs -- $W --max-num-batched-tokens 8192
run $RB x60-ctl-m0-fcfs-r$REP VLLM_EXP_DEADLINE_MS=$D VLLM_EXP_COST_MODEL=$CM/m0-one.json VLLM_EXP_PARTITION=fcfs -- $W --max-num-batched-tokens 8192
run $RB x60-ctl-m2n-equal-r$REP VLLM_EXP_DEADLINE_MS=$D VLLM_EXP_COST_MODEL=$CM/m2n-one.json VLLM_EXP_PARTITION=equal -- $W --max-num-batched-tokens 8192
RC=/root/lab/results/replay-azure-code-t180-$H
for S in 0.25 0.5; do
  P="--phases code=azure:$TR/AzureLLMInferenceTrace_code.csv:$S:180:240"
  run $RC x$S-default-r$REP NOOP=1 -- $P --max-num-batched-tokens 2048
  run $RC x$S-agentx-r$REP NOOP=1 -- $P --max-num-batched-tokens 2048 --long-prefill-token-threshold 512
  run $RC x$S-b8192-r$REP NOOP=1 -- $P --max-num-batched-tokens 8192
  run $RC x$S-ctl-m2n-fcfs-r$REP VLLM_EXP_DEADLINE_MS=$D VLLM_EXP_COST_MODEL=$CM/m2n-one.json VLLM_EXP_PARTITION=fcfs -- $P --max-num-batched-tokens 8192
  run $RC x$S-ctl-m0-fcfs-r$REP VLLM_EXP_DEADLINE_MS=$D VLLM_EXP_COST_MODEL=$CM/m0-one.json VLLM_EXP_PARTITION=fcfs -- $P --max-num-batched-tokens 8192
  run $RC x$S-ctl-m2n-equal-r$REP VLLM_EXP_DEADLINE_MS=$D VLLM_EXP_COST_MODEL=$CM/m2n-one.json VLLM_EXP_PARTITION=equal -- $P --max-num-batched-tokens 8192
done
echo "STAGE2_DONE GPU $GPU REP $REP"
