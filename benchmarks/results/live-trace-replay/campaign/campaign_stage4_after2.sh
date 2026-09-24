#!/bin/bash
# Stage 4 on one A100 GPU (pre-registration addendum 10): SLO-aligned D = 100 ms controller arms.
# Usage: campaign_stage4.sh GPU PORT REP_A REP_B STAGE3_LOG
set -u
GPU=$1; PORT=$2; RA=$3; RB=$4; S3=$5
until grep -q "STAGE2_DONE\|RUN_FAILED\|busy\|DIRTY\|not starting" $S3; do sleep 30; done
grep -q STAGE2_DONE $S3 || { echo "stage 2 did not finish cleanly - not starting"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPU VLLM_NO_USAGE_STATS=1 HF_HOME=/root/lab/hf HF_HUB_OFFLINE=1
cd /root/lab/lab-ctl3
H=$(git rev-parse --short HEAD); M=/root/lab/models/Qwen3-4B; CM=/root/lab/ctl-models; TR=/root/lab/traces
V1=/root/lab/venv-ctl/bin; V2=/root/lab/venv-ctl2/bin
guard() { local used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits); [ "$used" -gt 500 ] && { echo "GPU $GPU busy (${used} MiB) before $1 - stopping"; exit 2; }; }
run() { local R=$1 name=$2 VB=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  mkdir -p $R/trace; guard $name
  env PATH=$VB:$PATH "${envs[@]}" $VB/python scripts/replay_trace_serving.py --model $M --vllm-bin $VB/vllm --port $PORT --trace-dir $R/trace --output $R/$name.json "$@" > $R/$name.log 2>&1
  local rc=$?; echo "$(date +%H:%M:%S) $name exit=$rc $(tail -1 $R/$name.log | cut -c1-160)"; [ $rc -ne 0 ] && { echo "RUN_FAILED $name"; exit 3; }; }
M2N="VLLM_EXP_DEADLINE_MS=100 VLLM_EXP_COST_MODEL=$CM/m2n-one.json VLLM_EXP_PARTITION=fcfs"
M0="VLLM_EXP_DEADLINE_MS=100 VLLM_EXP_COST_MODEL=$CM/m0-one.json VLLM_EXP_PARTITION=fcfs"
d100() { local R=$1 rep=$2; shift 2
  run $R ctl-m2n-fcfs-D100-r$rep $V1 $M2N -- "$@" --max-num-batched-tokens 8192
  run $R ctl-m0-fcfs-D100-r$rep $V1 $M0 -- "$@" --max-num-batched-tokens 8192
  run $R ctl-m2n-fcfs-cache-D100-r$rep $V2 $M2N VLLM_EXP_CACHE_AWARE=1 -- "$@" --max-num-batched-tokens 8192; }
d100 /root/lab/results/replay-mooncake-toolagent-017bd22 $RA --trace mooncake:$TR/toolagent_trace.jsonl@3 --rate-scale 0.2 --window-s 240
PH="chat=azure:$TR/AzureLLMInferenceTrace_conv.csv:1:0:150;code=azure:$TR/AzureLLMInferenceTrace_code.csv:0.5:180:150;agent=mooncake:$TR/toolagent_trace.jsonl@3:0.2:0:150;chat2=azure:$TR/AzureLLMInferenceTrace_conv.csv:1:0:150"
d100 /root/lab/results/shift-a0538ee $RA --phases "$PH"
d100 /root/lab/results/shift-a0538ee $RB --phases "$PH"
echo "STAGE4_DONE GPU $GPU"
