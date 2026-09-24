#!/bin/bash
# Stage 3 on one A100 GPU (pre-registration addendum 8 A, C, D, E). Waits for this GPU's stage 2.
# Old arms (addendum 4/5) keep venv-ctl (same code as their earlier repeats); new arms use venv-ctl2.
# Usage: campaign_stage3.sh GPU PORT NEW_REP_FIRST OLD_REP STAGE2_LOG
set -u
GPU=$1; PORT=$2; NR1=$3; OR=$4; S2=$5
until grep -q "STAGE4_DONE\|RUN_FAILED\|busy\|DIRTY\|not starting" $S2; do sleep 30; done
grep -q STAGE4_DONE $S2 || { echo "stage 4 did not finish cleanly - not starting"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPU VLLM_NO_USAGE_STATS=1 HF_HOME=/root/lab/hf HF_HUB_OFFLINE=1
cd /root/lab/lab-ctl3
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
H=$(git rev-parse --short HEAD); M=/root/lab/models/Qwen3-4B; CM=/root/lab/ctl-models; D=65; TR=/root/lab/traces
V1=/root/lab/venv-ctl/bin; V2=/root/lab/venv-ctl2/bin
guard() { local used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits); [ "$used" -gt 500 ] && { echo "GPU $GPU busy (${used} MiB) before $1 - stopping"; exit 2; }; }
run() { local R=$1 name=$2 VB=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  mkdir -p $R/trace; guard $name
  env PATH=$VB:$PATH "${envs[@]}" $VB/python scripts/replay_trace_serving.py --model $M --vllm-bin $VB/vllm --port $PORT --trace-dir $R/trace --output $R/$name.json "$@" > $R/$name.log 2>&1
  local rc=$?; echo "$(date +%H:%M:%S) $name exit=$rc $(tail -1 $R/$name.log | cut -c1-160)"; [ $rc -ne 0 ] && { echo "RUN_FAILED $name"; exit 3; }; }
PPAS="VLLM_EXP_PPAS=1"; PPAS8="VLLM_EXP_PPAS=1 VLLM_EXP_PPAS_BMAX=8192"
M2N="VLLM_EXP_DEADLINE_MS=$D VLLM_EXP_COST_MODEL=$CM/m2n-one.json VLLM_EXP_PARTITION=fcfs"
M0="VLLM_EXP_DEADLINE_MS=$D VLLM_EXP_COST_MODEL=$CM/m0-one.json VLLM_EXP_PARTITION=fcfs"
new_arms() { local R=$1 rep=$2; shift 2
  run $R ppas-r$rep $V2 $PPAS -- "$@" --max-num-batched-tokens 16384
  run $R ppas-8k-r$rep $V2 $PPAS8 -- "$@" --max-num-batched-tokens 8192
  run $R ctl-m2n-fcfs-cache-r$rep $V2 $M2N VLLM_EXP_CACHE_AWARE=1 -- "$@" --max-num-batched-tokens 8192; }
# A. pairing swap (GPU 0 only)
if [ "$GPU" = 0 ]; then
  RP=/root/lab/results/pairswap-a100-Qwen3-4B-$H; mkdir -p $RP
  for i in 0 1 2 3 4 5; do guard pairswap$i; env PATH=$V1:$PATH $V1/python scripts/measure_pairing_swap.py --model $M --config-index $i --trace-dir $RP > $RP/config$i.log 2>&1; echo "$(date +%H:%M:%S) pairswap config $i exit=$?"; done
  $V1/python scripts/measure_pairing_swap.py --analyze --trace-dir $RP --slope 3.3 --output $RP/pairswap.json
fi
# C. workload shift: new arms first repeat, then every arm one more repeat
PH="chat=azure:$TR/AzureLLMInferenceTrace_conv.csv:1:0:150;code=azure:$TR/AzureLLMInferenceTrace_code.csv:0.5:180:150;agent=mooncake:$TR/toolagent_trace.jsonl@3:0.2:0:150;chat2=azure:$TR/AzureLLMInferenceTrace_conv.csv:1:0:150"
RS=/root/lab/results/shift-a0538ee
new_arms $RS $NR1 --phases "$PH"
for B in 512 1024 2048 4096 8192; do run $RS b$B-r$OR $V1 NOOP=1 -- --phases "$PH" --max-num-batched-tokens $B; done
run $RS agentx-r$OR $V1 NOOP=1 -- --phases "$PH" --max-num-batched-tokens 2048 --long-prefill-token-threshold 512
run $RS ctl-m2n-fcfs-r$OR $V1 $M2N -- --phases "$PH" --max-num-batched-tokens 8192
run $RS ctl-m0-fcfs-r$OR $V1 $M0 -- --phases "$PH" --max-num-batched-tokens 8192
new_arms $RS $OR --phases "$PH"
# C. Mooncake tool-agent x0.2: new arms
RM=/root/lab/results/replay-mooncake-toolagent-017bd22
new_arms $RM $NR1 --trace mooncake:$TR/toolagent_trace.jsonl@3 --rate-scale 0.2 --window-s 240
# D. robustness windows (rule of addendum 8 D; offsets from scripts/select_replay_windows.py)
RW=/root/lab/results/replay-robust-$H
rob() { local tag=$1; shift
  run $RW $tag-default-r1 $V1 NOOP=1 -- "$@" --max-num-batched-tokens 2048
  run $RW $tag-ppas-r1 $V2 $PPAS -- "$@" --max-num-batched-tokens 16384
  run $RW $tag-ctl-m2n-fcfs-r1 $V1 $M2N -- "$@" --max-num-batched-tokens 8192; }
if [ "$GPU" = 0 ]; then
  rob mooncake-w60 --phases "w=mooncake:$TR/toolagent_trace.jsonl@3:0.2:60:240"
  rob mooncake-w150 --phases "w=mooncake:$TR/toolagent_trace.jsonl@3:0.2:150:240"
  rob burstgpt-w25740 --phases "w=burstgpt:$TR/BurstGPT_1.csv:60:25740:240"
  # E. second within-slot jitter seed, Mooncake tool-agent x0.2
  RJ=/root/lab/results/replay-jitter1-$H; J="--trace mooncake:$TR/toolagent_trace.jsonl@3 --rate-scale 0.2 --window-s 240 --jitter-seed 1"
  run $RJ x0.2-default-r1 $V1 NOOP=1 -- $J --max-num-batched-tokens 2048
  run $RJ x0.2-ppas-r1 $V2 $PPAS -- $J --max-num-batched-tokens 16384
  run $RJ x0.2-ctl-m2n-fcfs-r1 $V1 $M2N -- $J --max-num-batched-tokens 8192
else
  rob azure-code-w540 --phases "w=azure:$TR/AzureLLMInferenceTrace_code.csv:0.5:540:240"
  rob azure-code-w1050 --phases "w=azure:$TR/AzureLLMInferenceTrace_code.csv:0.5:1050:240"
  rob burstgpt-w64800 --phases "w=burstgpt:$TR/BurstGPT_1.csv:60:64800:240"
fi
echo "STAGE3_DONE GPU $GPU"
