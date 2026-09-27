#!/bin/bash
# H100 (Hopper, FA3) replication: every cell of the A100 partition + prediction campaigns for one model.
# Single-GPU pod; refuses a busy GPU or a dirty tree. The 32k context cell uses 32768 tokens for the
# Qwen3 family (as on the A100) and 32640 for Qwen2.5 (32768-position limit, as in campaign_a100_shape.sh).
set -u
MODEL_NAME=${1:-Qwen3-4B}
export CUDA_VISIBLE_DEVICES=0 VLLM_NO_USAGE_STATS=1 HF_HOME=/root/lab/hf HF_HUB_OFFLINE=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PATH=/root/lab/venv/bin:$PATH
cd /root/lab/single-gpu-inference-lab
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
USED=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$USED" -gt 500 ] && { echo "GPU 0 busy (${USED} MiB) - refusing"; exit 2; }
R=/root/lab/results/h100-shape-$MODEL_NAME-$(git rev-parse --short HEAD); mkdir -p $R/trace
M=/root/lab/models/$MODEL_NAME
case $MODEL_NAME in Qwen2.5*) L32=32640;; *) L32=32768;; esac
nvidia-smi > $R/nvidia-smi.txt
run() { local name=$1; shift
  python scripts/measure_prefill_interference.py --model $M --vllm-bin /root/lab/venv/bin/vllm --repeats 3 --trace-dir $R/trace --output $R/$name.json "$@" > $R/$name.log 2>&1
  local rc=$?
  echo "$(date +%H:%M:%S) $MODEL_NAME $name exit=$rc"
  [ $rc -ne 0 ] && { echo "CELL_FAILED $name"; exit 3; }; }
run part-1x1024  --chunk-budgets 1024 --background 8 --background-tokens 4096 --inject 1 --long-tokens 16384
run part-2x512   --chunk-budgets 1024 --background 8 --background-tokens 4096 --inject 2 --long-tokens 16384 --long-prefill-token-threshold 512
run part-4x256   --chunk-budgets 1024 --background 8 --background-tokens 4096 --inject 4 --long-tokens 16384 --long-prefill-token-threshold 256
run part2-1x2048 --chunk-budgets 2048 --background 8 --background-tokens 4096 --inject 1 --long-tokens 16384
run part2-4x512  --chunk-budgets 2048 --background 8 --background-tokens 4096 --inject 4 --long-tokens 16384 --long-prefill-token-threshold 512
run part2-8x256  --chunk-budgets 2048 --background 8 --background-tokens 4096 --inject 8 --long-tokens 16384 --long-prefill-token-threshold 256
for L in 4096 8192 16384; do run ctx-bg8-L$L --chunk-budgets 512,2048 --background 8 --background-tokens 4096 --inject 2 --long-tokens $L; done
run ctx-bg8-L32768 --chunk-budgets 512,2048 --background 8 --background-tokens 4096 --inject 2 --long-tokens $L32
for B in 4 16 32; do run load-bg$B-L16384 --chunk-budgets 512 --background $B --background-tokens 4096 --inject 2 --long-tokens 16384; done
run skew-bal  --chunk-budgets 512 --background 8 --background-prompt-tokens-list 4096 --background-tokens 4096 --inject 2 --long-tokens 16384
run skew-skew --chunk-budgets 512 --background 8 --background-prompt-tokens-list 7936,256 --background-tokens 4096 --inject 2 --long-tokens 16384
echo "H100_SHAPE_DONE $MODEL_NAME $R"
