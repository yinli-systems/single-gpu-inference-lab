#!/bin/bash
# Attention-shape test of l20-prefill-cost-geometry: same cells as the A100 replication
# (campaign_a100_part.sh + campaign_a100_pred.sh), Qwen2.5-1.5B-Instruct (28 layers, 12 q / 2 kv heads)
# instead of Qwen3-4B (36 layers, 32 q / 8 kv). Refuses a busy GPU or a dirty tree; never kills anything.
set -u
source ~/inference/vllm_env.sh
MODEL_NAME=${1:-Qwen2.5-1.5B-Instruct}
# max-model-len stays 40960 like every other cell; prompts are random tokens, only step timing is measured
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_NO_USAGE_STATS=1
cd ~/inference/single-gpu-inference-lab
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$USED" -gt 500 ] && { echo "GPU busy (${USED} MiB) - refusing"; exit 2; }
R=~/inference/results/l20-shape-$MODEL_NAME-$(git rev-parse --short HEAD); mkdir -p $R/trace
M=~/inference/models/$MODEL_NAME
nvidia-smi > $R/nvidia-smi.txt
run() { local name=$1; shift
  python scripts/measure_prefill_interference.py --model $M --vllm-bin vllm --repeats 3 --trace-dir $R/trace --output $R/$name.json "$@" > $R/$name.log 2>&1
  local rc=$?
  echo "$(date +%H:%M:%S) $MODEL_NAME $name exit=$rc"
  [ $rc -ne 0 ] && { echo "CELL_FAILED $name"; exit 3; }; }
run part-1x1024  --chunk-budgets 1024 --background 8 --background-tokens 4096 --inject 1 --long-tokens 16384
run part-2x512   --chunk-budgets 1024 --background 8 --background-tokens 4096 --inject 2 --long-tokens 16384 --long-prefill-token-threshold 512
run part-4x256   --chunk-budgets 1024 --background 8 --background-tokens 4096 --inject 4 --long-tokens 16384 --long-prefill-token-threshold 256
run part2-1x2048 --chunk-budgets 2048 --background 8 --background-tokens 4096 --inject 1 --long-tokens 16384
run part2-4x512  --chunk-budgets 2048 --background 8 --background-tokens 4096 --inject 4 --long-tokens 16384 --long-prefill-token-threshold 512
run part2-8x256  --chunk-budgets 2048 --background 8 --background-tokens 4096 --inject 8 --long-tokens 16384 --long-prefill-token-threshold 256
for L in 4096 8192 16384 32768; do run ctx-bg8-L$L --chunk-budgets 512,2048 --background 8 --background-tokens 4096 --inject 2 --long-tokens $L; done
for B in 4 16 32; do run load-bg$B-L16384 --chunk-budgets 512 --background $B --background-tokens 4096 --inject 2 --long-tokens 16384; done
run skew-bal  --chunk-budgets 512 --background 8 --background-prompt-tokens-list 4096 --background-tokens 4096 --inject 2 --long-tokens 16384
run skew-skew --chunk-budgets 512 --background 8 --background-prompt-tokens-list 7936,256 --background-tokens 4096 --inject 2 --long-tokens 16384
echo "L20_SHAPE_DONE $MODEL_NAME $R"
