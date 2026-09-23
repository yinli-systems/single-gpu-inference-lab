#!/bin/bash
# A100 replication of l20-prefill-cost-geometry section 2.4 (partition of the same prefill budget
# at equal aggregate coordinate). Same harness arguments as campaign18 (C) and campaign19 (part2).
# GPU 1 only: GPU 0 runs another job on this pod and is never touched.
set -u
MODEL_NAME=${1:-Qwen3-4B}
export CUDA_VISIBLE_DEVICES=1 VLLM_NO_USAGE_STATS=1 HF_HOME=/root/lab/hf HF_HUB_OFFLINE=1
export PATH=/root/lab/venv/bin:$PATH
cd /root/lab/single-gpu-inference-lab
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
USED=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$USED" -gt 500 ] && { echo "GPU 1 busy (${USED} MiB) - refusing"; exit 2; }
R=/root/lab/results/a100-part-$MODEL_NAME-$(git rev-parse --short HEAD); mkdir -p $R/trace
M=/root/lab/models/$MODEL_NAME
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
echo "A100_PART_DONE $MODEL_NAME $R"
