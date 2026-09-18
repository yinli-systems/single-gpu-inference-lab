#!/bin/bash
# C. decode KV-distribution control (balanced vs skewed at equal aggregate decode KV) and
# a second equal-aggregate partition series at chunk 2048. Tracer v2 installed; 3 repeats per cell.
source ~/inference/vllm_env.sh
cd ~/inference/single-gpu-inference-lab
for i in 1 2 3 4; do git fetch -q origin stage2-compact-sampling-mask && break; sleep 10; done
git checkout -q -- . && git clean -fdq && git checkout -q -B stage2-compact-sampling-mask origin/stage2-compact-sampling-mask
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
R=~/inference/results/campaign19-$(git rev-parse --short HEAD); mkdir -p $R $R/trace
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 2
M=~/inference/models/Qwen3-4B
run() { local name=$1; shift
  python scripts/measure_prefill_interference.py --model $M --vllm-bin vllm --repeats 3 --trace-dir $R/trace --output $R/$name.json "$@" > $R/$name.log 2>&1; }
set -x
# C. 8 decoders, aggregate prompt KV 32768 in both: balanced 8x4096 vs skewed 4x7936+4x256; 2 injected 16k prefills, chunk 512
run skew-bal  --chunk-budgets 512 --background 8 --background-prompt-tokens-list 4096 --background-tokens 4096 --inject 2 --long-tokens 16384
run skew-skew --chunk-budgets 512 --background 8 --background-prompt-tokens-list 7936,256 --background-tokens 4096 --inject 2 --long-tokens 16384
# B'. second equal-aggregate partition series: 2048 prefill tokens/step as 1x2048, 4x512, 8x256 (16k prefills, 8 decoders)
run part2-1x2048 --chunk-budgets 2048 --background 8 --background-tokens 4096 --inject 1 --long-tokens 16384
run part2-4x512  --chunk-budgets 2048 --background 8 --background-tokens 4096 --inject 4 --long-tokens 16384 --long-prefill-token-threshold 512
run part2-8x256  --chunk-budgets 2048 --background 8 --background-tokens 4096 --inject 8 --long-tokens 16384 --long-prefill-token-threshold 256
echo CAMPAIGN19_DONE
