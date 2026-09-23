#!/bin/bash
# Spec-decode cost geometry sweep (Qwen3-4B, TRITON_ATTN): no-spec, DSpark K=7, DSpark K=7 + adaptive verification, EAGLE3 K=5.
source ~/inference/vllm_env.sh
cd ~/inference/single-gpu-inference-lab
for i in 1 2 3 4; do git fetch -q origin stage2-compact-sampling-mask && break; sleep 10; done
git checkout -q -- . && git clean -fdq && git checkout -q -B stage2-compact-sampling-mask origin/stage2-compact-sampling-mask
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
R=~/inference/results/campaign23-$(git rev-parse --short HEAD); mkdir -p $R $R/trace
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 2
D=/home/hhai/inference/models/dspark_qwen3_4b_block7; E=/home/hhai/inference/models/Qwen3-4B_eagle3
python scripts/measure_spec_geometry.py --model ~/inference/models/Qwen3-4B --vllm-bin vllm --trace-dir $R/trace --output $R/sweep.json --repeats 2 \
  --conditions "{\"nospec\":null,\"dspark7\":{\"method\":\"dspark\",\"model\":\"$D\",\"num_speculative_tokens\":7},\"dspark7-adaptive\":{\"method\":\"dspark\",\"model\":\"$D\",\"num_speculative_tokens\":7,\"enable_adaptive_verification\":true},\"eagle3-5\":{\"method\":\"eagle3\",\"model\":\"$E\",\"num_speculative_tokens\":5}}" \
  --batch-sizes 8,16,32,64 --max-tokens 256 > $R/sweep.log 2>&1
echo CAMPAIGN23_DONE
