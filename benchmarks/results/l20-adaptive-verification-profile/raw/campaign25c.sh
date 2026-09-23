#!/bin/bash
# 6k regime with a real co-decoding window: B=24, 2048-token completions; profiles 512 / 6144 (matched) / 8192 (default).
source ~/inference/vllm_env.sh
cd ~/inference/single-gpu-inference-lab
R=~/inference/results/campaign25c-$(git rev-parse --short HEAD); mkdir -p $R $R/trace
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 2
D=/home/hhai/inference/models/dspark_qwen3_4b_block7
AV="{\"adaptive\":{\"method\":\"dspark\",\"model\":\"$D\",\"num_speculative_tokens\":7,\"enable_adaptive_verification\":true}}"
for rep in 0 1; do for PC in 512 6144 8192; do
  VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=$PC python scripts/measure_spec_geometry.py --model ~/inference/models/Qwen3-4B --vllm-bin vllm --trace-dir $R/trace --output $R/pc$PC-r$rep.json --repeats 1 \
    --conditions "$AV" --classes code-ctx6000,prose-ctx6000 --batch-sizes 24 --max-tokens 2048 --kv-token-capacity 230000 >> $R/sweep.log 2>&1
done; done
echo CAMPAIGN25C_DONE
