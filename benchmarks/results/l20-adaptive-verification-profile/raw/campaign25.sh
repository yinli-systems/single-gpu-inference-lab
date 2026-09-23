#!/bin/bash
# Campaign25: does the startup-static adaptive-verification cost profile mis-price real context?
# 3 profile contexts (512 / 2048 / 8192-default) x 3 actual contexts (~512 / ~2048 / ~6000) x {code, prose}
# at B in {32, 64} (64 only where KV fits), DSpark K=7 + adaptive verification, TRITON_ATTN, 2 interleaved repeats.
# Tracer v4 records per step: chosen budget, max budget, predicted cost, table terms, estimated accepted.
source ~/inference/vllm_env.sh
cd ~/inference/single-gpu-inference-lab
for i in 1 2 3 4; do git fetch -q origin main && break; sleep 10; done
git checkout -q -- . && git clean -fdq && git checkout -q -B main origin/main
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
SP=$(python -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
python ~/inference/apply_tracer_v4_adaptive.py $SP || exit 1
python -c "import vllm.v1.worker.gpu.model_runner, vllm.v1.worker.gpu.spec_decode.adaptive_verification" || exit 1
R=~/inference/results/campaign25-$(git rev-parse --short HEAD); mkdir -p $R $R/trace
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 2
D=/home/hhai/inference/models/dspark_qwen3_4b_block7
CLS=code-ctx512,prose-ctx512,code-ctx2048,prose-ctx2048,code-ctx6000,prose-ctx6000
for rep in 0 1; do
  for PC in 512 2048 8192; do
    VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=$PC python scripts/measure_spec_geometry.py --model ~/inference/models/Qwen3-4B --vllm-bin vllm \
      --trace-dir $R/trace --output $R/pc$PC-r$rep.json --repeats 1 \
      --conditions "{\"adaptive\":{\"method\":\"dspark\",\"model\":\"$D\",\"num_speculative_tokens\":7,\"enable_adaptive_verification\":true}}" \
      --classes $CLS --batch-sizes 32,64 --max-tokens 512 >> $R/sweep.log 2>&1
  done
done
echo CAMPAIGN25_DONE
