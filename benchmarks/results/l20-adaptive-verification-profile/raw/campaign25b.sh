#!/bin/bash
# Campaign25b: (1) hindsight goodput-vs-budget curves on the decisive cells via a forced verification budget
# (same 8192 default profile, budget overridden per server); (2) the matched oracle cell profile 6144 x actual ~6144.
source ~/inference/vllm_env.sh
cd ~/inference/single-gpu-inference-lab
git fetch -q origin main; git checkout -q -- . && git clean -fdq && git checkout -q -B main origin/main
SP=$(python -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
python ~/inference/patch_force_budget.py $SP || exit 1
python -c "import vllm.v1.worker.gpu.spec_decode.adaptive_verification" || exit 1
R=~/inference/results/campaign25b-$(git rev-parse --short HEAD); mkdir -p $R $R/trace
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 2
D=/home/hhai/inference/models/dspark_qwen3_4b_block7
AV="{\"adaptive\":{\"method\":\"dspark\",\"model\":\"$D\",\"num_speculative_tokens\":7,\"enable_adaptive_verification\":true}}"
for FB in 64 96 128 160 192 224; do
  VLLM_EXP_FORCE_AV_BUDGET=$FB python scripts/measure_spec_geometry.py --model ~/inference/models/Qwen3-4B --vllm-bin vllm --trace-dir $R/trace --output $R/fb$FB-r0.json --repeats 1 \
    --conditions "$AV" --classes prose-ctx512,prose-ctx2048,code-ctx512 --batch-sizes 32 --max-tokens 512 >> $R/sweep.log 2>&1
done
for FB in 128 192 256 320 384 448; do
  VLLM_EXP_FORCE_AV_BUDGET=$FB python scripts/measure_spec_geometry.py --model ~/inference/models/Qwen3-4B --vllm-bin vllm --trace-dir $R/trace --output $R/fb$FB-b64-r0.json --repeats 1 \
    --conditions "$AV" --classes prose-ctx512 --batch-sizes 64 --max-tokens 512 >> $R/sweep.log 2>&1
done
for rep in 0 1; do
  VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=6144 python scripts/measure_spec_geometry.py --model ~/inference/models/Qwen3-4B --vllm-bin vllm --trace-dir $R/trace --output $R/pc6144-r$rep.json --repeats 1 \
    --conditions "$AV" --classes code-ctx6000,prose-ctx6000 --batch-sizes 32 --max-tokens 512 >> $R/sweep.log 2>&1
done
echo CAMPAIGN25B_DONE
