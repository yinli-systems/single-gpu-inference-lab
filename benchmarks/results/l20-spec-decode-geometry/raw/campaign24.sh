#!/bin/bash
# Campaign24: (A) K-prefix invariance and (B) drafter cost surface T_draft(B, L, K) for DSpark, K in {1,2,3,5,7},
# one server per K (graph shapes differ), homogeneous code/prose at short / ~3k / ~6k context, B in {4,8,16,32}.
# Spec trace now records proposed draft ids + generated ids per request per step (patch_spec_ids.py).
source ~/inference/vllm_env.sh
cd ~/inference/single-gpu-inference-lab
for i in 1 2 3 4; do git fetch -q origin stage2-compact-sampling-mask && break; sleep 10; done
git checkout -q -- . && git clean -fdq && git checkout -q -B stage2-compact-sampling-mask origin/stage2-compact-sampling-mask
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
SP=$(python -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
python ~/inference/patch_spec_ids.py $SP/v1/core/sched/scheduler.py || exit 1
R=~/inference/results/campaign24-$(git rev-parse --short HEAD); mkdir -p $R $R/trace
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 2
D=/home/hhai/inference/models/dspark_qwen3_4b_block7
C="{"; for K in 1 2 3 5 7; do C="$C\"dspark$K\":{\"method\":\"dspark\",\"model\":\"$D\",\"num_speculative_tokens\":$K},"; done; C="${C%,}}"
python scripts/measure_spec_geometry.py --model ~/inference/models/Qwen3-4B --vllm-bin vllm --trace-dir $R/trace --output $R/ksweep.json --repeats 1 \
  --conditions "$C" --classes code-short,prose-short,code-mid,prose-mid,code-long,prose-long --batch-sizes 4,8,16,32 --max-tokens 256 > $R/ksweep.log 2>&1
echo CAMPAIGN24_DONE
