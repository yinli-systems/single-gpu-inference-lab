#!/bin/bash
# SGLang pairing swap (addendum 2). SGLang 0.5.20 JIT-compiles its RoPE kernel, so the pod needs a CUDA
# toolkit (cuda-nvcc-13-0 et al. from NVIDIA's apt repo, CUDA_HOME below). DeepGEMM's JIT is disabled
# (SGLANG_ENABLE_JIT_DEEPGEMM=0): it refuses nvcc < 12.9 at init and is not used by a bf16 dense model.
set -u
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 SGLANG_ENABLE_JIT_DEEPGEMM=0
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
cd /root/lab/single-gpu-inference-lab
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
USED=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$USED" -gt 500 ] && { echo "GPU 0 busy (${USED} MiB) - refusing"; exit 2; }
R=/root/lab/results/h100-sglang-pairswap-Qwen3-4B-$(git rev-parse --short HEAD); mkdir -p $R
nvidia-smi > $R/nvidia-smi.txt
for i in 0 1 2 3 4 5; do
  PYTHONPATH=integrations/sglang_step_tracer /root/lab/venv-sgl/bin/python scripts/measure_pairing_swap_sglang.py --model /root/lab/models/Qwen3-4B --config-index $i --trace-dir $R > $R/config-$i.log 2>&1
  rc=$?; echo "$(date +%H:%M:%S) sglang config $i exit=$rc"
  [ $rc -ne 0 ] && { echo "CONFIG_FAILED $i"; exit 3; }
done
/root/lab/venv-sgl/bin/python scripts/measure_pairing_swap_sglang.py --analyze --trace-dir $R --output $R/pairswap.json > $R/analyze.log 2>&1
echo "SGLANG_DONE $R"
