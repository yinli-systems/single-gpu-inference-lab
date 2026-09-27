#!/bin/bash
# H100 pairing swap: the six configurations of measure_pairing_swap.py, one process each, then analysis.
set -u
MODEL_NAME=${1:-Qwen3-4B}
export CUDA_VISIBLE_DEVICES=0 VLLM_NO_USAGE_STATS=1 HF_HOME=/root/lab/hf HF_HUB_OFFLINE=1
export PATH=/root/lab/venv/bin:$PATH
cd /root/lab/single-gpu-inference-lab
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
USED=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$USED" -gt 500 ] && { echo "GPU 0 busy (${USED} MiB) - refusing"; exit 2; }
R=/root/lab/results/h100-pairswap-$MODEL_NAME-$(git rev-parse --short HEAD); mkdir -p $R
nvidia-smi > $R/nvidia-smi.txt
for i in 0 1 2 3 4 5; do
  python scripts/measure_pairing_swap.py --model /root/lab/models/$MODEL_NAME --config-index $i --trace-dir $R > $R/config-$i.log 2>&1
  rc=$?; echo "$(date +%H:%M:%S) $MODEL_NAME pairswap config $i exit=$rc"
  [ $rc -ne 0 ] && { echo "CONFIG_FAILED $i"; exit 3; }
done
python scripts/measure_pairing_swap.py --analyze --trace-dir $R --output $R/pairswap.json > $R/analyze.log 2>&1
echo "H100_PAIRSWAP_DONE $MODEL_NAME $R"
