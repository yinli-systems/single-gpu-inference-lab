#!/bin/bash
# Pairing-swap counterexample on the L20 (pre-registration addendum 8 A). Refuses a busy GPU; kills nothing.
source ~/inference/vllm_env.sh
export VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1
cd ~/inference/single-gpu-inference-lab
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
R=~/inference/results/pairswap-l20-Qwen3-4B-$(git rev-parse --short HEAD); mkdir -p $R
for i in 0 1 2 3 4 5; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$USED" -gt 500 ] && { echo "GPU busy (${USED} MiB) - stopping"; exit 2; }
  python scripts/measure_pairing_swap.py --model ~/inference/models/Qwen3-4B --config-index $i --trace-dir $R > $R/config$i.log 2>&1
  echo "$(date +%H:%M:%S) config $i exit=$?"
done
python scripts/measure_pairing_swap.py --analyze --trace-dir $R --slope 6.5 --output $R/pairswap.json
echo "PAIRSWAP_DONE $R"
