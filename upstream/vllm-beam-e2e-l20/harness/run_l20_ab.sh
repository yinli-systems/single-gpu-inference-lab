#!/bin/bash
# Offline beam search A/B on the L20: stock 0.29.0 twice, then patched.
set -u
source ~/inference/vllm_env.sh
export VLLM_LOGGING_LEVEL=WARNING VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1
D=~/inference/beam-ab; R=$D/results/$(date +%Y%m%d-%H%M%S); mkdir -p $R
PY=~/inference/venv-vllm/bin/python
BS=~/inference/venv-vllm/lib/python3.12/site-packages/vllm/entrypoints/generate/beam_search
cp $BS/offline.py $R/offline.stock.py; cp $BS/utils.py $R/utils.stock.py
md5sum $BS/offline.py $BS/utils.py > $R/stock.md5
restore() { cp $R/offline.stock.py $BS/offline.py; cp $R/utils.stock.py $BS/utils.py; }
trap restore EXIT
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader | tee $R/gpu.txt
cd $D
for arm in stock1 patched1 stock2 patched2; do
  case $arm in
    stock*) restore ;;
    patched*) cp patched/offline.py $BS/offline.py; cp patched/utils.py $BS/utils.py ;;
  esac
  echo "===== $arm"
  $PY beam_e2e.py ~/inference/models/Qwen2.5-0.5B-Instruct $R/$arm.json 2>&1 | grep -E "^(plain|json) |Error|error" | tee $R/$arm.log
done
restore; trap - EXIT
md5sum -c $R/stock.md5 && echo "VENV RESTORED"
$PY compare_arms.py $R | tee $R/compare.txt
echo "L20_AB_DONE $R"
