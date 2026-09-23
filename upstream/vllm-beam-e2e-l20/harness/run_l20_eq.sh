#!/bin/bash
# Output-equality check under batch invariance: stock1, stock2, patched1.
set -u
source ~/inference/vllm_env.sh
export VLLM_LOGGING_LEVEL=WARNING VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1
export VLLM_BATCH_INVARIANT=1 BEAM_EQUALITY_MODE=1 BEAM_PREFIX_CACHING=${BEAM_PREFIX_CACHING:-1}
D=~/inference/beam-ab; R=$D/results/eq-$(date +%Y%m%d-%H%M%S)-pc$BEAM_PREFIX_CACHING; mkdir -p $R
PY=~/inference/venv-vllm/bin/python
BS=~/inference/venv-vllm/lib/python3.12/site-packages/vllm/entrypoints/generate/beam_search
cp $BS/offline.py $R/offline.stock.py; cp $BS/utils.py $R/utils.stock.py
md5sum $BS/offline.py $BS/utils.py > $R/stock.md5
restore() { cp $R/offline.stock.py $BS/offline.py; cp $R/utils.stock.py $BS/utils.py; }
trap restore EXIT
cd $D
for arm in stock1 stock2 patched1; do
  case $arm in
    stock*) restore ;;
    patched*) cp patched/offline.py $BS/offline.py; cp patched/utils.py $BS/utils.py ;;
  esac
  echo "===== $arm"
  $PY beam_e2e.py ~/inference/models/Qwen2.5-0.5B-Instruct $R/$arm.json 2>&1 | grep -E "^(plain|json) |Error|error" | tee $R/$arm.log
done
restore; trap - EXIT
md5sum -c $R/stock.md5 && echo "VENV RESTORED"
echo "L20_EQ_DONE $R"
