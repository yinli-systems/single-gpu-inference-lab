#!/bin/bash
#SBATCH --job-name=beam-ab
#SBATCH --partition=hp_5090
#SBATCH --qos=gpugpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:30:00
#SBATCH --output=/ssd/scxi253/beam-ab/logs/beam-ab-%j.out
# Offline beam search A/B on one GPU: stock vLLM 0.29.0 (twice, for the
# run-to-run noise floor) vs 0.29.0 with the two beam-search patches.
set -u
W=/data/run01/scxi253/inference
mkdir -p /tmp/scxi253
NEWV=$(tar -xOf $W/venv-vllm.tar venv-vllm/VENV_VERSION 2>/dev/null || echo v1)
CURV=$(cat /tmp/scxi253/venv-vllm/VENV_VERSION 2>/dev/null || echo none)
if [ "$NEWV" != "$CURV" ]; then rm -rf /tmp/scxi253/venv-vllm; tar -C /tmp/scxi253 -xf $W/venv-vllm.tar; fi
[ -d /tmp/scxi253/libfix ] || tar -C /tmp/scxi253 -xf $W/libfix.tar
source /tmp/scxi253/venv-vllm/bin/activate
export LD_LIBRARY_PATH=/tmp/scxi253/libfix${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_HOME=/tmp/scxi253/venv-vllm/lib/python3.13/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING
export XDG_CACHE_HOME=/tmp/scxi253/xdg-cache XDG_CONFIG_HOME=/tmp/scxi253/xdg-config VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_CACHE_ROOT=/tmp/scxi253/vllm-cache TRITON_CACHE_DIR=/tmp/scxi253/triton-cache TORCHINDUCTOR_CACHE_DIR=/tmp/scxi253/inductor-cache HF_HOME=/tmp/scxi253/hf FLASHINFER_WORKSPACE_BASE=/tmp/scxi253/flashinfer
mkdir -p $XDG_CACHE_HOME $XDG_CONFIG_HOME $VLLM_CACHE_ROOT $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $HF_HOME $FLASHINFER_WORKSPACE_BASE

B=/ssd/scxi253/beam-ab
R=$B/results/beam-ab-$SLURM_JOB_ID; mkdir -p $R
echo "host $(hostname) gpus $CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader | tee $R/gpu.txt
# refuse to measure on a GPU someone else is already using (four earlier jobs died on one)
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${USED:-0}" -gt 1000 ]; then echo "GPU already has ${USED} MiB in use - aborting"; exit 2; fi

MODEL_NAME=Qwen2.5-0.5B-Instruct
M=/tmp/scxi253/models/$MODEL_NAME; mkdir -p $(dirname $M); [ -f $M/config.json ] || cp -r $B/models/$MODEL_NAME $M
BS=/tmp/scxi253/venv-vllm/lib/python3.13/site-packages/vllm/entrypoints/generate/beam_search
cp $BS/offline.py $R/offline.stock.py; cp $BS/utils.py $R/utils.stock.py
cd $B/scripts
for arm in stock1 stock2 patched; do
  if [ "$arm" = patched ]; then cp patched/offline.py $BS/offline.py; cp patched/utils.py $BS/utils.py; fi
  echo "===== $arm"
  python beam_e2e.py $M $R/$arm.json 2>&1 | grep -v "^INFO\|^WARNING\|it/s\]" | tee $R/$arm.log
done
# restore the node-local venv
cp $R/offline.stock.py $BS/offline.py; cp $R/utils.stock.py $BS/utils.py
python compare_arms.py $R | tee $R/compare.txt
echo BEAM_AB_DONE
