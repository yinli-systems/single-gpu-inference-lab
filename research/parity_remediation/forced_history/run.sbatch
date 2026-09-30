#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:45:00
#SBATCH --job-name=sgi-forced-history
set -euo pipefail
P=${1:?}; SOURCE=${2:?}
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
MODEL=/ssd/scxi253/check-before-act-20260927/models/Qwen3-4B-Instruct-2507
LOCAL=$(mktemp -d /tmp/sgi-forced-${SLURM_JOB_ID}-XXXXXX)
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TRITON_CACHE_DIR=$LOCAL/triton
CCCL=$P/overlays/pristine/flashinfer/data/cccl
export CPATH=$CCCL/libcudacxx/include:$CCCL/cub:$CCCL/thrust NVCC_PREPEND_FLAGS="-I$CCCL/libcudacxx/include -I$CCCL/cub -I$CCCL/thrust"
cd "$SOURCE"; sha256sum -c source.sha256
ROOT=$(dirname "$SOURCE");mkdir -p "$ROOT/runs/$SLURM_JOB_ID" "$ROOT/logs";R="$ROOT/runs/$SLURM_JOB_ID"
scontrol show job "$SLURM_JOB_ID" -o > "$R/slurm.txt";nvidia-smi --query-gpu=name,uuid,driver_version,power.limit --format=csv > "$R/hardware.csv"
for ITEM in pristine:0 cap:0 cap:1 pristine:1; do
 MODE=${ITEM%:*};REP=${ITEM#*:};OUT="$R/$MODE-$REP"
 export PYTHONPATH=$P/overlays/$MODE:$SOURCE FLASHINFER_WORKSPACE_BASE=$P/cache/$SLURM_JOB_PARTITION/$MODE XDG_CACHE_HOME=$LOCAL/cache-$MODE-$REP
 "$E/bin/python" -u http_probe.py --source "$SOURCE" --cases "$SOURCE/cases.json" --model "$MODEL" --out "$OUT" --mode "$MODE" --rep "$REP"
done
date -u +%FT%TZ > "$R/launcher-complete.txt"
