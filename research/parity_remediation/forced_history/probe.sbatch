#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --job-name=sgi-force-position
set -euo pipefail
P=${1:?};SOURCE=${2:?};E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312;MODEL=/ssd/scxi253/check-before-act-20260927/models/Qwen3-4B-Instruct-2507
LOCAL=$(mktemp -d /tmp/sgi-force-probe-${SLURM_JOB_ID}-XXXXXX)
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TRITON_CACHE_DIR=$LOCAL/triton
export PYTHONPATH=$P/overlays/pristine:$SOURCE FLASHINFER_WORKSPACE_BASE=$P/cache/$SLURM_JOB_PARTITION/pristine XDG_CACHE_HOME=$LOCAL/cache
ROOT=$(dirname "$SOURCE");R=$ROOT/probe-runs/$SLURM_JOB_ID;mkdir -p "$R" "$ROOT/logs";scontrol show job $SLURM_JOB_ID -o > "$R/slurm.txt"
"$E/bin/python" -u "$SOURCE/probe_http.py" --source "$SOURCE" --model "$MODEL" --out "$R/result"
date -u +%FT%TZ > "$R/launcher-complete.txt"
