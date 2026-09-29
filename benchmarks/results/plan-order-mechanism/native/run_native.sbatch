#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:15:00
#SBATCH --job-name=sgi-native-order
set -euo pipefail
P=/ssd/scxi253/single-gpu-inference-plan-order-20260929/native-integration
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
LOCAL=$(mktemp -d /tmp/sgi-native-order-${SLURM_JOB_ID}-XXXXXX)
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4
export XDG_CACHE_HOME=$LOCAL/cache TRITON_CACHE_DIR=$LOCAL/triton FLASHINFER_WORKSPACE_BASE=$LOCAL/flashinfer
export PYTHONPATH=$P/native-overlay:$P/source:${PYTHONPATH:-}
R=$P/runs/${SLURM_JOB_PARTITION}-${SLURM_JOB_ID}
cd $P/source
sha256sum -c native-source.sha256
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 2 > $P/logs/${SLURM_JOB_ID}-telemetry.csv &
MON=$!
trap 'kill "$MON" 2>/dev/null || true' EXIT
$E/bin/python -u native/measure_native_cycle.py --out "$R"
cp NATIVE_SOURCE_COMMIT.txt "$R/source_commit.txt"
date -u +%FT%TZ > "$R/completed_utc.txt"
