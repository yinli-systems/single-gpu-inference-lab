#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --job-name=sgi-full-http-paired
set -euo pipefail
P=${1:?HTTP campaign}; K=${2:?qualified kernel campaign}; MODEL=${3:?model}; STAGE=${4:?stage}; ALLOCATION=${5:?allocation}
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 PYTHONNOUSERSITE=1
export PYTHONPATH=$P/harness
cd "$P"
sha256sum -c validation.sha256 > "$P/receipts/source-$SLURM_JOB_ID.txt"
UUID=$("$E/bin/python" -c 'import torch; u=str(torch.cuda.get_device_properties(0).uuid); print(u if u.startswith("GPU-") else "GPU-"+u)')
nvidia-smi --id="$UUID" --query-gpu=timestamp,uuid,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 2 > "$P/logs/telemetry-$SLURM_JOB_ID.csv" & MON=$!
finish(){ RC=$?;trap - EXIT;kill "$MON" 2>/dev/null || true;printf '%s\n' "$RC" > "$P/receipts/exit-$SLURM_JOB_ID.txt";exit "$RC"; };trap finish EXIT
srun --cpu-bind=cores "$E/bin/python" "$P/harness/research/selector_v4/serving/http_pair.py" "$P" "$K" "$MODEL" "$STAGE" "$ALLOCATION"
sha256sum -c validation.sha256 > "$P/receipts/source-post-$SLURM_JOB_ID.txt"
