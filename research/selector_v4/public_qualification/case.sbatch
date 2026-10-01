#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --job-name=sgi-public-v43
set -euo pipefail
P=${1:?campaign};S=${2:?source};E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 PYTHONNOUSERSITE=1
CCCL=$S/flashinfer/data/cccl
export CPATH=$P/sdk-libraries:$CCCL/libcudacxx/include:$CCCL/cub:$CCCL/thrust
export NVCC_PREPEND_FLAGS="-I$P/sdk-libraries -I$CCCL/libcudacxx/include -I$CCCL/cub -I$CCCL/thrust"
export FLASHINFER_WORKSPACE_BASE=$P/cache/$SLURM_JOB_ID
export XDG_CACHE_HOME=$P/cache/xdg-$SLURM_JOB_ID PYTHONPATH=$S
cd "$S"; sha256sum -c source.sha256 > "$P/receipts/source-verification-$SLURM_JOB_ID.txt"
cd "$P"; sha256sum -c validation.sha256 > "$P/receipts/validation-verification-$SLURM_JOB_ID.txt"
UUID=$("$E/bin/python" -c 'import torch; u=str(torch.cuda.get_device_properties(0).uuid); print(u if u.startswith("GPU-") else "GPU-"+u)')
nvidia-smi --id="$UUID" --query-gpu=timestamp,uuid,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 2 > "$P/logs/telemetry-$SLURM_JOB_ID.csv" & MON=$!
finish(){ RC=$?;trap - EXIT;kill "$MON" 2>/dev/null || true;printf '%s\n' "$RC" > "$P/receipts/exit-$SLURM_JOB_ID.txt";exit "$RC"; };trap finish EXIT
CASE=${3:?source-bound stage case index}; PRISTINE=${4:?independent pristine source}
ROOT=$P/runs/$SLURM_JOB_PARTITION-$SLURM_JOB_ID
mkdir "$ROOT"
for IDENTITY in pristine-0 train-0 train-1 pristine-1 pristine-2 train-2 policy-0 policy-1 policy-2; do
  ROLE=${IDENTITY%-*}; REP=${IDENTITY##*-}
  if [[ "$ROLE" == pristine ]]; then CURRENT_SOURCE=$PRISTINE; else CURRENT_SOURCE=$S; fi
  export PYTHONPATH=$P/harness/research/selector_v4/public_qualification:$P/harness:$CURRENT_SOURCE
  CURRENT_CCCL=$CURRENT_SOURCE/flashinfer/data/cccl
  export CPATH=$P/sdk-libraries:$CURRENT_CCCL/libcudacxx/include:$CURRENT_CCCL/cub:$CURRENT_CCCL/thrust
  export NVCC_PREPEND_FLAGS="-I$P/sdk-libraries -I$CURRENT_CCCL/libcudacxx/include -I$CURRENT_CCCL/cub -I$CURRENT_CCCL/thrust"
  export FLASHINFER_WORKSPACE_BASE=$P/cache/$SLURM_JOB_ID/$IDENTITY
  export XDG_CACHE_HOME=$P/cache/xdg-$SLURM_JOB_ID/$IDENTITY
  cd "$CURRENT_SOURCE"
  sha256sum -c source.sha256 > "$P/receipts/source-$SLURM_JOB_ID-$IDENTITY.txt"
  srun --cpu-bind=cores "$E/bin/python" "$P/harness/research/selector_v4/public_qualification/measure.py" --campaign "$P" --out "$ROOT/$IDENTITY" --role "$ROLE" --rep "$REP" --case "$CASE" > "$P/logs/$SLURM_JOB_ID-$IDENTITY.log" 2>&1
done
for CURRENT_SOURCE in "$S" "$PRISTINE"; do
 cd "$CURRENT_SOURCE"
 sha256sum -c source.sha256 > "$P/receipts/source-post-$SLURM_JOB_ID-$(basename "$CURRENT_SOURCE").txt"
done
