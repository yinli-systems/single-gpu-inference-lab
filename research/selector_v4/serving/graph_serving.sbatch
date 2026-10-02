#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --job-name=sgi-native-graph-http
set -euo pipefail
TASK_ROOT=${1:?owned diagnostic root}
TASK_MODEL=${2:?model}
TASK_MEMORY=${3:?memory fraction}
TASK_ENV=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
TASK_SOURCE=/ssd/scxi253/sgi-official-wheel-validation-75544a17-20261001/wheel-overlay
TASK_SGLANG=/ssd/scxi253/sgi-http-ownership-current-main-3825c5a-post1/candidate
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$TASK_ENV/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$TASK_ROOT/harness:$TASK_SGLANG/python:$TASK_SOURCE
export XDG_CACHE_HOME=$TASK_ROOT/cache/xdg-$SLURM_JOB_ID
export CPATH=/ssd/scxi253/sgi-public-complete-r2-0a5c735-20261002/sdk-libraries:$TASK_SOURCE/flashinfer/data/cccl/libcudacxx/include:$TASK_SOURCE/flashinfer/data/cccl/cub:$TASK_SOURCE/flashinfer/data/cccl/thrust
export NVCC_PREPEND_FLAGS="-I/ssd/scxi253/sgi-public-complete-r2-0a5c735-20261002/sdk-libraries -I$TASK_SOURCE/flashinfer/data/cccl/libcudacxx/include -I$TASK_SOURCE/flashinfer/data/cccl/cub -I$TASK_SOURCE/flashinfer/data/cccl/thrust"
finish(){ TASK_RC=$?;trap - EXIT;printf '%s\n' "$TASK_RC" > "$TASK_ROOT/receipts/exit-$SLURM_JOB_ID.txt";exit "$TASK_RC"; };trap finish EXIT
cd "$TASK_ROOT"
sha256sum -c validation.sha256 > "$TASK_ROOT/receipts/source-pre-$SLURM_JOB_ID.txt"
cd "$TASK_SOURCE"
sha256sum -c source.sha256 > "$TASK_ROOT/receipts/native-pre-$SLURM_JOB_ID.txt"
cd "$TASK_ROOT/harness"
srun --cpu-bind=cores "$TASK_ENV/bin/python" -m research.selector_v4.serving.graph_serving_probe --out "$TASK_ROOT/runs/$SLURM_JOB_PARTITION-$SLURM_JOB_ID" --model-binding "$TASK_ROOT/models/$TASK_MODEL.json" --model-id "$TASK_MODEL" --memory "$TASK_MEMORY"
cd "$TASK_ROOT"
sha256sum -c validation.sha256 > "$TASK_ROOT/receipts/source-post-$SLURM_JOB_ID.txt"
cd "$TASK_SOURCE"
sha256sum -c source.sha256 > "$TASK_ROOT/receipts/native-post-$SLURM_JOB_ID.txt"
