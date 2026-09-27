#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:25:00
#SBATCH --job-name=sgi-causal-plan
#SBATCH --output=/ssd/scxi253/single-gpu-inference-plan-cost-20260928/logs/%x-%A_%a.log
set -euo pipefail
P=${SGI_CAMPAIGN_ROOT:-/ssd/scxi253/single-gpu-inference-plan-cost-20260928}
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
STAGE=${1:?canary or full required}
REP=${2:-${SLURM_ARRAY_TASK_ID:-0}}
LOCAL=$(mktemp -d /tmp/sgi-causal-${SLURM_JOB_ID}-XXXXXX)
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME=$LOCAL/cache TRITON_CACHE_DIR=$LOCAL/triton
export FLASHINFER_WORKSPACE_BASE=$LOCAL/flashinfer MAX_JOBS=4
R=$P/runs/$STAGE-${SLURM_JOB_PARTITION}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-$REP
cd "$P/research/causal_plan_cost"
sha256sum -c "$P/source_manifest.sha256"
"$E/bin/python" -m unittest -v test_geometry
nvidia-smi --query-gpu=name,uuid,driver_version,power.limit,memory.total --format=csv
"$E/bin/python" -u measure.py --out "$R" --replicate "$REP" --stage "$STAGE"
