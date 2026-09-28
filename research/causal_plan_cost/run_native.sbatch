#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:20:00
#SBATCH --job-name=sgi-native-plan
#SBATCH --output=/ssd/scxi253/single-gpu-inference-plan-cost-20260928/logs/%x-%A_%a.log
set -euo pipefail
P=/ssd/scxi253/single-gpu-inference-plan-cost-20260928
C=$P/campaigns/resume-20260928T2348Z
N=$P/campaigns/native-policy-v1
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
STAGE=${1:?canary or test required}; REP=${2:-${SLURM_ARRAY_TASK_ID:-0}}
LOCAL=$(mktemp -d /tmp/sgi-native-${SLURM_JOB_ID}-XXXXXX)
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME=$LOCAL/cache TRITON_CACHE_DIR=$LOCAL/triton
export FLASHINFER_WORKSPACE_BASE=$LOCAL/flashinfer MAX_JOBS=4
R=$N/runs/$STAGE-${SLURM_JOB_PARTITION}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-$REP
cd "$N/research/causal_plan_cost"
sha256sum -c "$N/source_manifest.sha256"
nvidia-smi --query-gpu=name,uuid,driver_version,power.limit,memory.total --format=csv
"$E/bin/python" -u measure_native.py --out "$R" --replicate "$REP" --stage "$STAGE" --native "$C/native" --analysis "$C/analysis"
