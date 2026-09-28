#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --job-name=sgi-reuse-gate
#SBATCH --output=/ssd/scxi253/single-gpu-inference-plan-cost-20260928/campaigns/reuse-gate-v1/logs/%x-%A_%a.log
set -euo pipefail
P=/ssd/scxi253/single-gpu-inference-plan-cost-20260928
C=$P/campaigns/reuse-gate-v1
OLD=$P/campaigns/resume-20260928T2348Z
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
STAGE=${1:?canary or test required}
REP=${2:-${SLURM_ARRAY_TASK_ID:-0}}
L=$(mktemp -d /tmp/sgi-reuse-${SLURM_JOB_ID}-XXXXXX)
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME=$L/cache TRITON_CACHE_DIR=$L/triton FLASHINFER_WORKSPACE_BASE=$L/flashinfer MAX_JOBS=4
export PYTHONPATH=$C/research/causal_plan_cost
R=$C/runs/$STAGE-${SLURM_JOB_PARTITION}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-$REP
cd "$C"
sha256sum -c source_manifest.sha256
printf 'COMMIT '; cat source_commit.txt
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit --format=csv
"$E/bin/python" -u research/causal_plan_cost/reuse_gate/measure_reuse.py --stage "$STAGE" --replicate "$REP" --blocks 4 --out "$R" --native "$OLD/native" --analysis "$OLD/analysis" --campaign "$OLD"
