#!/bin/bash
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:40:00
#SBATCH --job-name=sgi-full-resume
#SBATCH --output=/ssd/scxi253/single-gpu-inference-plan-cost-20260928/campaigns/reuse-engine-resume-v1/logs/%x-%A_%a.log
set -euo pipefail
P=/ssd/scxi253/single-gpu-inference-plan-cost-20260928
C=$P/campaigns/reuse-engine-resume-v1
OLD=$P/campaigns/resume-20260928T2348Z
REP=${1:-${SLURM_ARRAY_TASK_ID:-0}}
L=$(mktemp -d /tmp/sgi-full-resume-${SLURM_JOB_ID}-XXXXXX)
R=$C/runs/rep${REP}-job${SLURM_JOB_ID}
mkdir -p "$C/runs"
cd "$C"
sha256sum -c source_manifest.sha256
printf 'SOURCE_COMMIT '; cat source_commit.txt
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit --format=csv
date -u -Is
tar -xf /data/run01/scxi253/inference/venv-vllm.tar -C "$L"
tar -xf /data/run01/scxi253/inference/libfix.tar -C "$L"
export VIRTUAL_ENV=$L/venv-vllm
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH
export SGI_CUDA_INCLUDE=$VIRTUAL_ENV/lib/python3.13/site-packages/nvidia/cu13/include
export CPATH=$SGI_CUDA_INCLUDE
export CPLUS_INCLUDE_PATH=$SGI_CUDA_INCLUDE
export LD_LIBRARY_PATH=$L/libfix:$CUDA_HOME/lib:$VIRTUAL_ENV/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=$CUDA_HOME/lib:$VIRTUAL_ENV/lib/python3.13/site-packages/nvidia/cu13/lib
export PYTHONPATH=$C/research/causal_plan_cost/reuse_gate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4
export FLASHINFER_WORKSPACE_BASE=$L/flashinfer TRITON_CACHE_DIR=$L/triton VLLM_CACHE_ROOT=$L/vllm XDG_CACHE_HOME=$L/cache
python -u research/causal_plan_cost/reuse_gate/resume_engine.py \
  --campaign "$C" --out "$R" --replicate "$REP" --native "$OLD/native" \
  --old-paged "$P/campaigns/reuse-engine-v1/runs/job-1632883/paged-qualification/summary.json" \
  --model /ssd/scxi253/check-before-act-20260927/models/Qwen3-4B-Instruct-2507
