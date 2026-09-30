#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:45:00
set -euo pipefail
P=${1:?campaign root};E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH;export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 PYTHONNOUSERSITE=1
mkdir -p "$P/refs/$SLURM_JOB_PARTITION" "$P/runs/$SLURM_JOB_PARTITION" "$P/cache/$SLURM_JOB_PARTITION" "$P/receipts"
cd "$P/source";sha256sum -c source.sha256
export FLASHINFER_WORKSPACE_BASE="$P/cache/$SLURM_JOB_PARTITION/pristine";export XDG_CACHE_HOME="$FLASHINFER_WORKSPACE_BASE/xdg";export PYTHONPATH="$P/source:$P/overlays/pristine"
"$E/bin/python" -m research.selector_v4.smoke_gpu --mode pristine --refs "$P/refs/$SLURM_JOB_PARTITION" --out "$P/runs/$SLURM_JOB_PARTITION/pristine-$SLURM_JOB_ID"
export FLASHINFER_WORKSPACE_BASE="$P/cache/$SLURM_JOB_PARTITION/candidate";export XDG_CACHE_HOME="$FLASHINFER_WORKSPACE_BASE/xdg";export PYTHONPATH="$P/source:$P/overlays/candidate"
"$E/bin/python" -m research.selector_v4.smoke_gpu --mode candidate --refs "$P/refs/$SLURM_JOB_PARTITION" --out "$P/runs/$SLURM_JOB_PARTITION/candidate-$SLURM_JOB_ID"
"$E/bin/python" -m research.selector_v4.audit_binary --workspace "$P/cache/$SLURM_JOB_PARTITION/candidate" --overlay "$P/overlays/candidate" --out "$P/receipts/binary-audit-$SLURM_JOB_ID.json"
