#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:35:00
set -euo pipefail
R=${1:?root}
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 PYTHONNOUSERSITE=1
mkdir -p "$R/refs/$SLURM_JOB_PARTITION" "$R/runs/$SLURM_JOB_PARTITION" "$R/cache/$SLURM_JOB_PARTITION"
export FLASHINFER_WORKSPACE_BASE="$R/cache/$SLURM_JOB_PARTITION/pristine"
export XDG_CACHE_HOME="$FLASHINFER_WORKSPACE_BASE/xdg"
export PYTHONPATH=/ssd/scxi253/single-gpu-inference-selector-v321-native-20260930T100521Z/overlays/pristine
"$E/bin/python" -u "$R/source/smoke_gpu.py" --mode pristine --refs "$R/refs/$SLURM_JOB_PARTITION" --out "$R/runs/$SLURM_JOB_PARTITION/pristine-$SLURM_JOB_ID"
export FLASHINFER_WORKSPACE_BASE="$R/cache/$SLURM_JOB_PARTITION/v4"
export XDG_CACHE_HOME="$FLASHINFER_WORKSPACE_BASE/xdg"
export PYTHONPATH="$R/overlays/v4"
"$E/bin/python" -u "$R/source/smoke_gpu.py" --mode v4 --refs "$R/refs/$SLURM_JOB_PARTITION" --out "$R/runs/$SLURM_JOB_PARTITION/v4-$SLURM_JOB_ID"

"$E/bin/python" "$R/source/audit_binary.py" --workspace "$R/cache/$SLURM_JOB_PARTITION/v4" --out "$R/runs/$SLURM_JOB_PARTITION/binary-audit-$SLURM_JOB_ID.json"
