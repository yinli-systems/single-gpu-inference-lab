#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
set -euo pipefail
R=${1:?diagnostic root}
P=/ssd/scxi253/single-gpu-inference-selector-v321-native-20260930T100521Z
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$P/overlays/guarded
export FLASHINFER_WORKSPACE_BASE=$P/cache/$SLURM_JOB_PARTITION/guarded
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1
cd "$P/source"; sha256sum -c source.sha256
cd "$R/source"; sha256sum -c source.sha256
for REP in 0 1 2; do
 "$E/bin/python" -u diagnose.py --out "$R/runs/$SLURM_JOB_PARTITION-r$REP-$SLURM_JOB_ID" --rep "$REP"
done
