#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:25:00
#SBATCH --job-name=sgi-resource-numeric
set -euo pipefail
P=${1:?}
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
L=$(mktemp -d /tmp/sgi-numerics-${SLURM_JOB_ID}-XXXXXX)
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4
R=$P/numerical-runs/$SLURM_JOB_ID
mkdir "$R"; cd "$P/numerics-source"; sha256sum -c numerics.sha256
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "$R/hardware.csv"
for MODE in pristine off cap wide; do
 export PYTHONPATH=$P/overlays/$MODE:$P/source
 export FLASHINFER_WORKSPACE_BASE=$P/cache/$SLURM_JOB_PARTITION/$MODE
 "$E/bin/python" robustness.py --mode "$MODE" --out "$R/$MODE" --refs "$L/refs"
done
date -u +%FT%TZ > "$R/launcher-complete.txt"
