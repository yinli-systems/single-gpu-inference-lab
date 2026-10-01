#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --job-name=sgi-v42-rpc-resume
set -euo pipefail
R=${1:?recovery}; OLD_JOB=${2:?original job}
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 PYTHONNOUSERSITE=1
export SGI_OFFICIAL_MANIFEST_SHA256=$(cat "$R/receipts/official-overlay.sha256")
export SGI_SOURCE_ARCHIVE_SHA256=$(cat "$R/receipts/source-archive.sha256")
export PYTHONPATH="$R/source"
cd "$R/source"; sha256sum -c source.sha256 > "$R/receipts/resume-source-$SLURM_JOB_ID.txt"
# No new Slurm step RPC: both processes inherit the legally allocated batch resources.
"$E/bin/python" "$R/resume_measure.py" "$R" "$OLD_JOB"
