#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --job-name=sgi-v43-smoke
set -euo pipefail
P=${1:?campaign}; S=${2:?source}; E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 PYTHONNOUSERSITE=1
CCCL=$S/flashinfer/data/cccl
SDK=$P/sdk-libraries
export CPATH=$SDK:$CCCL/libcudacxx/include:$CCCL/cub:$CCCL/thrust
export NVCC_PREPEND_FLAGS="-I$SDK -I$CCCL/libcudacxx/include -I$CCCL/cub -I$CCCL/thrust"
export FLASHINFER_WORKSPACE_BASE=$P/cache/$SLURM_JOB_ID
export XDG_CACHE_HOME=$P/cache/xdg-$SLURM_JOB_ID
export PYTHONPATH=$P/harness:$S:$S/test-support PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
mkdir -p "$P/logs" "$P/receipts"
cd "$S"; sha256sum -c source.sha256 > "$P/receipts/source-verification-$SLURM_JOB_ID.txt"
cd "$P"; sha256sum -c validation.sha256 > "$P/receipts/validation-verification-$SLURM_JOB_ID.txt"
cd "$S"
scontrol show job "$SLURM_JOB_ID" -o > "$P/receipts/slurm-$SLURM_JOB_ID.txt"
UUID=$("$E/bin/python" -c 'import torch; raw=str(torch.cuda.get_device_properties(0).uuid); print(raw if raw.startswith("GPU-") else "GPU-"+raw)')
nvidia-smi --id="$UUID" --query-gpu=name,uuid,driver_version,power.limit --format=csv > "$P/receipts/hardware-$SLURM_JOB_ID.csv"
nvidia-smi --id="$UUID" --query-gpu=timestamp,uuid,utilization.gpu,clocks.sm,power.draw,temperature.gpu --format=csv -l 2 > "$P/logs/telemetry-$SLURM_JOB_ID.csv" & MON=$!
finish(){ RC=$?; trap - EXIT; kill "$MON" 2>/dev/null || true; printf '%s\n' "$RC" > "$P/receipts/exit-$SLURM_JOB_ID.txt"; exit "$RC"; };trap finish EXIT
srun --cpu-bind=cores "$E/bin/python" -m pytest --confcutdir=tests/experimental/prefill_resource tests/experimental/prefill_resource -q --junitxml="$P/receipts/tests-$SLURM_JOB_ID.xml"
srun --cpu-bind=cores "$E/bin/python" "$P/harness/research/selector_v4/public_qualification/runtime/prefill_resource_v2_example.py" > "$P/receipts/calibration-$SLURM_JOB_ID.txt"
srun --cpu-bind=cores "$E/bin/python" "$P/harness/research/selector_v4/public_qualification/runtime/main_execution_modes_v2.py" --out "$P/modes-$SLURM_JOB_ID"

cd "$S"
sha256sum -c source.sha256 > "$P/receipts/source-post-$SLURM_JOB_ID.txt"
