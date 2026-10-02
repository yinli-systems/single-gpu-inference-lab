#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --job-name=sgi-graph-epochs
set -euo pipefail
TASK_ROOT=${1:?diagnostic root}
TASK_SOURCE=${2:?frozen normal FlashInfer source}
TASK_ENV=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$TASK_ENV/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH=$TASK_ROOT/harness:$TASK_SOURCE:$TASK_ROOT/test-tooling
export XDG_CACHE_HOME=$TASK_ROOT/cache/xdg-$SLURM_JOB_ID
export FLASHINFER_WORKSPACE_BASE=$TASK_ROOT/cache/$SLURM_JOB_ID
export CPATH=/ssd/scxi253/sgi-public-complete-r2-0a5c735-20261002/sdk-libraries:$TASK_SOURCE/flashinfer/data/cccl/libcudacxx/include:$TASK_SOURCE/flashinfer/data/cccl/cub:$TASK_SOURCE/flashinfer/data/cccl/thrust
export NVCC_PREPEND_FLAGS="-I/ssd/scxi253/sgi-public-complete-r2-0a5c735-20261002/sdk-libraries -I$TASK_SOURCE/flashinfer/data/cccl/libcudacxx/include -I$TASK_SOURCE/flashinfer/data/cccl/cub -I$TASK_SOURCE/flashinfer/data/cccl/thrust"
export SGI_GRAPH_EPOCH_EVIDENCE=$TASK_ROOT/runs/$SLURM_JOB_PARTITION-$SLURM_JOB_ID
finish(){ TASK_RC=$?; trap - EXIT; printf '%s\n' "$TASK_RC" > "$TASK_ROOT/receipts/exit-$SLURM_JOB_ID.txt"; exit "$TASK_RC"; }; trap finish EXIT
cd "$TASK_ROOT/harness"
sha256sum -c source.sha256 > "$TASK_ROOT/receipts/helper-pre-$SLURM_JOB_ID.txt"
cd "$TASK_SOURCE"
sha256sum -c source.sha256 > "$TASK_ROOT/receipts/native-pre-$SLURM_JOB_ID.txt"
cd "$TASK_ROOT"
sha256sum -c sdk.sha256 > "$TASK_ROOT/receipts/sdk-pre-$SLURM_JOB_ID.txt"
sha256sum -c tooling.sha256 > "$TASK_ROOT/receipts/tooling-pre-$SLURM_JOB_ID.txt"
cd "$TASK_ROOT/harness"
srun --cpu-bind=cores "$TASK_ENV/bin/python" -m pytest -q -rs research/selector_v4/exposed_diagnostics/test_graph_epoch.py research/selector_v4/exposed_diagnostics/test_graph_epoch_gpu.py --junitxml="$TASK_ROOT/receipts/tests-$SLURM_JOB_ID.xml" > "$TASK_ROOT/logs/tests-$SLURM_JOB_ID.log" 2>&1
"$TASK_ENV/bin/python" research/selector_v4/public_qualification/runtime/audit_main_sass.py "$TASK_ROOT" "$SLURM_JOB_ID" "$TASK_ROOT/sass-$SLURM_JOB_ID" > "$TASK_ROOT/logs/sass-$SLURM_JOB_ID.log" 2>&1
sha256sum -c source.sha256 > "$TASK_ROOT/receipts/helper-post-$SLURM_JOB_ID.txt"
cd "$TASK_SOURCE"
sha256sum -c source.sha256 > "$TASK_ROOT/receipts/native-post-$SLURM_JOB_ID.txt"
cd "$TASK_ROOT"
sha256sum -c sdk.sha256 > "$TASK_ROOT/receipts/sdk-post-$SLURM_JOB_ID.txt"
sha256sum -c tooling.sha256 > "$TASK_ROOT/receipts/tooling-post-$SLURM_JOB_ID.txt"
