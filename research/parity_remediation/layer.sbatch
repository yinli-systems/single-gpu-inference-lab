#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --job-name=sgi-parity-layers
set -euo pipefail
D=${1:?}; KIND=${2:?}; P=/ssd/scxi253/single-gpu-inference-resource-generalization-20260930
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
LOCAL=$(mktemp -d /tmp/sgi-parity-${SLURM_JOB_ID}-XXXXXX)
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4
CCCL=$P/overlays/pristine/flashinfer/data/cccl
export CPATH=$CCCL/libcudacxx/include:$CCCL/cub:$CCCL/thrust
export NVCC_PREPEND_FLAGS="-I$CCCL/libcudacxx/include -I$CCCL/cub -I$CCCL/thrust"
export SGLANG_JIT_CACHE_DIR=$D/sglang-cache/$SLURM_JOB_ID/jit SGLANG_CACHE_DIR=$LOCAL/runtime
export SGLANG_CUTE_AOT_CACHE_DIR=$LOCAL/cute TRITON_CACHE_DIR=$LOCAL/triton
R=$D/runs/$KIND-$SLURM_JOB_ID; mkdir -p "$R"
cd "$D/source";sha256sum -c source.sha256
scontrol show job "$SLURM_JOB_ID" -o > "$R/slurm.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,power.limit --format=csv > "$R/hardware.csv"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 2 > "$R/telemetry.csv" &
MON=$!
finish(){ RC=$?; trap - EXIT; kill "$MON" 2>/dev/null || true; printf '%s\n' "$RC" > "$R/exit.txt"; exit "$RC"; }
trap finish EXIT
EXTRA=(); if [ "$KIND" = deterministic ]; then EXTRA=(--deterministic); elif [ "$KIND" != standard ]; then exit 2; fi
for REP in 0; do
 for MODE in pristine cap; do
  export PYTHONPATH=$P/overlays/$MODE:$P/source
  export FLASHINFER_WORKSPACE_BASE=$P/cache/$SLURM_JOB_PARTITION/$MODE
  export XDG_CACHE_HOME=$LOCAL/cache-$MODE-$KIND
  "$E/bin/python" layer_http.py --historic-root "$P" --out "$R/$MODE-$REP" --mode "$MODE" --rep "$REP" "${EXTRA[@]}"
 done
done
date -u +%FT%TZ > "$R/completed_utc.txt"
