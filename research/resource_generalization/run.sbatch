#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --job-name=sgi-resource-v7
set -euo pipefail
P=${1:?}; STAGE=${2:?}; SHARD=${3:-0}; SHARDS=${4:-1}
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
LOCAL=$(mktemp -d /tmp/sgi-resource-${SLURM_JOB_ID}-XXXXXX)
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4
export TRITON_CACHE_DIR=$LOCAL/triton
cd "$P/source"; sha256sum -c source.sha256
scontrol show job "$SLURM_JOB_ID" -o > "$P/receipts/slurm-${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 2 > "$P/logs/telemetry-${SLURM_JOB_ID}.csv" &
MON=$!
finish(){ RC=$?; trap - EXIT; kill "$MON" 2>/dev/null || true; printf '%s\n' "$RC" > "$P/receipts/exit-${SLURM_JOB_ID}.txt"; exit "$RC"; }
trap finish EXIT
REPS="0 1 2"; [[ "$STAGE" == canary ]] && REPS="0"
for REP in $REPS; do
 case "$REP" in 0) MODES="pristine off cap wide";; 1) MODES="cap wide pristine off";; 2) MODES="wide off cap pristine";; esac
 for MODE in $MODES; do
  export PYTHONPATH=$P/overlays/$MODE
  export FLASHINFER_WORKSPACE_BASE=$P/cache/$SLURM_JOB_PARTITION/$MODE
  export XDG_CACHE_HOME=$LOCAL/cache-$MODE
  R="$P/runs/$STAGE-$SLURM_JOB_PARTITION-s$SHARD-r$REP-$MODE-$SLURM_JOB_ID"
  "$E/bin/python" -u measure.py --mode "$MODE" --stage "$STAGE" --rep "$REP" --shard "$SHARD" --shards "$SHARDS" --refs "$LOCAL/refs" --out "$R"
  cp SOURCE_COMMIT.txt "$R/source_commit.txt"; date -u +%FT%TZ > "$R/launcher-complete.txt"
 done
done
