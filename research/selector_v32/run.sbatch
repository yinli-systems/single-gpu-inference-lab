#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --job-name=sgi-selector-v32
set -euo pipefail
P=${1:?campaign root}; STAGE=${2:?canary or release}; SHARD=${3:-0}; SHARDS=${4:-1}
[[ "$STAGE" == canary || "$STAGE" == release ]]
[[ "$SLURM_JOB_PARTITION" == gpu_4090 || "$SLURM_JOB_PARTITION" == gpu_5090 ]]
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 PYTHONNOUSERSITE=1
export SGI_OFFICIAL_MANIFEST_SHA256=$(cat "$P/receipts/official-overlay.sha256")
export SGI_SOURCE_ARCHIVE_SHA256=$(cat "$P/receipts/source-archive.sha256")
LOCAL=$(mktemp -d /tmp/sgi-v32-${SLURM_JOB_ID}-XXXXXX)
mkdir -p "$P/receipts" "$P/logs" "$P/runs" "$P/cache/$SLURM_JOB_PARTITION" "$P/refs/$SLURM_JOB_PARTITION/s$SHARD"
cd "$P/source"; sha256sum -c source.sha256
scontrol show job "$SLURM_JOB_ID" -o > "$P/receipts/slurm-${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 2 > "$P/logs/telemetry-${SLURM_JOB_ID}.csv" &
MON=$!
finish(){ RC=$?; trap - EXIT; kill "$MON" 2>/dev/null || true; printf '%s\n' "$RC" > "$P/receipts/exit-${SLURM_JOB_ID}.txt"; rm -rf "$LOCAL"; exit "$RC"; }
trap finish EXIT
REPS="0 1 2"; [[ "$STAGE" == canary ]] && REPS="0"
for REP in $REPS; do
 case "$REP" in 0|2) MODES="pristine paired";; 1) MODES="paired pristine";; *) exit 22;; esac
 for MODE in $MODES; do
  if [[ "$MODE" == pristine ]]; then OVERLAY=pristine; else OVERLAY=guarded; fi
  export PYTHONPATH=$P/overlays/$OVERLAY
  export FLASHINFER_WORKSPACE_BASE=$P/cache/$SLURM_JOB_PARTITION/$OVERLAY
  export XDG_CACHE_HOME=$LOCAL/cache-$MODE-r$REP
  R="$P/runs/$STAGE-$SLURM_JOB_PARTITION-s$SHARD-r$REP-$MODE-$SLURM_JOB_ID"
  "$E/bin/python" -u measure.py --mode "$MODE" --stage "$STAGE" --rep "$REP" --shard "$SHARD" --shards "$SHARDS" --refs "$P/refs/$SLURM_JOB_PARTITION/s$SHARD" --out "$R"
  cp SOURCE_COMMIT.txt "$R/source_commit.txt"; date -u +%FT%TZ > "$R/launcher-complete.txt"
 done
done
