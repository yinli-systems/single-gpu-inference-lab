#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --job-name=sgi-selector-v322-release
set -euo pipefail
P=${1:?campaign root}; SHARDS=${2:-10}; SHARD=${SLURM_ARRAY_TASK_ID:?array task required}
[[ "$SLURM_JOB_PARTITION" == gpu_4090 || "$SLURM_JOB_PARTITION" == gpu_5090 ]]
[[ "$SHARD" -ge 0 && "$SHARD" -lt "$SHARDS" ]]
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
PARENT=/ssd/scxi253/single-gpu-inference-selector-v321-native-20260930T100521Z
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 PYTHONNOUSERSITE=1
export SGI_OFFICIAL_MANIFEST_SHA256=$(cat "$P/receipts/official-overlay.sha256")
export SGI_SOURCE_ARCHIVE_SHA256=$(cat "$P/receipts/source-archive.sha256")
LOCAL=$(mktemp -d /tmp/sgi-v322-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}-XXXXXX)
mkdir -p "$P/receipts" "$P/logs" "$P/runs" "$P/refs/$SLURM_JOB_PARTITION/s$SHARD"
cd "$P/source"; sha256sum -c source.sha256
"$E/bin/python" release_v322/authorize_release_v322.py --root "$P" > "$P/receipts/release-preflight-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
scontrol show job "$SLURM_JOB_ID" -o > "$P/receipts/slurm-${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 2 > "$P/logs/telemetry-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" & MON=$!
finish(){ RC=$?; trap - EXIT; kill "$MON" 2>/dev/null || true; printf '%s\n' "$RC" > "$P/receipts/exit-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"; rm -rf "$LOCAL"; exit "$RC"; };trap finish EXIT
for REP in 0 1 2; do
 case "$REP" in 0|2) MODES="pristine paired";; 1) MODES="paired pristine";; *) exit 22;; esac
 for MODE in $MODES; do
  [[ "$MODE" == pristine ]] && OVERLAY=pristine || OVERLAY=guarded
  export PYTHONPATH=$P/overlays/$OVERLAY:$P/source
  export FLASHINFER_WORKSPACE_BASE=$PARENT/cache/$SLURM_JOB_PARTITION/$OVERLAY
  export XDG_CACHE_HOME=$LOCAL/cache-$MODE-r$REP
  R="$P/runs/release-$SLURM_JOB_PARTITION-s$SHARD-r$REP-$MODE-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  "$E/bin/python" -u measure.py --mode "$MODE" --stage release --rep "$REP" --shard "$SHARD" --shards "$SHARDS" --refs "$P/refs/$SLURM_JOB_PARTITION/s$SHARD" --out "$R"
  cp SOURCE_COMMIT.txt "$R/source_commit.txt";date -u +%FT%TZ > "$R/launcher-complete.txt"
 done
done
