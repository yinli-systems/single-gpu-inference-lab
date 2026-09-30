#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --job-name=sgi-selector-v4
set -euo pipefail
P=${1:?campaign root}; STAGE=${2:?dev canary release or stress}; SHARD=${3:?}; SHARDS=${4:?}
[[ "$STAGE" == dev || "$STAGE" == canary || "$STAGE" == release || "$STAGE" == stress ]]
[[ "$SLURM_JOB_PARTITION" == gpu_4090 || "$SLURM_JOB_PARTITION" == gpu_5090 ]]
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 PYTHONNOUSERSITE=1
export SGI_OFFICIAL_MANIFEST_SHA256=$(cat "$P/receipts/official-overlay.sha256")
export SGI_SOURCE_ARCHIVE_SHA256=$(cat "$P/receipts/source-archive.sha256")
LOCAL=$(mktemp -d /tmp/sgi-v4-${SLURM_JOB_ID}-XXXXXX)
mkdir -p "$P/receipts" "$P/logs" "$P/runs" "$P/cache/$SLURM_JOB_PARTITION" "$P/refs/$SLURM_JOB_PARTITION/s$SHARD"
cd "$P/source"; sha256sum -c source.sha256
if [[ "$STAGE" == canary || "$STAGE" == release || "$STAGE" == stress ]]; then
  "$E/bin/python" -m research.selector_v4.authorize_v4 --root "$P" --stage "$STAGE" > "$P/receipts/${STAGE}-authorization-${SLURM_JOB_ID}.json"
fi
"$E/bin/python" -m research.selector_v4.smoke_verdict --root "$P" --verify-only
scontrol show job "$SLURM_JOB_ID" -o > "$P/receipts/slurm-${SLURM_JOB_ID}.txt"
export PYTHONPATH="$P/source"
DEVICE_UUID=$("$E/bin/python" -m research.selector_v4.device_identity)
nvidia-smi --id="$DEVICE_UUID" --query-gpu=timestamp,uuid,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 2 > "$P/logs/telemetry-${SLURM_JOB_ID}.csv" & MON=$!
finish(){ RC=$?; trap - EXIT; kill "$MON" 2>/dev/null || true; printf '%s\n' "$RC" > "$P/receipts/exit-${SLURM_JOB_ID}.txt"; rm -rf "$LOCAL"; exit "$RC"; }; trap finish EXIT
for REP in 0 1 2; do
  case "$REP" in 0|2) MODES="pristine paired";; 1) MODES="paired pristine";; *) exit 22;; esac
  for MODE in $MODES; do
    [[ "$MODE" == pristine ]] && OVERLAY=pristine || OVERLAY=candidate
    export PYTHONPATH="$P/source:$P/overlays/$OVERLAY"
    export FLASHINFER_WORKSPACE_BASE="$P/cache/$SLURM_JOB_PARTITION/$OVERLAY"
    export XDG_CACHE_HOME="$P/cache/$SLURM_JOB_PARTITION/$OVERLAY/xdg-$MODE-r$REP"
    mkdir -p "$XDG_CACHE_HOME"
    OUT="$P/runs/$STAGE-$SLURM_JOB_PARTITION-s$SHARD-r$REP-$MODE-$SLURM_JOB_ID"
    srun --cpu-bind=cores "$E/bin/python" -m research.selector_v4.measure_v4 --mode "$MODE" --stage "$STAGE" --rep "$REP" --shard "$SHARD" --shards "$SHARDS" --refs "$P/refs/$SLURM_JOB_PARTITION/s$SHARD" --out "$OUT"
    cp "$P/source/SOURCE_COMMIT.txt" "$OUT/source_commit.txt"; date -u +%FT%TZ > "$OUT/launcher-complete.txt"
  done
done

# The candidate JIT must contain co-resident native/resource symbols.
export PYTHONPATH="$P/source:$P/overlays/candidate"
"$E/bin/python" -m research.selector_v4.audit_binary \
  --workspace "$P/cache/$SLURM_JOB_PARTITION/candidate" --pristine-workspace "$P/cache/$SLURM_JOB_PARTITION/pristine" \
  --overlay "$P/overlays/candidate" \
  --out "$P/receipts/binary-audit-${SLURM_JOB_ID}.json"
