#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --job-name=sgi-decision-collision
set -euo pipefail

P=${1:?campaign root}
STAGE=${2:?canary or formal}
REP=${3:?repeat}
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
LOCAL=$(mktemp -d /tmp/sgi-decision-collision-${SLURM_JOB_ID}-XXXXXX)

export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MAX_JOBS=4

cd "$P"
sha256sum -c source.sha256 > "receipts/source-check-${SLURM_JOB_ID}.txt"
scontrol show job "$SLURM_JOB_ID" -o > "receipts/slurm-${SLURM_JOB_ID}.txt"
date -u +%FT%TZ > "receipts/start-${SLURM_JOB_ID}.txt"

nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 2 > "logs/telemetry-${SLURM_JOB_ID}.csv" &
MON=$!

finish() {
  RC=$?
  trap - EXIT
  kill "$MON" 2>/dev/null || true
  printf '%s\n' "$RC" > "receipts/exit-${SLURM_JOB_ID}.txt"
  date -u +%FT%TZ > "receipts/end-${SLURM_JOB_ID}.txt"
  exit "$RC"
}
trap finish EXIT

if [[ "$STAGE" == "canary" ]]; then
  [[ "$REP" == "0" ]]
  MODES="pristine cap"
elif [[ "$STAGE" == "formal" ]]; then
  case "$REP" in
    0|3) MODES="pristine cap" ;;
    1|2) MODES="cap pristine" ;;
    *) echo "formal repeat must be 0..3" >&2; exit 2 ;;
  esac
else
  echo "unknown stage" >&2
  exit 2
fi

for MODE in $MODES; do
  export XDG_CACHE_HOME="$LOCAL/cache-$MODE"
  export TRITON_CACHE_DIR="$LOCAL/triton-$MODE"
  export FLASHINFER_WORKSPACE_BASE="$P/cache/${SLURM_JOB_PARTITION}/$MODE"
  if [[ "$MODE" == "cap" ]]; then
    export PYTHONPATH="$P/overlays/cap:$P/source-tree/src"
  else
    export PYTHONPATH="$P/source-tree/src"
  fi

  R="$P/runs/${STAGE}-${SLURM_JOB_PARTITION}-r${REP}-${MODE}-${SLURM_JOB_ID}"
  "$E/bin/python" -u     "$P/source-tree/research/decision_sufficiency/collision_factorial/measure.py"     --stage "$STAGE"     --resource-mode "$MODE"     --rep "$REP"     --out "$R"
  cp SOURCE_COMMIT.txt "$R/source_commit.txt"
  date -u +%FT%TZ > "$R/launcher-complete.txt"
done
