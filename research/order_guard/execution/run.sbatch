#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:25:00
#SBATCH --job-name=sgi-order-guard
set -euo pipefail
P=${1:?campaign root}; STAGE=${2:?stage}; TASK=${SLURM_ARRAY_TASK_ID:-0}
if [[ "$STAGE" == formal ]]; then REP=$((TASK/4)); SHARD=$((TASK%4)); SHARDS=4; test -f "$P/CANARY_PASS.json"; else REP=0; SHARD=0; SHARDS=1; fi
KEY=${STAGE}-${SLURM_JOB_PARTITION}-r${REP}-s${SHARD}
mkdir "$P/reservations/$KEY"
if [[ -f "$P/HOLD.json" ]]; then echo 'Campaign is on HOLD'; exit 90; fi
R=$P/runs/${KEY}-${SLURM_JOB_ID}
LOCAL=$(mktemp -d /tmp/sgi-order-guard-${SLURM_JOB_ID}-XXXXXX)
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4
export XDG_CACHE_HOME=$LOCAL/cache TRITON_CACHE_DIR=$LOCAL/triton FLASHINFER_WORKSPACE_BASE=$LOCAL/flashinfer
cd "$P/source"
sha256sum -c source.sha256
scontrol show job "$SLURM_JOB_ID" -o > "$P/receipts/$KEY-slurm.txt"
printf '%s
' "$R" > "$P/receipts/$KEY-output-path.txt"
date -u +%FT%TZ > "$P/receipts/$KEY-started.txt"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu --format=csv -l 2 > "$P/logs/$KEY-telemetry.csv" &
MON=$!
finish() {
 RC=$?; trap - EXIT; kill "$MON" 2>/dev/null || true
 printf '%s
' "$RC" > "$P/receipts/$KEY-exit-code.txt"
 date -u +%FT%TZ > "$P/receipts/$KEY-finished.txt"
 if [[ "$RC" != 0 ]]; then printf '{"hold":true,"run":"%s","exit_code":%s}
' "$KEY" "$RC" > "$P/HOLD.json"; fi
 exit "$RC"
}
trap finish EXIT
$E/bin/python -u research/order_guard/gpu_probe.py --stage "$STAGE" --rep "$REP" --shard "$SHARD" --shards "$SHARDS" --out "$R"
cp SOURCE_COMMIT.txt "$R/source_commit.txt"
date -u +%FT%TZ > "$R/completed_utc.txt"
echo "GUARD_GPU_COMPLETE $R"
