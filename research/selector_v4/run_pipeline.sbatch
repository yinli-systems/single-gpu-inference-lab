#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
set -euo pipefail
R=${1:?root};STAGE=${2:?stage};SHARD=${3:?shard};SHARDS=${4:?shards}
[[ "$STAGE" == canary || "$STAGE" == release ]]
[[ "$SLURM_JOB_PARTITION" == gpu_4090 || "$SLURM_JOB_PARTITION" == gpu_5090 ]]
[[ "$SHARD" =~ ^[0-9]+$ && "$SHARDS" =~ ^[0-9]+$ && "$SHARD" -lt "$SHARDS" && "$SHARDS" -le 16 ]]
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 PYTHONNOUSERSITE=1
export PYTHONPATH=$R/overlays/v4
export FLASHINFER_WORKSPACE_BASE=$R/cache/$SLURM_JOB_PARTITION/v4
export XDG_CACHE_HOME=$FLASHINFER_WORKSPACE_BASE/xdg
export SGI_SOURCE_ARCHIVE_SHA256=$(cat $R/receipts/source-archive.sha256)
mkdir -p "$R/runs" "$R/refs/$SLURM_JOB_PARTITION/s$SHARD" "$R/decisions/$SLURM_JOB_PARTITION" "$R/analysis/$SLURM_JOB_PARTITION" "$R/logs" "$R/receipts"
cd "$R/source";sha256sum -c source.sha256
scontrol show job "$SLURM_JOB_ID" -o > "$R/receipts/slurm-$SLURM_JOB_ID.txt"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu --format=csv -l 1 > "$R/logs/telemetry-$SLURM_JOB_ID.csv" & MON=$!
finish(){ rc=$?;trap - EXIT;kill "$MON" 2>/dev/null||true;printf '%s\n' "$rc" > "$R/receipts/exit-$SLURM_JOB_ID.txt";date -u +%FT%TZ > "$R/receipts/finished-$SLURM_JOB_ID.txt";exit "$rc";};trap finish EXIT
for REP in 0 1 2; do
 OUT=$R/runs/$STAGE-$SLURM_JOB_PARTITION-s$SHARD-r$REP-calibration-$SLURM_JOB_ID
 $E/bin/python -u measure.py --mode calibration --stage $STAGE --rep $REP --shard $SHARD --shards $SHARDS --refs $R/refs/$SLURM_JOB_PARTITION/s$SHARD --out $OUT
done
$E/bin/python audit_binary.py --workspace $FLASHINFER_WORKSPACE_BASE --out $R/receipts/binary-audit-$SLURM_JOB_ID.json
DEC=$R/decisions/$SLURM_JOB_PARTITION/s$SHARD.json
$E/bin/python analyze_calibration.py --root $R --stage $STAGE --gpu $SLURM_JOB_PARTITION --shard $SHARD --shards $SHARDS --telemetry $R/logs/telemetry-$SLURM_JOB_ID.csv --out $DEC
for REP in 0 1 2; do
 OUT=$R/runs/$STAGE-$SLURM_JOB_PARTITION-s$SHARD-r$REP-evaluation-$SLURM_JOB_ID
 $E/bin/python -u measure.py --mode evaluation --stage $STAGE --rep $REP --shard $SHARD --shards $SHARDS --refs $R/refs/$SLURM_JOB_PARTITION/s$SHARD --decisions $DEC --out $OUT
done
$E/bin/python analyze_evaluation.py --root $R --stage $STAGE --gpu $SLURM_JOB_PARTITION --shard $SHARD --shards $SHARDS --decisions $DEC --out $R/analysis/$SLURM_JOB_PARTITION/s$SHARD.json
