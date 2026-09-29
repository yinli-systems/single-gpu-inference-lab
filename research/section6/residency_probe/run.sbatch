#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:08:00
#SBATCH --job-name=sgi-residency
set -euo pipefail
P=${1:?}; KEY=${SLURM_JOB_PARTITION:?}
mkdir "$P/receipts/residency-reservation-$KEY"
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
LOCAL=$(mktemp -d /tmp/sgi-residency-${SLURM_JOB_ID}-XXXXXX)
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4
export XDG_CACHE_HOME=$LOCAL/cache TRITON_CACHE_DIR=$LOCAL/triton FLASHINFER_WORKSPACE_BASE=$LOCAL/flashinfer
export PYTHONPATH=$P/overlays/residency070-v2
cd "$P/residency-v2-source"
sha256sum -c source.sha256 > "$P/receipts/residency-$KEY-source.txt"
scontrol show job "$SLURM_JOB_ID" -o > "$P/receipts/residency-$KEY-slurm.txt"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 2 > "$P/logs/residency-$KEY-telemetry.csv" &
MON=$!
finish(){ RC=$?; trap - EXIT; kill "$MON" 2>/dev/null || true; printf '%s\n' "$RC" > "$P/receipts/residency-$KEY-exit.txt"; exit "$RC"; }
trap finish EXIT
for REP in 0 1 2; do
 R="$P/residency-runs/$KEY-r$REP-$SLURM_JOB_ID"
 "$E/bin/python" -u research/section6/residency_probe/probe.py --rep "$REP" --out "$R"
 cp SOURCE_COMMIT.txt "$R/source_commit.txt"
 date -u +%FT%TZ > "$R/launcher-complete.txt"
done
