#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:40:00
#SBATCH --job-name=sgi-resource-http
set -euo pipefail
P=${1:?}; STAGE=${2:?}; REP=${3:-0}
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
MODEL=/ssd/scxi253/check-before-act-20260927/models/Qwen3-4B-Instruct-2507
LOCAL=$(mktemp -d /tmp/sgi-resource-http-${SLURM_JOB_ID}-XXXXXX)
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4
export TRITON_CACHE_DIR=$LOCAL/triton
cd "$P/serving-source"; sha256sum -c serving.sha256
mkdir "$P/serving-runs/$SLURM_JOB_ID"
R="$P/serving-runs/$SLURM_JOB_ID"
scontrol show job "$SLURM_JOB_ID" -o > "$R/slurm.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,power.limit --format=csv > "$R/hardware.csv"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 2 > "$R/telemetry.csv" &
MON=$!
finish(){ RC=$?; trap - EXIT; kill "$MON" 2>/dev/null || true; printf '%s\n' "$RC" > "$R/exit.txt"; exit "$RC"; }
trap finish EXIT
case "$REP" in 0) MODES="pristine off cap";; 1) MODES="cap pristine off";; 2) MODES="off cap pristine";; *) exit 2;; esac
# Lifecycle reference must always be produced by pristine first.
for MODE in pristine off cap; do
 export PYTHONPATH=$P/overlays/$MODE:$P/source
 export FLASHINFER_WORKSPACE_BASE=$P/cache/$SLURM_JOB_PARTITION/$MODE
 export XDG_CACHE_HOME=$LOCAL/cache-$MODE
 "$E/bin/python" lifecycle.py --out "$R/lifecycle-$MODE" --refs "$LOCAL/lifecycle-refs" --mode "$MODE"
done
# File hashing is startup work, never benchmark time. The complete weights are loaded.
"$E/bin/python" - "$MODEL" "$R/model-binding.json" <<'INNER'
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]);files=[root/'config.json',root/'model.safetensors.index.json']+sorted(root.glob('model-*.safetensors'))
out={}
for p in files:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
 out[p.name]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
Path(sys.argv[2]).write_text(json.dumps(out,indent=2)+'\n')
INNER
for MODE in $MODES; do
 export PYTHONPATH=$P/overlays/$MODE:$P/source
 export FLASHINFER_WORKSPACE_BASE=$P/cache/$SLURM_JOB_PARTITION/$MODE
 export XDG_CACHE_HOME=$LOCAL/cache-$MODE
 "$E/bin/python" http_bench.py --model "$MODEL" --out "$R/$MODE" --mode "$MODE" --stage "$STAGE" --rep "$REP"
done
date -u +%FT%TZ > "$R/launcher-complete.txt"
