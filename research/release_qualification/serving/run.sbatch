#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:15:00
#SBATCH --job-name=sgi-release-http
set -euo pipefail
P=${1:?}; MODEL_ID=${2:?}; STAGE=${3:?}; REP=${4:-0}
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
LOCAL=$(mktemp -d /tmp/sgi-release-http-${SLURM_JOB_ID}-XXXXXX)
readarray -t MODEL_INFO < <("$E/bin/python" - "$P/source/serving/models.json" "$MODEL_ID" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))[sys.argv[2]]
print(x['path']);print(x['mem_fraction'])
PY
)
MODEL=${MODEL_INFO[0]}; MEM=${MODEL_INFO[1]}
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4
export TRITON_CACHE_DIR=$LOCAL/triton
CCCL=$P/overlays/pristine/flashinfer/data/cccl
export CPATH=$CCCL/libcudacxx/include:$CCCL/cub:$CCCL/thrust
export NVCC_PREPEND_FLAGS="-I$CCCL/libcudacxx/include -I$CCCL/cub -I$CCCL/thrust"
cd "$P/source"; sha256sum -c source.sha256
mkdir -p "$P/serving-runs/$SLURM_JOB_ID" "$P/logs" "$P/cache/$SLURM_JOB_PARTITION/$MODEL_ID" "$P/sglang-cache/$SLURM_JOB_PARTITION/$MODEL_ID"
R="$P/serving-runs/$SLURM_JOB_ID"
scontrol show job "$SLURM_JOB_ID" -o > "$R/slurm.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,power.limit --format=csv > "$R/hardware.csv"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 2 > "$R/telemetry.csv" & MON=$!
finish(){ RC=$?; trap - EXIT; kill "$MON" 2>/dev/null || true; printf '%s\n' "$RC" > "$R/exit.txt"; exit "$RC"; }; trap finish EXIT
"$E/bin/python" - "$MODEL" "$R/model-binding.json" <<'PY'
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]);files=[root/'config.json']
idx=root/'model.safetensors.index.json'
if idx.exists(): files.append(idx);files+=sorted(root.glob('model-*.safetensors'))
else: files+=sorted(root.glob('*.safetensors'))
out={}
for p in files:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(32*1024**2),b''):h.update(b)
 out[p.name]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
Path(sys.argv[2]).write_text(json.dumps(out,indent=2)+'\n')
PY
case "$REP" in
 0) MODES="pristine off cap guarded";;
 1) MODES="guarded cap pristine off";;
 2) MODES="cap off guarded pristine";;
 *) exit 2;;
esac
for MODE in $MODES; do
 export PYTHONPATH=$P/overlays/$MODE:$P/source
 export FLASHINFER_WORKSPACE_BASE=$P/cache/$SLURM_JOB_PARTITION/$MODEL_ID/$MODE
 export SGLANG_JIT_CACHE_DIR=$P/sglang-cache/$SLURM_JOB_PARTITION/$MODEL_ID/$MODE/jit
 export SGLANG_CACHE_DIR=$P/sglang-cache/$SLURM_JOB_PARTITION/$MODEL_ID/$MODE/runtime
 export SGLANG_CUTE_AOT_CACHE_DIR=$P/sglang-cache/$SLURM_JOB_PARTITION/$MODEL_ID/$MODE/cute-aot
 export XDG_CACHE_HOME=$LOCAL/cache-$MODE
 "$E/bin/python" -u serving/http_bench.py --model "$MODEL" --model-id "$MODEL_ID" --mem-fraction "$MEM" --out "$R/$MODE" --mode "$MODE" --stage "$STAGE" --rep "$REP"
done
date -u +%FT%TZ > "$R/launcher-complete.txt"
