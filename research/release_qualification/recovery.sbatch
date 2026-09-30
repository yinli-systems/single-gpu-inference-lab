#!/bin/bash
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:25:00
#SBATCH --job-name=sgi-release-recovery
set -euo pipefail
P=${1:?}
E=/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312
EXPECTED_SOURCE=1aa479baa1e7672e3411fabaa6a019ca76269c76
OLD_JOB=1639982
OLD_REL=runs/release-gpu_4090-s3-r2-pristine-1639982
MODE=pristine; STAGE=release; SHARD=3; SHARDS=6; REP=2
[[ "$(cat "$P/source/SOURCE_COMMIT.txt")" == "$EXPECTED_SOURCE" ]]
cd "$P/source"; sha256sum -c source.sha256
OLD="$P/$OLD_REL"; [[ -f "$OLD/environment.json" && -f "$OLD/progress.json" && ! -e "$OLD/complete.json" && ! -e "$OLD/failure.json" ]]
STATE=$(sacct -X -j "$OLD_JOB" --format=State -n -P | head -1); [[ "$STATE" == TIMEOUT* ]]
"$E/bin/python" - "$OLD/environment.json" "$OLD/progress.json" <<'PYREC'
import json,sys
e=json.load(open(sys.argv[1]));p=json.load(open(sys.argv[2]))
assert (e['stage'],e['shard'],e['shards'],e['rep'],e['mode'])==('release',3,6,2,'pristine')
assert p['rows']==3456 and p['qualifications']==54
PYREC
for d in "$P"/runs/release-gpu_4090-s3-r2-pristine-*; do
 [[ "$d" == "$OLD" ]] && continue
 [[ ! -e "$d/complete.json" ]] || { echo "complete replacement already exists: $d" >&2; exit 3; }
done
LOCAL=$(mktemp -d /tmp/sgi-release-recovery-${SLURM_JOB_ID}-XXXXXX)
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$CUDA_HOME/bin:$E/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TRITON_CACHE_DIR=$LOCAL/triton
export PYTHONPATH=$P/overlays/$MODE FLASHINFER_WORKSPACE_BASE=$P/cache/$SLURM_JOB_PARTITION/$MODE XDG_CACHE_HOME=$LOCAL/cache-$MODE
R="$P/runs/$STAGE-$SLURM_JOB_PARTITION-s$SHARD-r$REP-$MODE-$SLURM_JOB_ID"
mkdir -p "$P/receipts" "$P/logs" "$P/cache/$SLURM_JOB_PARTITION"; [[ ! -e "$R" ]]
scontrol show job "$SLURM_JOB_ID" -o > "$P/receipts/slurm-${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 2 > "$P/logs/telemetry-${SLURM_JOB_ID}.csv" & MON=$!
finish(){ RC=$?; trap - EXIT; kill "$MON" 2>/dev/null || true; printf '%s\n' "$RC" > "$P/receipts/exit-${SLURM_JOB_ID}.txt"; exit "$RC"; }; trap finish EXIT
"$E/bin/python" -u measure.py --mode "$MODE" --stage "$STAGE" --rep "$REP" --shard "$SHARD" --shards "$SHARDS" --refs "$LOCAL/refs" --out "$R"
cp SOURCE_COMMIT.txt "$R/source_commit.txt"; date -u +%FT%TZ > "$R/launcher-complete.txt"
"$E/bin/python" - "$P" "$OLD_REL" "$R" "$OLD_JOB" "$STATE" "$EXPECTED_SOURCE" "$0" <<'PYMAP'
from pathlib import Path
import hashlib,json,os,sys
root=Path(sys.argv[1]);old_rel=sys.argv[2];new=Path(sys.argv[3]);old_job=int(sys.argv[4]);state=sys.argv[5]
source=sys.argv[6];script=Path(sys.argv[7]);old=root/old_rel
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rec={'schema':1,'stage':'release','gpu':'gpu_4090','measurement_source_commit':source,'recovery_script_sha256':sha(script),'entries':[{
 'superseded':old_rel,'replacement':new.relative_to(root).as_posix(),'key':{'shard':3,'rep':2,'mode':'pristine'},
 'reason':'Slurm time limit after 3456/4096 rows and 54/64 qualifications; partial retained and never scored',
 'old_job':old_job,'old_state':state,'replacement_job':int(os.environ['SLURM_JOB_ID']),
 'old_environment_sha256':sha(old/'environment.json'),'old_progress_sha256':sha(old/'progress.json'),
 'replacement_complete_sha256':sha(new/'complete.json')}]}
out=root/'receipts'/'recovery-map.json'
if out.exists():raise FileExistsError(out)
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(rec,indent=2)+'\n');tmp.replace(out)
print(json.dumps(rec,indent=2))
PYMAP
