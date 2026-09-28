#!/bin/bash
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --job-name=sgi-full-v2
#SBATCH --output=/ssd/scxi253/single-gpu-inference-plan-cost-20260928/campaigns/full-engine-v2/logs/%x-%A_%a.log
set -euo pipefail
P=/ssd/scxi253/single-gpu-inference-plan-cost-20260928
C=$P/campaigns/full-engine-v2
OLD=$P/campaigns/resume-20260928T2348Z
STAGE=${1:?screen or test required}
REP=${2:-${SLURM_ARRAY_TASK_ID:-0}}
case "$STAGE" in screen|test) ;; *) exit 2 ;; esac
L=$(mktemp -d /tmp/sgi-full-v2-${SLURM_JOB_ID}-XXXXXX)
R=$C/runs/$STAGE-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-$REP
mkdir "$R"
cd "$C"
sha256sum -c source_manifest.sha256 > "$R/source-checks.txt"
cp source_commit.txt "$R/source_commit.txt"
date -u -Is > "$R/start.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit --format=csv > "$R/hardware.csv"
tar -xf /data/run01/scxi253/inference/venv-vllm.tar -C "$L"
tar -xf /data/run01/scxi253/inference/libfix.tar -C "$L"
export VIRTUAL_ENV=$L/venv-vllm
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
NVIDIA=$VIRTUAL_ENV/lib/python3.13/site-packages/nvidia/cu13
test -f "$NVIDIA/include/curand.h" && test -f "$NVIDIA/include/curand_kernel.h"
export PATH=$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$NVIDIA/lib:$L/libfix:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=$NVIDIA/lib:$CUDA_HOME/lib
export CPATH=$NVIDIA/include
export NVCC_PREPEND_FLAGS="-I$NVIDIA/include"
export PYTHONPATH=$C/research/causal_plan_cost/reuse_gate:$C/research/causal_plan_cost/reuse_gate/full_engine_v2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4
export FLASHINFER_WORKSPACE_BASE=$L/flashinfer TRITON_CACHE_DIR=$L/triton VLLM_CACHE_ROOT=$L/vllm XDG_CACHE_HOME=$L/cache
sha256sum "$NVIDIA/include/curand.h" "$NVIDIA/include/curand_kernel.h" > "$R/header_hashes.txt"
printf '#include <curand.h>\n#include <curand_kernel.h>\nint main(){return 0;}\n' > "$L/header_probe.cu"
nvcc -std=c++17 -c "$L/header_probe.cu" -o "$L/header_probe.o" > "$R/header_probe.log" 2>&1
python -m unittest -v test_measure_engine > "$R/cpu_tests.txt" 2>&1
python -u - "$C" "$R" "$OLD" "$REP" "$STAGE" <<'PY'
import hashlib,itertools,json,pathlib,subprocess,sys,time
C,R,OLD=map(pathlib.Path,sys.argv[1:4]);rep=int(sys.argv[4]);stage=sys.argv[5]
ARMS=('auto','fixed_budget','native_cycle')
orders=list(itertools.permutations(ARMS));order=orders[rep%len(orders)]
(R/'arm_order.json').write_text(json.dumps(order))
receipts=[]
for arm in order:
    cmd=[sys.executable,str(C/'research/causal_plan_cost/reuse_gate/full_engine_v2/measure_engine.py'),
         '--native',str(OLD/'native'),'--model','/ssd/scxi253/check-before-act-20260927/models/Qwen3-4B-Instruct-2507',
         '--out',str(R/arm),'--arm',arm,'--replicate',str(rep)]
    tick=time.monotonic()
    with (R/(arm+'.log')).open('w') as f:
        try:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=540).returncode
        except subprocess.TimeoutExpired:rc=124
    receipts.append(dict(arm=arm,returncode=rc,seconds=time.monotonic()-tick,command=cmd))
    (R/'receipts.json').write_text(json.dumps(receipts,indent=2))
    if rc:raise RuntimeError('Model arm failed: '+arm+' exit '+str(rc)+'; retain evidence and stop this job')
parity={}
for name in ('burst','staggered'):
    outputs={arm:json.loads((R/arm/('measured-'+name+'.json')).read_text()) for arm in ARMS}
    if len({x['workload_sha256'] for x in outputs.values()})!=1:raise AssertionError('workload changed')
    ref={r['id']:r['token_ids'] for r in outputs['auto']['requests_detail']}
    parity[name]={}
    for arm,x in outputs.items():
        actual={r['id']:r['token_ids'] for r in x['requests_detail']}
        parity[name][arm]=dict(identical=actual==ref,request_count=len(actual))
(R/'token_parity.json').write_text(json.dumps(parity,indent=2))
passed=all(x['identical'] for w in parity.values() for x in w.values())
summary=dict(stage=stage,replicate=rep,all_arms_complete=True,greedy_outputs_identical=passed,
    performance_promotion=False,qualification_only=(stage=='screen'))
(R/'complete.json').write_text(json.dumps(summary,indent=2))
print('FULL_MODEL_JOB_COMPLETE',json.dumps(summary),flush=True)
if not passed:raise AssertionError('Strict greedy parity gate failed; no performance promotion')
PY
date -u -Is > "$R/completed.txt"
