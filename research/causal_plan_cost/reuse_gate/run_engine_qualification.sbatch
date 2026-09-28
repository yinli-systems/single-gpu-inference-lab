#!/bin/bash
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --job-name=sgi-reuse-engine
#SBATCH --output=/ssd/scxi253/single-gpu-inference-plan-cost-20260928/campaigns/reuse-engine-v1/logs/%x-%j.log
set -euo pipefail
P=/ssd/scxi253/single-gpu-inference-plan-cost-20260928
C=$P/campaigns/reuse-engine-v1
OLD=$P/campaigns/resume-20260928T2348Z
REP=${1:-0}
L=$(mktemp -d /tmp/sgi-reuse-engine-${SLURM_JOB_ID}-XXXXXX)
R=$C/runs/job-${SLURM_JOB_ID}
mkdir -p "$R"
cd "$C"
sha256sum -c source_manifest.sha256 | tee "$R/source-checks.txt"
cp source_commit.txt "$R/source_commit.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit --format=csv | tee "$R/hardware.csv"
date -u -Is > "$R/start.txt"
tar -xf /data/run01/scxi253/inference/venv-vllm.tar -C "$L"
tar -xf /data/run01/scxi253/inference/libfix.tar -C "$L"
export VIRTUAL_ENV=$L/venv-vllm
export CUDA_HOME=/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13
export PATH=$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$L/libfix:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=$CUDA_HOME/lib
export PYTHONPATH=$C/research/causal_plan_cost/reuse_gate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4
export FLASHINFER_WORKSPACE_BASE=$L/flashinfer TRITON_CACHE_DIR=$L/triton VLLM_CACHE_ROOT=$L/vllm XDG_CACHE_HOME=$L/cache
python -u research/causal_plan_cost/reuse_gate/qualify_paged.py --native "$OLD/native" --out "$R/paged-qualification" > "$R/paged-qualification.log" 2>&1
python -u - "$C" "$R" "$OLD" "$REP" <<'PY'
import json,pathlib,random,subprocess,sys,time
C,R,OLD=map(pathlib.Path,sys.argv[1:4]);rep=int(sys.argv[4])
qual=json.loads((R/'paged-qualification/summary.json').read_text())
assert qual['passed']
arms=['auto','fixed_budget','native_cycle'];random.Random(20261020+rep).shuffle(arms)
(R/'arm_order.json').write_text(json.dumps(arms))
receipts=[]
for arm in arms:
    cmd=[sys.executable,str(C/'research/causal_plan_cost/reuse_gate/engine_screen.py'),
         '--native',str(OLD/'native'),'--model','/ssd/scxi253/check-before-act-20260927/models/Qwen3-4B-Instruct-2507',
         '--out',str(R/arm),'--arm',arm,'--replicate',str(rep)]
    tick=time.monotonic()
    with (R/(arm+'.log')).open('w') as f:
        try:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=480);rc=r.returncode
        except subprocess.TimeoutExpired:rc=124
    receipts.append(dict(arm=arm,returncode=rc,seconds=time.monotonic()-tick,command=cmd))
    (R/'engine_receipts.json').write_text(json.dumps(receipts,indent=2))
    if rc:raise RuntimeError('Engine qualification failed: '+arm+' exit '+str(rc))
for name in ('burst','staggered'):
    outputs={arm:json.loads((R/arm/(name+'.json')).read_text()) for arm in arms}
    hashes={x['workload_sha256'] for x in outputs.values()};assert len(hashes)==1
    ref={r['id']:r['token_ids'] for r in outputs['auto']['requests_detail']}
    for arm,x in outputs.items():
        actual={r['id']:r['token_ids'] for r in x['requests_detail']}
        if actual!=ref:raise AssertionError('Greedy outputs differ for '+arm+'/'+name)
(R/'complete.json').write_text(json.dumps(dict(paged_qualified=True,all_arms_complete=True,greedy_outputs_identical=True,replicate=rep),indent=2))
print('FULL_ENGINE_QUALIFIED',flush=True)
PY
date -u -Is > "$R/completed.txt"
