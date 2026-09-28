"""Bounded actual vLLM continuation. No results are inferred from log filenames."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ORDERS=(('auto','fixed_budget','native_cycle'),('fixed_budget','native_cycle','auto'),('native_cycle','auto','fixed_budget'))
PAGED_SHA='90e34cd509e428cbc596f34e0919bf88e5e5ee0e6a3f366b9520577e4eb93b91'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser()
    for name in ('campaign','out','native','model','old-paged'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--replicate',type=int,required=True);a=p.parse_args()
    if a.replicate not in range(3):raise ValueError('three frozen repetition orders only')
    a.out.mkdir(parents=True,exist_ok=False)
    if sha(a.old_paged)!=PAGED_SHA:raise ValueError('recovered qualification hash mismatch')
    paged=json.loads(a.old_paged.read_text())
    if not paged['passed'] or paged['nonauto_full_layer_transition_checks']!=576:raise ValueError('paged qualification incomplete')
    source=a.campaign/'research/causal_plan_cost/reuse_gate'
    for name in ('paged_router.py','contracts.py'):
        if sha(source/name)!=paged['source_sha256'][name]:raise ValueError('changed adapter invalidates old qualification')
    # This sampling check compiles the same module which failed in the prior job.
    import torch,flashinfer
    torch.manual_seed(20261020)
    x=torch.randn(4,1024,device='cuda',dtype=torch.float32)
    tick=time.perf_counter()
    y=flashinfer.sampling.top_k_top_p_sampling_from_logits(x,1,1.0,deterministic=True)
    torch.cuda.synchronize()
    if not torch.equal(y.to(torch.int64),x.argmax(-1)):raise AssertionError('sampling argmax sentinel mismatch')
    sentinel=dict(passed=True,torch=torch.__version__,flashinfer=flashinfer.__version__,GPU=torch.cuda.get_device_name(),
        seconds=time.perf_counter()-tick,old_paged_sha256=PAGED_SHA,
        included_headers={name:sha(Path(os.environ['SGI_CUDA_INCLUDE'])/name) for name in ('curand.h','curand_kernel.h')})
    save(a.out/'preflight.json',sentinel)
    # Release this process's CUDA allocations before model child execution.
    del x,y
    torch.cuda.empty_cache()
    order=ORDERS[a.replicate];save(a.out/'arm_order.json',list(order));receipts=[]
    for arm in order:
        cmd=[sys.executable,str(source/'engine_screen.py'),'--model',str(a.model),'--native',str(a.native),
             '--out',str(a.out/arm),'--arm',arm,'--replicate',str(a.replicate)]
        tick=time.monotonic()
        with (a.out/(arm+'.log')).open('w') as output:
            try:rc=subprocess.run(cmd,stdout=output,stderr=subprocess.STDOUT,timeout=600).returncode
            except subprocess.TimeoutExpired:rc=124
        receipts.append(dict(arm=arm,returncode=rc,seconds=time.monotonic()-tick,command=cmd))
        save(a.out/'engine_receipts.json',receipts)
        print('ARM_FINISHED',arm,rc,flush=True)
        if rc:
            save(a.out/'FAILED.json',dict(stage='actual_engine',arm=arm,returncode=rc,replicate=a.replicate))
            raise SystemExit(2)
    comparisons=[]
    for name in ('burst','staggered'):
        outputs={arm:json.loads((a.out/arm/(name+'.json')).read_text()) for arm in order}
        if len({x['workload_sha256'] for x in outputs.values()})!=1:raise ValueError('workloads differ')
        ref={r['id']:r['token_ids'] for r in outputs['auto']['requests_detail']}
        for arm,data in outputs.items():
            actual={r['id']:r['token_ids'] for r in data['requests_detail']}
            mismatched=[rid for rid in ref if actual.get(rid)!=ref[rid]]
            comparisons.append(dict(workload=name,arm=arm,request_count=len(actual),
                token_match=actual==ref,mismatched_request_ids=mismatched,
                workload_sha256=data['workload_sha256'],source_result_sha256=sha(a.out/arm/(name+'.json'))))
    passed=all(x['token_match'] for x in comparisons)
    save(a.out/'output_parity.json',dict(passed=passed,comparisons=comparisons))
    save(a.out/'complete.json',dict(all_arms_complete=True,greedy_outputs_identical=passed,replicate=a.replicate,
        paged_qualified=True,qualification_reused_not_rerun=True,source_commit=(a.campaign/'source_commit.txt').read_text().strip()))
    print('FULL_ENGINE_COMPLETE',passed,flush=True)
    if not passed:raise SystemExit(3)

if __name__=='__main__':main()
