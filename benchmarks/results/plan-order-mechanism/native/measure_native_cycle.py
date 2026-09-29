"""Actual source-integrated host planning pilot, not a serving benchmark.

Both modes use the same freshly compiled experimental source; identity is default-off.
Full-cycle measurements charge native plan generation plus actual graph replays.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time
import subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from measure_order import cases,oracle
from plan_contract import PlanSnapshot,order_indices,canonical_hash

def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def run(a):
 import torch,flashinfer
 torch.set_num_threads(1);torch.manual_seed(20261204)
 torch.backends.cuda.matmul.allow_tf32=False
 if flashinfer.__version__!='0.6.18':raise ValueError('pinned version required')
 import flashinfer.jit.env as jit_env
 # Rebuild this deliberately modified source; do not consume an unmodified AOT binary.
 jit_env.FLASHINFER_AOT_DIR=Path(os.environ['FLASHINFER_WORKSPACE_BASE'])/'source-only-aot'
 package=Path(flashinfer.__file__).parent
 binding=json.loads((Path(__file__).parent/'integration-source.json').read_text())
 header=package/'data/include/flashinfer/attention/scheduler.cuh'
 if hashlib.sha256(header.read_bytes()).hexdigest()!=binding['patched_scheduler_sha256']:raise ValueError('patched include not active')
 if 'native-overlay' not in str(package):raise ValueError('isolated package required')
 selected=[c for c in cases('test') if c['case'] in ('discovery-n2','discovery-n4') and c['state'] in ('same','opposite')]
 selected += [c for c in cases('test') if c['holdout'] and c['state']=='same']
 a.out.mkdir(parents=True,exist_ok=False)
 rows=[];checks=[];plans=[];start=time.monotonic();rng=random.Random(81024)
 meta=dict(gpu=torch.cuda.get_device_name(),package=str(package),torch=torch.__version__,flashinfer=flashinfer.__version__,
   job=os.getenv('SLURM_JOB_ID'),source_binding=binding,cases=selected,blocks=6,reuse_counts=[1,36],
   scope='source-integrated FA2 host planner plus graph replay cycle; not full model or HTTP',
   baseline='identity path of same compiled source (extra disabled environment check charged)',
   full_model=False,serving_promotion=False,prototype_only=True,
   source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).parent/'sgi_experimental_order.cuh']})
 p=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,power.limit','--format=csv'],capture_output=True,text=True,timeout=10)
 meta['hardware']=dict(rc=p.returncode,out=p.stdout,err=p.stderr);dump(a.out/'environment.json',meta)
 try:
  for dtype_name,c,split in itertools.product(['float16','bfloat16'],selected,['auto','unsplit']):
   dtype=getattr(torch,dtype_name);qs=c['query'];ls=[q+k for q,k in zip(qs,c['cached'])]
   q=torch.randn((sum(qs),32,128),device='cuda',dtype=dtype)
   k=torch.randn((sum(ls),8,128),device='cuda',dtype=dtype);v=torch.randn_like(k)
   qp=torch.tensor([0]+list(itertools.accumulate(qs)),device='cuda',dtype=torch.int32)
   kp=torch.tensor([0]+list(itertools.accumulate(ls)),device='cuda',dtype=torch.int32)
   space=torch.empty(512*1024**2,device='cuda',dtype=torch.uint8)
   w=flashinfer.BatchPrefillWithRaggedKVCacheWrapper(space,backend='fa2')
   def plan(policy):
    os.environ['FLASHINFER_EXP_WORK_ORDER']='heavy_first' if policy=='heavy_first' else 'identity'
    w.plan(qp,kp,32,8,128,causal=True,q_data_type=dtype,kv_data_type=dtype,disable_split_kv=(split=='unsplit'))
   plan('identity');torch.cuda.synchronize();original=PlanSnapshot(w,qs,ls)
   desc=original.desc
   ids=order_indices(desc,qs,ls,original.info['cta_tile_q'],original.chunk,bool(original.info['split_kv']),4,'heavy_first')
   expected={'identity':canonical_hash(desc),'identity_repeat':canonical_hash(desc),'heavy_first':canonical_hash([desc[i] for i in ids])}
   vectors=original.vector;invariant=original.invariant
   out=torch.empty_like(q);lse=torch.empty((sum(qs),32),device='cuda',dtype=torch.float32)
   def call():return w.run(q,k,v,out=out,lse=lse,return_lse=True)
   for _ in range(4):call()
   torch.cuda.synchronize();ref=out.clone();rlse=lse.clone();oref=oracle(q,k,v,ref,rlse,qs,ls)
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):call()
   key=dict(case=c['case'],state=c['state'],holdout=c['holdout'],dtype=dtype_name,split=split)
   for policy in expected:
    plan(policy);torch.cuda.synchronize();snap=PlanSnapshot(w,qs,ls)
    if snap.vector!=vectors or snap.invariant!=invariant:raise ValueError('non-order planner state changed')
    if canonical_hash(snap.desc)!=expected[policy]:raise ValueError('actual native planner does not match frozen policy')
    out.fill_(float('nan'));lse.fill_(float('nan'));call();torch.cuda.synchronize()
    torch.testing.assert_close(out,ref,atol=0,rtol=0);torch.testing.assert_close(lse,rlse,atol=0,rtol=0)
    out.fill_(float('nan'));lse.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
    torch.testing.assert_close(out,ref,atol=0,rtol=0);torch.testing.assert_close(lse,rlse,atol=0,rtol=0)
    checks.append(dict(key,policy=policy,descriptor_hash=expected[policy],output_exact=True,lse_exact=True,graph_exact=True))
    del snap
   plans.append(dict(key,baseline=original.report,heavy_descriptor_hash=expected['heavy_first'],FP32_reference=oref))
   for reuse in (1,36):
    for block in range(-2,6):
     policies=list(expected);rng.shuffle(policies)
     for policy in policies:
      os.environ['FLASHINFER_EXP_WORK_ORDER']='heavy_first' if policy=='heavy_first' else 'identity'
      graph.replay();torch.cuda.synchronize()
      t=time.perf_counter()
      w.plan(qp,kp,32,8,128,causal=True,q_data_type=dtype,kv_data_type=dtype,disable_split_kv=(split=='unsplit'))
      for _ in range(reuse):graph.replay()
      torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1e6
      torch.testing.assert_close(out,ref,atol=0,rtol=0);torch.testing.assert_close(lse,rlse,atol=0,rtol=0)
      if not (0<elapsed<float('inf')):raise ValueError('invalid cycle')
      if block>=0:rows.append(dict(key,policy=policy,reuse=reuse,block=block,cycle_us=elapsed,actual_attention_calls=reuse))
   dump(a.out/'partial_measurements.json',rows);dump(a.out/'progress.json',dict(key,rows=len(rows),checks=len(checks),seconds=time.monotonic()-start))
   print('NATIVE_CELL',dtype_name,c['case'],c['state'],split,len(rows),flush=True)
   del graph,original,w,space,ref,rlse,q,k,v,qp,kp,out,lse
   torch.cuda.empty_cache()
  expected_rows=len(selected)*2*2*2*6*3
  if len(rows)!=expected_rows:raise ValueError('incomplete cycle matrix')
  dump(a.out/'measurements.json',rows);dump(a.out/'qualification.json',checks);dump(a.out/'plans.json',plans)
  dump(a.out/'compiled-libraries.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(os.environ['FLASHINFER_WORKSPACE_BASE']).rglob('*.so')})
  dump(a.out/'complete.json',dict(complete=True,rows=len(rows),expected_rows=expected_rows,qualifications=len(checks),
       seconds=time.monotonic()-start,full_model=False,serving_promotion=False,
       files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.glob('*.json')}))
  print('NATIVE_COMPLETE',len(rows),flush=True)
 except BaseException as exc:
  dump(a.out/'partial_measurements.json',rows);dump(a.out/'qualification.json',checks);dump(a.out/'plans.json',plans)
  dump(a.out/'failure.json',dict(type=type(exc).__name__,message=str(exc),rows=len(rows)));raise
 finally:os.environ.pop('FLASHINFER_EXP_WORK_ORDER',None)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args())
