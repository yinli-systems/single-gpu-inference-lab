"""Controlled FlashInfer permutation assay; not a model or serving benchmark."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

CONFIGS = {
 'n2': ([4096,16384], [256,768], None),
 'n4': ([0,4096,12288,24576], [128,256,512,1024],
        ([128,1024,512,256], [512,256,1024,128])),
 'n8': ([0,2048,4096,6144,8192,12288,16384,24576], list(range(64,513,64)),
        ([64,128,320,384,512,448,256,192], [512,384,64,128,256,192,320,448])),
 'n16': ([1024*i for i in (0,1,2,3,4,5,6,8,10,12,14,16,18,20,24,28)], list(range(32,513,32)),
         ([32,64,512,256,448,160,416,288,480,384,128,352,224,320,192,96],
          [512,480,192,448,96,320,128,160,288,32,352,64,416,224,384,256]))}
ARMS = ('auto','unsplit')
MODES = ('eager','graph')
HEADS, KV_HEADS, DIM = 32, 8, 128

def digest(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, data):
 Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')

def states(name):
 k,q,eq = CONFIGS[name]
 out=[('same',q[:]),('opposite',q[::-1])]
 if eq: out += [('eqC-a',list(eq[0])),('eqC-b',list(eq[1]))]
 seen={tuple(p) for _,p in out}; rng=random.Random(919+len(k))
 for _ in range(10000):
  if len(out) >= min(math.factorial(len(k)), 10): break
  p=q[:]; rng.shuffle(p)
  if tuple(p) not in seen:
   seen.add(tuple(p)); out.append((f'rand{len(out)}',p))
 out.append(('order-null',q[:]))
 return out

def geometry(name, label, pairing):
 k,qs,_=CONFIGS[name]; q=list(pairing); k=list(k)
 ids=list(range(len(k)))
 if label=='order-null': q.reverse(); k.reverse(); ids.reverse()
 if sorted(q)!=sorted(qs): raise ValueError('not a permutation')
 C=sum(a*b for a,b in zip(q,k))
 W=C+sum(a*(a+1)//2 for a in q)
 return dict(name=name,state=label,query=q,cached=k,logical_ids=ids,
             marginal_key=[sorted(q),sorted(k)],C=C,W=W,
             total_kv=[a+b for a,b in zip(q,k)])

def schema_check(row):
 if row['arm'] not in ARMS or row['mode'] not in MODES: raise ValueError('unknown arm/mode')
 if type(row['block']) is not int or row['block']<0: raise ValueError('bad block')
 for key in ('device_us','wall_us','plan_us'):
  if not math.isfinite(row[key]) or row[key]<=0: raise ValueError('invalid duration')
 if row['W'] != row['C']+sum(q*(q+1)//2 for q in row['query']): raise ValueError('bad work')

def run(args):
 import torch
 import flashinfer
 torch.set_num_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 torch.manual_seed(20260929+args.rep)
 if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no synthetic timing')
 if flashinfer.__version__!='0.6.18': raise RuntimeError('frozen FlashInfer version required')
 args.out.mkdir(parents=True,exist_ok=False)
 rows=[]; validations=[]; graph_checks=0; written=0; source=digest(__file__)
 start=time.monotonic()
 gpu=torch.cuda.get_device_properties(0)
 meta=dict(source_sha256=source,rep=args.rep,stage=args.stage,torch=torch.__version__,
           flashinfer=flashinfer.__version__,cuda=torch.version.cuda,gpu=gpu.name,
           sm=[gpu.major,gpu.minor],sm_count=gpu.multi_processor_count,
           backend='fa2',heads=HEADS,kv_heads=KV_HEADS,head_dim=DIM,
           scratch_MiB_per_arm=512,seed=20260929+args.rep,
           job=os.getenv('SLURM_JOB_ID'),host=os.uname().nodename,
           attention_only=True,full_model=False,HTTP=False,
           timing_scope='plan excluded; eager device/wall and graph replay measured separately')
 for key, cmd in [('hardware',['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,power.limit','--format=csv']),
                  ('compute_apps',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'])]:
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
  meta[key]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 meta['flashinfer_source_sha256']=digest(Path(flashinfer.__file__).parent/'prefill.py')
 dump(args.out/'environment.json',meta)
 names=['n2','n4'] if args.stage=='canary' else list(CONFIGS)
 dtypes=['float16'] if args.stage=='canary' else ['float16','bfloat16']
 blocks=2 if args.stage=='canary' else 12
 inner=4 if args.stage=='canary' else 12
 def reference(q,k,v,g,out,dtype):
  qo=ko=0; largest=0.; checks=0
  atol,rtol=(.005,.02) if dtype==torch.float16 else (.02,.04)
  for nq,nk in zip(g['query'],g['total_kv']):
   idx=sorted(set([0,nq//2,nq-1]))
   qq=q[qo+torch.tensor(idx,device='cuda')].float().transpose(0,1)
   kk=k[ko:ko+nk].float().repeat_interleave(HEADS//KV_HEADS,dim=1).transpose(0,1)
   vv=v[ko:ko+nk].float().repeat_interleave(HEADS//KV_HEADS,dim=1).transpose(0,1)
   scores=torch.matmul(qq,kk.transpose(-1,-2))/math.sqrt(DIM)
   last=torch.tensor(idx,device='cuda')+nk-nq
   mask=torch.arange(nk,device='cuda')[None,:] > last[:,None]
   scores.masked_fill_(mask[None],float('-inf'))
   ref=(scores.softmax(-1)@vv).transpose(0,1)
   actual=out[qo+torch.tensor(idx,device='cuda')].float()
   torch.testing.assert_close(actual,ref,atol=atol,rtol=rtol)
   largest=max(largest,(actual-ref).abs().max().item()); checks+=len(idx)*HEADS
   qo+=nq; ko+=nk
  return largest,checks
 try:
  for dtype_name,name in itertools.product(dtypes,names):
   dtype=getattr(torch,dtype_name); depths,chunks,_=CONFIGS[name]
   ss=states(name); rng=random.Random(7183+args.rep*100+len(depths))
   # Distinct logical request buffers; physical concatenation happens outside timing.
   logical_q=[torch.randn((max(chunks),HEADS,DIM),device='cuda',dtype=dtype) for k in depths]
   logical_k=[torch.randn((k+max(chunks),KV_HEADS,DIM),device='cuda',dtype=dtype) for k in depths]
   logical_v=[torch.randn_like(k) for k in logical_k]
   spaces={a:torch.empty(512*1024**2,device='cuda',dtype=torch.uint8) for a in ARMS}
   wrappers={a:flashinfer.BatchPrefillWithRaggedKVCacheWrapper(spaces[a],backend='fa2') for a in ARMS}
   def prepare(label,pairing):
    g=geometry(name,label,pairing)
    q=torch.cat([logical_q[j][:n] for j,n in zip(g['logical_ids'],g['query'])])
    k=torch.cat([logical_k[j][:n] for j,n in zip(g['logical_ids'],g['total_kv'])])
    v=torch.cat([logical_v[j][:n] for j,n in zip(g['logical_ids'],g['total_kv'])])
    qp=[0]+list(itertools.accumulate(g['query'])); kp=[0]+list(itertools.accumulate(g['total_kv']))
    return g,q,k,v,torch.tensor(qp,device='cuda',dtype=torch.int32),torch.tensor(kp,device='cuda',dtype=torch.int32)
   null_reference=None
   # Qualify every state/arm independently before collecting this cell's timing.
   for label,pairing in ss:
    g,q,k,v,qp,kp=prepare(label,pairing); auto=None
    for arm in ARMS:
     w=wrappers[arm]; w.plan(qp,kp,HEADS,KV_HEADS,DIM,causal=True,q_data_type=dtype,kv_data_type=dtype,disable_split_kv=arm=='unsplit')
     out=w.run(q,k,v); torch.cuda.synchronize()
     if not torch.isfinite(out).all(): raise ValueError('nonfinite attention output')
     err,nref=reference(q,k,v,g,out,dtype)
     if auto is None: auto=out.clone()
     else: torch.testing.assert_close(out,auto,atol=.005 if dtype==torch.float16 else .02,rtol=.02 if dtype==torch.float16 else .04)
     validations.append(dict(config=name,dtype=dtype_name,state=label,arm=arm,fp32_max_abs=err,fp32_vectors=nref,finite=True))
     if arm=='auto' and label=='same': null_reference=out.clone()
     if arm=='auto' and label=='order-null':
      inverse=torch.cat(list(out.split(g['query']))[::-1])
      torch.testing.assert_close(inverse,null_reference,atol=.005 if dtype==torch.float16 else .02,rtol=.02 if dtype==torch.float16 else .04)
   # Two unscored warm-up blocks; no outlier filtering afterwards.
   for block in range(-2,blocks):
    ordered=ss[:]; rng.shuffle(ordered)
    for label,pairing in ordered:
     g,q,k,v,qp,kp=prepare(label,pairing)
     arm_order=list(ARMS); rng.shuffle(arm_order)
     for arm in arm_order:
      w=wrappers[arm]; torch.cuda.synchronize(); t=time.perf_counter()
      w.plan(qp,kp,HEADS,KV_HEADS,DIM,causal=True,q_data_type=dtype,kv_data_type=dtype,disable_split_kv=arm=='unsplit')
      torch.cuda.synchronize(); plan_us=(time.perf_counter()-t)*1e6
      out=torch.empty_like(q)
      for _ in range(3): w.run(q,k,v,out=out)
      torch.cuda.synchronize()
      eager_output=out.clone()
      graph=torch.cuda.CUDAGraph()
      with torch.cuda.graph(graph):
       for _ in range(4): w.run(q,k,v,out=out)
      graph.replay(); torch.cuda.synchronize()
      torch.testing.assert_close(out,eager_output,atol=0,rtol=0)
      graph_checks+=1
      modes=list(MODES);rng.shuffle(modes)
      for mode in modes:
       ev0,ev1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
       torch.cuda.synchronize(); t=time.perf_counter();ev0.record()
       if mode=='eager':
        for _ in range(inner): w.run(q,k,v,out=out)
        count=inner
       else:
        for _ in range(inner): graph.replay()
        count=inner*4
       ev1.record();ev1.synchronize();wall=(time.perf_counter()-t)*1e6/count
       row=dict(g,rep=args.rep,dtype=dtype_name,arm=arm,mode=mode,block=block,
                device_us=ev0.elapsed_time(ev1)*1000/count,wall_us=wall,plan_us=plan_us,
                timed_calls=count,trial_id=f'{args.rep}/{dtype_name}/{name}/{block}/{label}/{arm}/{mode}')
       if block>=0: schema_check(row);rows.append(row)
      del graph
    if block>=0:
     with (args.out/'raw.jsonl').open('a') as raw:
      for row in rows[written:]: raw.write(json.dumps(row,allow_nan=False)+'\n')
     written=len(rows)
     dump(args.out/'progress.json',dict(config=name,dtype=dtype_name,block=block,rows=len(rows),seconds=time.monotonic()-start))
   print('CELL_COMPLETE',name,dtype_name,len(rows),flush=True)
   del wrappers,spaces,logical_q,logical_k,logical_v,null_reference,auto,out,q,k,v
   torch.cuda.empty_cache()
  expected=sum(len(states(n)) for n in names)*len(dtypes)*blocks*len(ARMS)*len(MODES)
  if len(rows)!=expected or len({r['trial_id'] for r in rows})!=expected: raise ValueError('incomplete/duplicated trial matrix')
  dump(args.out/'measurements.json',rows);dump(args.out/'validation.json',validations)
  dump(args.out/'complete.json',dict(complete=True,stage=args.stage,rep=args.rep,rows=len(rows),expected=expected,
       checks=len(validations),graph_exact_checks=graph_checks,fp32_vectors=sum(v['fp32_vectors'] for v in validations),
       seconds=time.monotonic()-start,source_sha256=source,
       measurements_sha256=digest(args.out/'measurements.json'),validation_sha256=digest(args.out/'validation.json'),
       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),full_model=False,performance_promotion=False))
  print('COMPLETE',json.dumps(json.loads((args.out/'complete.json').read_text())),flush=True)
 except BaseException as exc:
  dump(args.out/'partial_measurements.json',rows);dump(args.out/'validation.json',validations)
  dump(args.out/'failure.json',dict(type=type(exc).__name__,error=str(exc),rows=len(rows),complete=False))
  raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
 p.add_argument('--stage',choices=['canary','test'],required=True);p.add_argument('--rep',type=int,default=0)
 run(p.parse_args())
