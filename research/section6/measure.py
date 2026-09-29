"""Bounded real-GPU measurement qualification and execution-order diagnostics.
No serving claim. All six geometries are exposed diagnostic states, not holdouts.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, os, random, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/order_guard'),str(ROOT/'benchmarks/results/plan-order-mechanism')]
from order_guard import Geometry, propose, digest
from plan_contract import order_indices, validate_descriptors
from measure_order import oracle

CASES=[
 ('short',[1],[512]),
 ('reversal2',[1024,63],[128,16384]),
 ('old-sentinel',[672,176,96],[8192,16896,24064]),
 ('changed12',[1024,256,128,128,128,64,64,63,33,31,1,1],[63,63,512,512,512,2048,2048,16384,16384,16384,16384,16384]),
 ('collision-a',[64,128,192],[64,128,0]),
 ('collision-b',[64,128,192],[128,0,64])]
POLICIES=('identity_label','identity_copy','locality_packet8','causal_heavy','request_reverse')
REGIMES=('eager_one_warm','graph_one_warm','graph32_steady','graph_one_pressure128MiB')
BLOCKS=12

def save(p,x):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def filehash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def command(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=15)
 return dict(rc=p.returncode,out=p.stdout,err=p.stderr)
def apply(s,order,torch):
 s.apply('identity')
 if sorted(order)!=list(range(len(s.desc))):raise ValueError('not a bijection')
 for col,key in enumerate(('request_indices_offset','qo_tile_indices_offset','kv_tile_indices_offset')):
  vals=torch.tensor([s.desc[i][col] for i in order],dtype=torch.int32,device=s.device)
  s.buffer.view(torch.uint8).narrow(0,s.info[key],4*len(order)).view(torch.int32).copy_(vals)
 torch.cuda.synchronize()
 actual=list(zip(*(s._read(s.info[k],len(order)) for k in ('request_indices_offset','qo_tile_indices_offset','kv_tile_indices_offset'))))
 if actual!=[s.desc[i] for i in order]:raise ValueError('metadata write/read mismatch')
 validate_descriptors(actual,s.q,s.lengths,s.info,s.chunk,s.group,s.o,s.merge)
 s.last_descriptor_hash=digest(actual)
 return s.last_descriptor_hash

def run(a):
 import torch,flashinfer
 from metadata_adapter import PlanSnapshot
 if not torch.cuda.is_available() or flashinfer.__version__!=a.version:raise RuntimeError('wrong real CUDA environment')
 if a.out.exists():raise FileExistsError('preserve evidence')
 a.out.mkdir(parents=True)
 torch.set_num_threads(1);torch.manual_seed(391831+a.rep);torch.backends.cuda.matmul.allow_tf32=False
 affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(affinity[:min(2,len(affinity))]))
 env=dict(version=flashinfer.__version__,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
  hardware=command(['nvidia-smi','--query-gpu=name,uuid,driver_version,pci.bus_id,power.limit','--format=csv']),
  process_before=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv']),
  source={p.name:filehash(p) for p in Path(__file__).parent.glob('*.py')},rep=a.rep,layout=a.layout,
  affinity_before=affinity,affinity=list(os.sched_getaffinity(0)),job=os.getenv('SLURM_JOB_ID'),
  sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
  cases=CASES,policies=POLICIES,blocks=BLOCKS,regimes=REGIMES,all_cases_exposed=True,
  default_promotion=False,serving_promotion=False,clock_lock=False,whole_node_exclusivity=False)
 if env['hardware']['rc']:raise RuntimeError('no hardware evidence')
 save(a.out/'environment.json',env)
 rows=[];checks=[];plans=[];traces=[]
 try:
  for dtype_name,(name,qs,ks),split in itertools.product(('float16','bfloat16'),CASES,('auto','unsplit')):
   dtype=getattr(torch,dtype_name);ls=[q+k for q,k in zip(qs,ks)]
   q=torch.randn((sum(qs),32,128),device='cuda',dtype=dtype)
   k=torch.randn((sum(ls),8,128),device='cuda',dtype=dtype);v=torch.randn_like(k)
   qi=torch.tensor([0]+list(itertools.accumulate(qs)),device='cuda',dtype=torch.int32)
   ki=torch.tensor([0]+list(itertools.accumulate(ls)),device='cuda',dtype=torch.int32)
   ws=torch.empty(512*1024**2,device='cuda',dtype=torch.uint8)
   w=flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws,backend='fa2')
   t=time.perf_counter();w.plan(qi,ki,32,8,128,causal=True,q_data_type=dtype,kv_data_type=dtype,disable_split_kv=split=='unsplit');torch.cuda.synchronize()
   plan_us=(time.perf_counter()-t)*1e6
   s=PlanSnapshot(w,qs,ls);g=Geometry(tuple(qs),tuple(ls),4,s.info['cta_tile_q'],s.chunk,bool(s.info['split_kv']))
   out=torch.empty_like(q);lse=torch.empty((sum(qs),32),device='cuda',dtype=torch.float32)
   def call():return w.run(q,k,v,out=out,lse=lse,return_lse=True)
   for _ in range(8):call()
   torch.cuda.synchronize();ref=out.clone();rlse=lse.clone();fp32=oracle(q,k,v,ref,rlse,qs,ls)
   graphs={}
   for n in (1,32):
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
     for _ in range(n):call()
    graphs[n]=graph
   flush=torch.empty(128*1024**2,device='cuda',dtype=torch.uint8)
   ptrs=[x.data_ptr() for x in (q,k,v,out,lse,ws)]
   orders={'identity':tuple(range(len(s.desc)))};generation={}
   for pol in POLICIES:
    tick=time.perf_counter()
    if pol.startswith('identity'):orders[pol]=orders['identity']
    elif pol in ('causal_heavy','locality_packet8'):orders[pol]=propose(g,s.desc,pol)
    else:orders[pol]=tuple(order_indices(s.desc,qs,ls,g.tile,g.chunk,g.split,g.group,pol))
    generation[pol]=(time.perf_counter()-tick)*1e6
   key=dict(case=name,dtype=dtype_name,split=split,rep=a.rep,version=a.version,layout=a.layout)
   plan=dict(key,plan=s.report,plan_us=plan_us,FP32=fp32,pointers=ptrs,generation_us=generation)
   plans.append(plan)
   e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   e0.record();e1.record();e1.synchronize();e0.elapsed_time(e1)
   for pol,order in orders.items():
    signature=apply(s,order,torch)
    for mode in ('eager','graph1','graph32'):
     out.fill_(float('nan'));lse.fill_(float('nan'))
     call() if mode=='eager' else graphs[1 if mode=='graph1' else 32].replay()
     torch.cuda.synchronize();torch.testing.assert_close(out,ref,atol=0,rtol=0);torch.testing.assert_close(lse,rlse,atol=0,rtol=0)
    checks.append(dict(key,policy=pol,descriptor_hash=signature,changed=tuple(order)!=orders['identity'],exact=True))
   for _ in range(8):graphs[32].replay()
   torch.cuda.synchronize()
   rng=random.Random(73001+a.rep)
   for block in range(BLOCKS):
    combos=list(itertools.product(POLICIES,REGIMES));rng.shuffle(combos)
    for pol,regime in combos:
     sequence=('identity',pol,pol,'identity') if block%2==0 else (pol,'identity','identity',pol)
     for position,arm in enumerate(sequence):
      tick=time.perf_counter();signature=apply(s,orders[arm],torch);setup_us=(time.perf_counter()-tick)*1e6
      for _ in range(4):graphs[1].replay()
      if regime.endswith('pressure128MiB'):flush.zero_()
      torch.cuda.synchronize();start=time.perf_counter();e0.record()
      if regime=='eager_one_warm':call()
      else:graphs[32 if regime=='graph32_steady' else 1].replay()
      e1.record();e1.synchronize();wall_us=(time.perf_counter()-start)*1e6
      calls=32 if regime=='graph32_steady' else 1
      device_us=e0.elapsed_time(e1)*1000/calls
      if not math.isfinite(device_us) or device_us<=0:raise ValueError('invalid timing')
      rows.append(dict(key,block=block,policy=pol,regime=regime,position=position,arm=arm,
       device_us=device_us,wall_us=wall_us/calls,setup_us=setup_us,calls=calls,descriptor_hash=signature))
     torch.testing.assert_close(out,ref,atol=0,rtol=0);torch.testing.assert_close(lse,rlse,atol=0,rtol=0)
     if ptrs!=[x.data_ptr() for x in (q,k,v,out,lse,ws)]:raise ValueError('buffer changed')
   # Profiling AFTER unprofiled measurements, separate diagnostic artifact.
   if a.rep==0 and name=='reversal2' and dtype_name=='bfloat16' and split=='unsplit':
    for pol in ('identity','causal_heavy','request_reverse'):
     apply(s,orders[pol],torch);graphs[1].replay();torch.cuda.synchronize()
     try:
      with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
       with torch.profiler.record_function('SECTION6_DIAGNOSTIC_'+pol):
        graphs[1].replay();torch.cuda.synchronize()
      target=a.out/('profile-'+pol+'.json');prof.export_chrome_trace(str(target))
      events=json.loads(target.read_text())['traceEvents']
      traces.append(dict(policy=pol,kernels=[{k:e.get(k) for k in ('name','cat','dur','args')} for e in events if e.get('cat')=='kernel']))
     except Exception as exc:traces.append(dict(policy=pol,error=str(exc),not_performance_evidence=True))
   s.apply('identity')
   save(a.out/'progress.json',dict(key,rows=len(rows),checks=len(checks)))
   print('CELL',a.version,a.rep,name,dtype_name,split,len(rows),flush=True)
   del graphs,graph,s,w,ws,q,k,v,qi,ki,out,lse,ref,rlse,flush
   torch.cuda.empty_cache()
  expected=2*len(CASES)*2*len(POLICIES)*len(REGIMES)*BLOCKS*4
  if len(rows)!=expected:raise ValueError('incomplete matrix')
  save(a.out/'measurements.json',rows);save(a.out/'qualification.json',checks);save(a.out/'plans.json',plans);save(a.out/'kernel-diagnostics.json',traces)
  save(a.out/'complete.json',dict(complete=True,rows=len(rows),expected=expected,checks=len(checks),
   files={p.name:filehash(p) for p in a.out.glob('*.json')},no_serving_promotion=True))
 except BaseException as exc:
  save(a.out/'partial_measurements.json',rows);save(a.out/'qualification.json',checks);save(a.out/'plans.json',plans)
  save(a.out/'failure.json',dict(type=type(exc).__name__,message=str(exc)));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--version',required=True);p.add_argument('--rep',type=int,required=True)
 p.add_argument('--layout',choices=['ragged'],default='ragged');p.add_argument('--out',type=Path,required=True);run(p.parse_args())
