"""Pristine/disabled/candidate plan+run comparison on one exposed witness."""
from __future__ import annotations
import argparse,hashlib,json,os,random,sys,time,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from measure import save,filehash,command
from metadata_adapter import PlanSnapshot
from order_guard import Geometry,propose,digest
from measure_order import oracle

def run(a):
 import torch,flashinfer
 if flashinfer.__version__!='0.7.0' or not torch.cuda.is_available():raise RuntimeError('wrong environment')
 if a.out.exists():raise FileExistsError('preserve evidence')
 a.out.mkdir(parents=True);torch.set_num_threads(1);torch.manual_seed(77291+a.rep);torch.backends.cuda.matmul.allow_tf32=False
 mask=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(mask[:min(2,len(mask))]))
 pkg=Path(flashinfer.__file__).parent
 if a.mode=='pristine':
  if filehash(pkg/'data/include/flashinfer/attention/prefill.cuh')!='e66bce2652c0a3c5b54f88510aff2b4b48ff6a52c4ee23fa97f91fa06a935b87':raise RuntimeError('not pristine header')
 else:
  bind=json.loads((pkg.parent/'NATIVE_BINDING.json').read_text())
  if filehash(pkg/'data/include/flashinfer/attention/scheduler.cuh')!=bind['scheduler_modified_sha256']:raise RuntimeError('wrong native planner')
 os.environ['SGI_FA2_RESIDENCY_PROBE']='1' if a.mode=='candidate' else '0'
 env=dict(mode=a.mode,rep=a.rep,gpu=torch.cuda.get_device_name(),torch=torch.__version__,flashinfer=flashinfer.__version__,
  hardware=command(['nvidia-smi','--query-gpu=name,uuid,driver_version,power.limit','--format=csv']),
  source_sha256=filehash(__file__),prefill_header_sha256=filehash(pkg/'data/include/flashinfer/attention/prefill.cuh'),
  scheduler_header_sha256=filehash(pkg/'data/include/flashinfer/attention/scheduler.cuh'),
  affinity=list(os.sched_getaffinity(0)),all_cases_exposed=True,full_model=False,includes_native_plan_sort_transfer=True,
  excludes=['JIT','import','qualification','initial_graph_capture'],default_promotion=False)
 save(a.out/'environment.json',env);rows=[]
 try:
  qs=[1024,63];ls=[1152,16447];dtype=torch.bfloat16
  q=torch.randn((1087,32,128),device='cuda',dtype=dtype);k=torch.randn((17599,8,128),device='cuda',dtype=dtype);v=torch.randn_like(k)
  qi=torch.tensor([0,1024,1087],device='cuda',dtype=torch.int32);ki=torch.tensor([0,1152,17599],device='cuda',dtype=torch.int32)
  ws=torch.empty(512*1024**2,device='cuda',dtype=torch.uint8);w=flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws,backend='fa2')
  def plan():w.plan(qi,ki,32,8,128,causal=True,q_data_type=dtype,kv_data_type=dtype,disable_split_kv=True)
  plan();snap=PlanSnapshot(w,qs,ls);g=Geometry(tuple(qs),tuple(ls),4,snap.info['cta_tile_q'],snap.chunk,False)
  native=g.descriptors();order=propose(g,native,'causal_heavy') if a.mode=='candidate' else tuple(range(len(native)))
  if snap.desc!=[native[i] for i in order]:raise RuntimeError('native order not applied as specified')
  dh=digest(snap.desc);del snap
  out=torch.empty_like(q);lse=torch.empty((1087,32),device='cuda',dtype=torch.float32)
  def call():return w.run(q,k,v,out=out,lse=lse,return_lse=True)
  for _ in range(8):call()
  torch.cuda.synchronize();ref=out.clone();rlse=lse.clone();fp32=oracle(q,k,v,ref,rlse,qs,ls)
  def tensorhash(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
  hashes={n:tensorhash(t) for n,t in [('q',q),('k',k),('v',v),('out',out),('lse',lse)]}
  ptrs=[x.data_ptr() for x in (q,k,v,out,lse,ws,w._int_workspace_buffer)]
  graphs={}
  for n in (1,32):
   gr=torch.cuda.CUDAGraph()
   with torch.cuda.graph(gr):
    for _ in range(n):call()
   graphs[n]=gr
  for n in (1,32):
   plan();out.fill_(float('nan'));lse.fill_(float('nan'));graphs[n].replay();torch.cuda.synchronize()
   torch.testing.assert_close(out,ref,atol=0,rtol=0);torch.testing.assert_close(lse,rlse,atol=0,rtol=0)
  if a.rep==0:
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as pr:
    graphs[1].replay();torch.cuda.synchronize()
   p=a.out/'launch-trace.json';pr.export_chrome_trace(str(p));ks=[e for e in json.loads(p.read_text())['traceEvents'] if e.get('cat')=='kernel' and 'BatchPrefillWithRaggedKVCacheKernel' in e.get('name','')]
   if len(ks)!=1 or ks[0]['args']['shared memory']!=(65536 if a.mode=='candidate' else 49152):raise RuntimeError('incorrect launch')
   save(a.out/'launch-proof.json',[dict(name=x['name'],args=x['args']) for x in ks])
  for _ in range(8):graphs[32].replay()
  torch.cuda.synchronize();rng=random.Random(18991+a.rep)
  for block in range(12):
   ns=[1,32];rng.shuffle(ns)
   for n in ns:
    seq=('main','repeat','repeat','main') if block%2==0 else ('repeat','main','main','repeat')
    for pos,label in enumerate(seq):
     graphs[n].replay();torch.cuda.synchronize()
     tick=time.perf_counter();plan();graphs[n].replay();torch.cuda.synchronize();cycle_us=(time.perf_counter()-tick)*1e6
     if not math.isfinite(cycle_us) or cycle_us<=0:raise RuntimeError('invalid timing')
     rows.append(dict(rep=a.rep,mode=a.mode,block=block,calls=n,label=label,position=pos,cycle_us=cycle_us))
    torch.testing.assert_close(out,ref,atol=0,rtol=0);torch.testing.assert_close(lse,rlse,atol=0,rtol=0)
    if ptrs!=[x.data_ptr() for x in (q,k,v,out,lse,ws,w._int_workspace_buffer)]:raise RuntimeError('plan storage identity changed')
  snap=PlanSnapshot(w,qs,ls)
  if digest(snap.desc)!=dh:raise RuntimeError('planner order unstable')
  save(a.out/'qualification.json',dict(full_output_exact=True,lse_exact=True,FP32=fp32,hashes=hashes,descriptor_hash=dh,plan=snap.report,buffer_pointers=ptrs))
  if len(rows)!=96:raise RuntimeError('incomplete cycles')
  save(a.out/'measurements.json',rows);save(a.out/'complete.json',dict(complete=True,rows=len(rows),files={p.name:filehash(p) for p in a.out.glob('*.json')},serving_promotion=False))
  print('NATIVE_CYCLE_COMPLETE',a.mode,a.rep,len(rows),flush=True)
 except BaseException as e:
  save(a.out/'partial_measurements.json',rows);save(a.out/'failure.json',dict(type=type(e).__name__,message=str(e)));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pristine','disabled','candidate'],required=True);p.add_argument('--rep',type=int,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
