"""Real FA2 host-launch residency intervention, one exposed shape only."""
from __future__ import annotations
import argparse,hashlib,itertools,json,math,os,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from measure import apply,save,filehash,command
from metadata_adapter import PlanSnapshot
from order_guard import Geometry,propose
from measure_order import oracle
ARMS={'native48':('identity',0),'repeat48':('identity',0),'heavy48':('causal_heavy',0),'native64':('identity',1),'heavy64':('causal_heavy',1)}
REGIMES=('graph_one_warm','graph32_steady','graph_one_pressure128MiB')

def run(a):
 import torch,flashinfer
 if flashinfer.__version__!='0.7.0' or not torch.cuda.is_available():raise RuntimeError('requires pinned real CUDA 0.7.0 overlay')
 if a.out.exists():raise FileExistsError('preserve evidence')
 a.out.mkdir(parents=True);torch.set_num_threads(1);torch.manual_seed(314159+a.rep);torch.backends.cuda.matmul.allow_tf32=False
 mask=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(mask[:min(2,len(mask))]))
 package=Path(flashinfer.__file__).parent;binding=json.loads((package.parent/'SOURCE_BINDING.json').read_text())
 if filehash(package/'data/include/flashinfer/attention/prefill.cuh')!=binding['modified_sha256']:raise RuntimeError('unexpected native source')
 env=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,flashinfer=flashinfer.__version__,rep=a.rep,
  hardware=command(['nvidia-smi','--query-gpu=name,uuid,driver_version,power.limit','--format=csv']),
  processes=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv']),
  affinity=list(os.sched_getaffinity(0)),source_binding=binding,driver_clocks_locked=False,
  source_sha256=filehash(__file__),all_cases_exposed=True,full_model=False,production_promotion=False)
 save(a.out/'environment.json',env);rows=[];checks=[]
 try:
  qs=[1024,63];ls=[1152,16447];dtype=torch.bfloat16
  q=torch.randn((sum(qs),32,128),device='cuda',dtype=dtype);k=torch.randn((sum(ls),8,128),device='cuda',dtype=dtype);v=torch.randn_like(k)
  qi=torch.tensor([0,1024,1087],device='cuda',dtype=torch.int32);ki=torch.tensor([0,1152,17599],device='cuda',dtype=torch.int32)
  ws=torch.empty(512*1024**2,device='cuda',dtype=torch.uint8)
  w=flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws,backend='fa2');w.plan(qi,ki,32,8,128,causal=True,q_data_type=dtype,kv_data_type=dtype,disable_split_kv=True)
  s=PlanSnapshot(w,qs,ls)
  if s.info['split_kv'] or s.info['cta_tile_q']!=128 or len(s.desc)!=34:raise RuntimeError('unsupported witness plan')
  g=Geometry(tuple(qs),tuple(ls),4,128,s.chunk,False)
  orders={name:propose(g,s.desc,name) for name in ('identity','causal_heavy')}
  out=torch.empty_like(q);lse=torch.empty((1087,32),device='cuda',dtype=torch.float32)
  def call():return w.run(q,k,v,out=out,lse=lse,return_lse=True)
  os.environ['SGI_FA2_RESIDENCY_PROBE']='0'
  for _ in range(8):call()
  torch.cuda.synchronize();ref=out.clone();rlse=lse.clone();fp32=oracle(q,k,v,ref,rlse,qs,ls)
  graphs={}
  for cap,n in itertools.product((0,1),(1,32)):
   os.environ['SGI_FA2_RESIDENCY_PROBE']=str(cap);call();torch.cuda.synchronize()
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):
    for _ in range(n):call()
   graphs[(cap,n)]=graph
  ptrs=[x.data_ptr() for x in (q,k,v,out,lse,ws)]
  for name,(pol,cap) in ARMS.items():
   h=apply(s,orders[pol],torch);os.environ['SGI_FA2_RESIDENCY_PROBE']=str(cap)
   for n in (0,1,32):
    out.fill_(float('nan'));lse.fill_(float('nan'))
    call() if n==0 else graphs[(cap,n)].replay()
    torch.cuda.synchronize();torch.testing.assert_close(out,ref,atol=0,rtol=0);torch.testing.assert_close(lse,rlse,atol=0,rtol=0)
   checks.append(dict(arm=name,descriptor_hash=h,exact=True,shared_bytes=65536 if cap else 49152))
  save(a.out/'qualification.json',checks);save(a.out/'plan.json',dict(plan=s.report,FP32=fp32,pointers=ptrs))
  # Actual launch proof, not benchmark timing. A failed proof stops this probe.
  launch=[]
  for name in ('native48','heavy48','native64','heavy64'):
   pol,cap=ARMS[name];apply(s,orders[pol],torch);graphs[(cap,1)].replay();torch.cuda.synchronize()
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    graphs[(cap,1)].replay();torch.cuda.synchronize()
   path=a.out/('launch-'+name+'.json');prof.export_chrome_trace(str(path))
   es=[e for e in json.loads(path.read_text())['traceEvents'] if e.get('cat')=='kernel']
   es=[e for e in es if 'BatchPrefillWithRaggedKVCacheKernel' in e.get('name','')]
   if len(es)!=1:raise RuntimeError('kernel launch proof unavailable')
   args=es[0]['args'];expected=65536 if cap else 49152
   if args['shared memory']!=expected or args['grid']!=[34,1,8] or args['block']!=[32,4,1]:raise RuntimeError('host launch patch did not reach GPU')
   launch.append(dict(arm=name,name=es[0]['name'],args=args,profiled_duration_not_used=es[0]['dur']))
  if len({(x['name'],x['args']['registers per thread']) for x in launch})!=1:raise RuntimeError('kernel code/resource mismatch')
  save(a.out/'launch-proof.json',launch)
  flush=torch.empty(128*1024**2,device='cuda',dtype=torch.uint8)
  e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);e0.record();e1.record();e1.synchronize();e0.elapsed_time(e1)
  for cap in (0,1):
   for _ in range(8):graphs[(cap,32)].replay()
  torch.cuda.synchronize();rng=random.Random(19191+a.rep)
  for block in range(12):
   combos=list(itertools.product(tuple(ARMS)[1:],REGIMES));rng.shuffle(combos)
   for pol,regime in combos:
    seq=('native48',pol,pol,'native48') if block%2==0 else (pol,'native48','native48',pol)
    for pos,arm in enumerate(seq):
     order,cap=ARMS[arm];tick=time.perf_counter();h=apply(s,orders[order],torch);setup_us=(time.perf_counter()-tick)*1e6
     for _ in range(4):graphs[(cap,1)].replay()
     if regime.endswith('pressure128MiB'):flush.zero_()
     torch.cuda.synchronize();calls=32 if regime=='graph32_steady' else 1
     start=time.perf_counter();e0.record();graphs[(cap,calls)].replay();e1.record();e1.synchronize()
     wall=(time.perf_counter()-start)*1e6/calls;dev=e0.elapsed_time(e1)*1000/calls
     if not all(math.isfinite(x) and x>0 for x in (wall,dev,setup_us)):raise RuntimeError('invalid timing')
     rows.append(dict(rep=a.rep,block=block,comparison=pol,regime=regime,position=pos,arm=arm,device_us=dev,wall_us=wall,setup_us=setup_us,calls=calls,descriptor_hash=h))
    torch.testing.assert_close(out,ref,atol=0,rtol=0);torch.testing.assert_close(lse,rlse,atol=0,rtol=0)
    if ptrs!=[x.data_ptr() for x in (q,k,v,out,lse,ws)]:raise RuntimeError('buffers changed')
  if len(rows)!=576:raise RuntimeError('incomplete diagnostic matrix')
  save(a.out/'measurements.json',rows);save(a.out/'complete.json',dict(complete=True,rows=len(rows),checks=len(checks),files={p.name:filehash(p) for p in a.out.glob('*.json')},serving_promotion=False))
  print('RESIDENCY_PROBE_COMPLETE',a.rep,len(rows),flush=True)
 except BaseException as e:
  save(a.out/'partial_measurements.json',rows);save(a.out/'failure.json',dict(type=type(e).__name__,message=str(e)));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--rep',type=int,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
