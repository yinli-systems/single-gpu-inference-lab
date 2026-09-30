"""v4 same-binary calibration/evaluation measurements."""
from __future__ import annotations
import argparse,gc,hashlib,itertools,json,math,os,random,subprocess,sys,time
from pathlib import Path
from manifest import load,digest
from policy import candidate_pool
from autotune import canonical_hash
from measurement_contract import REVISION,choose_eager_calls,check_window

EXECUTIONS=('eager_full_call','graph1_replay','graph16_replay')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def command(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=20);return {'rc':p.returncode,'out':p.stdout,'err':p.stderr}
def thash(t):
 import torch
 return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

class Runtime:
 def __init__(self,flashinfer,torch,layout,qi,ks,vs,k,v,lengths,dtype,split,policy,paged=None):
  self.torch=torch;self.policy=policy;self.ws=torch.empty(512*1024**2,device='cuda',dtype=torch.uint8);self.graphs={}
  kw=dict(causal=True,q_data_type=dtype,kv_data_type=dtype,disable_split_kv=(split=='unsplit'))
  if layout=='ragged':
   ki=torch.tensor([0]+list(itertools.accumulate(lengths)),device='cuda',dtype=torch.int32)
   self.wrapper=flashinfer.BatchPrefillWithRaggedKVCacheWrapper(self.ws,backend='fa2')
   self._plan=lambda:self.wrapper.plan(qi,ki,32,8,128,**kw);self._call=lambda q:self.wrapper.run(q,k,v,out=self.out,lse=self.lse,return_lse=True);self.keep=(ki,)
  else:
   page,pp,pi,last,kp,vp=paged;self.wrapper=flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.ws,'NHD',backend='fa2')
   self._plan=lambda:self.wrapper.plan(qi,pp,pi,last,32,8,128,page,**kw);self._call=lambda q:self.wrapper.run(q,(kp,vp),out=self.out,lse=self.lse,return_lse=True);self.keep=(pp,pi,last,kp,vp)
  self.wrapper._sgi_resource_policy=policy;self.eager_calls=16
 def attach(self,q):
  self.out=self.torch.empty_like(q);self.lse=self.torch.empty((q.shape[0],32),device='cuda',dtype=self.torch.float32)
 def plan(self):
  self._plan();info=[int(x) for x in self.wrapper._plan_info]
  if len(info)!=16:raise RuntimeError('plan vector size')
  if self.policy in (0,1) and info[15]!=self.policy:raise RuntimeError('plan policy')
  if self.policy==2 and info[15] not in (0,1):raise RuntimeError('candidate-pool policy')
  return info
 def call(self,q):return self._call(q)
 def capture(self,q,count):
  self.plan();g=self.torch.cuda.CUDAGraph()
  with self.torch.cuda.graph(g):
   for _ in range(count):self.call(q)
  self.graphs[count]=g
 def pointers(self,q,k,v):return [x.data_ptr() for x in (q,k,v,self.out,self.lse,self.ws,self.wrapper._int_workspace_buffer)]

def calibrate_calls(rt,q,torch):
 p=[]
 for _ in range(2):
  rt.plan();rt.call(q);torch.cuda.synchronize();t=time.perf_counter_ns()
  for __ in range(16):rt.plan();rt.call(q)
  torch.cuda.synchronize();p.append((time.perf_counter_ns()-t)/1000)
 return choose_eager_calls(p),p

def timed(rt,execution,q,torch):
 if execution=='eager_full_call':
  rt.plan();rt.call(q);torch.cuda.synchronize();a=torch.cuda.Event(True);b=torch.cuda.Event(True);t=time.perf_counter_ns();a.record()
  for _ in range(rt.eager_calls):rt.plan();rt.call(q)
  b.record();b.synchronize();elapsed=(time.perf_counter_ns()-t)/1000;check_window(elapsed,rt.eager_calls);return elapsed/rt.eager_calls,a.elapsed_time(b)*1000/rt.eager_calls,rt.eager_calls
 count=1 if execution=='graph1_replay' else 16;replays=16 if count==1 else 1;g=rt.graphs[count];g.replay();torch.cuda.synchronize();a=torch.cuda.Event(True);b=torch.cuda.Event(True);t=time.perf_counter_ns();a.record()
 for _ in range(replays):g.replay()
 b.record();b.synchronize();return (time.perf_counter_ns()-t)/1000/(count*replays),a.elapsed_time(b)*1000/(count*replays),count*replays

def identity(env,case,dtype,layout,split,execution,plan):
 op={'execution_mode':execution,'backend':'fa2','causal':True,'layout':layout,'dtype':dtype,'requested_split':split,'actual_split':'split' if plan[14] else 'unsplit','num_qo_heads':32,'num_kv_heads':8,'head_dim_qk':128,'head_dim_vo':128,'page_size':1 if layout=='ragged' else 16,'q':case['q'],'cached':case['cached'],'plan_core':plan[:-1]}
 e={k:env[k] for k in ('gpu_name','gpu_uuid','num_sms','driver','cuda','torch','flashinfer','python','cudnn','nvcc','source_archive_sha256','overlay_sha256','measurement_revision','measurement_policy')}
 payload={'environment':e,'operation':op};return payload,canonical_hash(payload)

def run(a):
 import torch,flashinfer
 if not torch.cuda.is_available() or flashinfer.__version__!='0.7.0':raise RuntimeError('CUDA/FlashInfer')
 if a.out.exists():raise FileExistsError(a.out)
 m=load();cases=[c for i,c in enumerate([x for x in m['cases'] if x['family']==a.stage]) if i%a.shards==a.shard]
 if not cases or not 0<=a.rep<3:raise ValueError('shard/repeat')
 root=Path(flashinfer.__file__).parent.parent;binding=json.loads((root/'RESOURCE_BINDING.json').read_text())
 if binding.get('selector_version')!='4.0' or not binding.get('kernel_symbol_isolation') or binding.get('default_policy')!='native':raise RuntimeError('binding')
 for rel,h in binding['modified_hashes'].items():
  if sha(root/rel)!=h:raise RuntimeError('overlay drift '+rel)
 a.out.mkdir(parents=True);a.refs.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 hw=command(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader']);parts=[x.strip() for x in hw['out'].strip().split(',')];props=torch.cuda.get_device_properties(0)
 nvcc=command([os.path.join(os.environ.get('CUDA_HOME',''),'bin','nvcc'),'--version'])
 if nvcc['rc']:raise RuntimeError('nvcc identity')
 env={'mode':a.mode,'stage':a.stage,'rep':a.rep,'shard':a.shard,'shards':a.shards,'gpu_name':parts[0],'gpu_uuid':parts[1],'driver':parts[2],'num_sms':props.multi_processor_count,'cuda':str(torch.version.cuda),'torch':str(torch.__version__),'flashinfer':flashinfer.__version__,'python':sys.version.split()[0],'cudnn':str(torch.backends.cudnn.version()),'nvcc':nvcc['out'].strip(),'measurement_policy':'paired-3-process-8-block-per-execution-mode','source_archive_sha256':os.environ['SGI_SOURCE_ARCHIVE_SHA256'],'overlay_sha256':sha(root/'RESOURCE_BINDING.json'),'measurement_revision':REVISION,'case_hash':m['case_hash'],'release_hash':m['release_hash'],'cases':cases,'profiled':False,'default_promotion':False,'source':{x.name:sha(x) for x in Path(__file__).parent.glob('*.py')}}
 decisions=json.loads(a.decisions.read_text()) if a.mode=='evaluation' else None
 if a.mode=='evaluation':env['decisions_sha256']=sha(a.decisions)
 save(a.out/'environment.json',env);rows=[];quals=[];memory=[];start=time.monotonic()
 def cell(case,dtype_name,layout,split):
  key={'case':case['id'],'dtype':dtype_name,'layout':layout,'split':split};cellid=digest(key);torch.manual_seed(int(cellid[:8],16));dtype=getattr(torch,dtype_name);qlens=case['q'];lengths=[q+c for q,c in zip(qlens,case['cached'])]
  q=torch.randn((sum(qlens),32,128),device='cuda',dtype=dtype);k=torch.randn((sum(lengths),8,128),device='cuda',dtype=dtype);v=torch.randn_like(k);ks=list(k.split(lengths));vs=list(v.split(lengths));qi=torch.tensor([0]+list(itertools.accumulate(qlens)),device='cuda',dtype=torch.int32)
  paged=None
  if layout=='paged':
   page=16;counts=[(n+page-1)//page for n in lengths];pp=torch.tensor([0]+list(itertools.accumulate(counts)),device='cuda',dtype=torch.int32);npages=sum(counts);pi=torch.arange(npages,device='cuda',dtype=torch.int32);last=torch.tensor([(n-1)%page+1 for n in lengths],device='cuda',dtype=torch.int32);kp=torch.zeros((npages,page,8,128),device='cuda',dtype=dtype);vp=torch.zeros_like(kp);off=0
   for n,pages,kr,vr in zip(lengths,counts,ks,vs):kp[off:off+pages].view(-1,8,128)[:n].copy_(kr);vp[off:off+pages].view(-1,8,128)[:n].copy_(vr);off+=pages
   paged=(page,pp,pi,last,kp,vp)
  native=Runtime(flashinfer,torch,layout,qi,ks,vs,k,v,lengths,dtype,split,0,paged);native.attach(q);np=native.plan()
  pool=candidate_pool(q=qlens,cached=case['cached'],padded_batch_size=np[0],num_qo_heads=32,num_kv_heads=8,num_sms=env['num_sms'],split_kv=bool(np[14]))
  pool_rt=Runtime(flashinfer,torch,layout,qi,ks,vs,k,v,lengths,dtype,split,2,paged);pool_rt.attach(q);pool_plan=pool_rt.plan()
  if pool_plan[:-1]!=np[:-1] or bool(pool_plan[15])!=pool.eligible:raise RuntimeError('C++/Python candidate-pool mismatch')
  del pool_rt
  cap_supported=not bool(np[14])
  cap=Runtime(flashinfer,torch,layout,qi,ks,vs,k,v,lengths,dtype,split,1,paged) if cap_supported else None
  if cap is not None:cap.attach(q);cp=cap.plan()
  else:cp=list(np)
  if np[:-1]!=cp[:-1]:raise RuntimeError('plan core')
  active=[native]+([] if cap is None else [cap])
  for rt in active:
   for _ in range(3):rt.call(q)
   torch.cuda.synchronize();rt.capture(q,1);rt.capture(q,16)
  ref=a.refs/(cellid+'.pt')
  native.call(q);torch.cuda.synchronize()
  inputs_sha256=digest([thash(q),thash(k),thash(v)])
  if ref.exists():
   baseline=torch.load(ref,map_location='cpu',weights_only=True)
   if inputs_sha256!=baseline['inputs_sha256']:raise RuntimeError('input drift')
  else:
   calls,pilots=calibrate_calls(native,q,torch);baseline={'out':native.out.cpu(),'lse':native.lse.cpu(),'inputs_sha256':inputs_sha256,'eager_calls':calls,'pilots':pilots,'hardware':hw['out']};torch.save(baseline,ref)
  if baseline['hardware']!=hw['out']:raise RuntimeError('hardware drift')
  for rt in active:
   rt.eager_calls=baseline['eager_calls'];rt.out.fill_(float('nan'));rt.lse.fill_(float('nan'));rt.call(q);torch.cuda.synchronize()
   if not torch.equal(rt.out.cpu(),baseline['out']) or not torch.equal(rt.lse.cpu(),baseline['lse']):raise RuntimeError('numerical mismatch')
   for count,g in rt.graphs.items():
    rt.out.fill_(float('nan'));rt.lse.fill_(float('nan'));g.replay();torch.cuda.synchronize()
    if not torch.equal(rt.out.cpu(),baseline['out']) or not torch.equal(rt.lse.cpu(),baseline['lse']):raise RuntimeError('graph mismatch '+str(count))
  ids={};chosen={}
  for ex in EXECUTIONS:
   payload,h=identity(env,case,dtype_name,layout,split,ex,np);ids[ex]={'payload':payload,'sha256':h}
   if decisions is not None:
    rec=decisions['records'].get(h);chosen[ex]='native' if rec is None else rec['tactic']
  quals.append({**key,'native_plan':np,'cap_plan':cp,'pool_plan':pool_plan,'cap_supported':cap_supported,'candidate_pool':pool.__dict__,'identities':ids,'chosen':chosen,'out_sha256':thash(native.out),'lse_sha256':thash(native.lse),'eager_calls':baseline['eager_calls'],'exact':True})
  blocks=8;rng=random.Random(int(cellid[:8],16)+a.rep*1009)
  for block in range(blocks):
   exs=list(EXECUTIONS);rng.shuffle(exs)
   for ex in exs:
    groups=['candidate','null'] if a.mode=='calibration' else ['chosen','oracle','null'];rng.shuffle(groups)
    for group in groups:
     if group in ('candidate','oracle'):other='cap'
     elif group=='chosen':other=chosen[ex]
     else:other='native'
     seq=('native',other,other,'native') if block%2==0 else (other,'native','native',other)
     for pos,arm in enumerate(seq):
      actual_arm='native' if arm=='native' or cap is None else 'cap';rt=native if actual_arm=='native' else cap;wall,device,calls=timed(rt,ex,q,torch);rows.append({**key,'mode':a.mode,'rep':a.rep,'block':block,'execution_mode':ex,'comparison_group':group,'position':pos,'role':'A' if ((block%2==0 and pos in (0,3)) or (block%2==1 and pos in (1,2))) else 'B','arm':arm,'actual_arm':actual_arm,'wall_us':wall,'device_us':device,'kernel_calls':calls,'identity_sha256':ids[ex]['sha256']})
  save(a.out/'progress.json',{'key':key,'rows':len(rows),'qualifications':len(quals),'elapsed':time.monotonic()-start});print('CELL',a.mode,a.rep,key,len(rows),flush=True)
 try:
  for case,dtype,layout,split in itertools.product(cases,m['dtypes'],m['layouts'],m['splits']):
   cell(case,dtype,layout,split);allocated_before=int(torch.cuda.memory_allocated());gc.collect();torch.cuda.empty_cache();free,total=torch.cuda.mem_get_info();memory.append({'case':case['id'],'dtype':dtype,'layout':layout,'split':split,'allocated_before_gc':allocated_before,'allocated_after_gc':int(torch.cuda.memory_allocated()),'reserved_after_gc':int(torch.cuda.memory_reserved()),'free_device_bytes':int(free),'total_device_bytes':int(total)})
  save(a.out/'rows.json',rows);save(a.out/'qualification.json',quals);save(a.out/'memory.json',memory);save(a.out/'complete.json',{'complete':True,'rows':len(rows),'qualifications':len(quals),'files':{p.name:sha(p) for p in a.out.glob('*.json')},'release_evidence':a.mode=='evaluation','default_promotion':False})
 except BaseException as e:
  save(a.out/'partial_rows.json',rows);save(a.out/'qualification.json',quals);save(a.out/'partial_memory.json',memory);save(a.out/'failure.json',{'error':type(e).__name__,'message':str(e),'last_key':locals().get('key')});raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['calibration','evaluation'],required=True);p.add_argument('--stage',choices=['canary','release'],required=True);p.add_argument('--rep',type=int,required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,required=True);p.add_argument('--refs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--decisions',type=Path);a=p.parse_args();
 if a.mode=='evaluation' and not a.decisions: p.error('--decisions required')
 run(a)
