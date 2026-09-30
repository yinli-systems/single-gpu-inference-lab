"""Selector-v3.2 deployment-matched pristine and same-binary paired measurements."""
from __future__ import annotations
import argparse,hashlib,itertools,json,math,os,random,subprocess,sys,time,gc
from pathlib import Path
from manifest import load,digest,plan_guard,tactic_identity
from prepare_guarded import EXPECTED

BACKEND_FILES=tuple(sorted(EXPECTED))
ARMS=("off","cap","guarded")
EXECUTIONS=("eager_full_call","graph1_replay","graph16_replay")
KERNEL_CALLS={"eager_full_call":16,"graph1_replay":16,"graph16_replay":16}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def command(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=20);return dict(rc=p.returncode,out=p.stdout,err=p.stderr)
def thash(t):return hashlib.sha256(t.contiguous().view(__import__('torch').uint8).cpu().numpy().tobytes()).hexdigest()
def backend_digest(root):return digest({rel:sha(Path(root)/rel) for rel in BACKEND_FILES})

def oracle(q,ks,vs,qs,out,lse,dtype):
 import torch
 absmax=0.;squared=0.;elements=0;lsmax=0.;vectors=0;offset=0
 for nq,k,v in zip(qs,ks,vs):
  ids=list(range(nq)) if sum(qs)<=128 else sorted({0,nq//8,nq//4,nq//2,3*nq//4,7*nq//8,nq-1})
  ix=torch.tensor(ids,device=q.device);L=k.shape[0]
  qq=q[offset+ix].float().transpose(0,1);kk=k.float().repeat_interleave(4,dim=1).transpose(0,1);vv=v.float().repeat_interleave(4,dim=1).transpose(0,1)
  scores=(qq@kk.transpose(-1,-2))/math.sqrt(q.shape[-1]);scores.masked_fill_((torch.arange(L,device=q.device)[None,:]>(ix+L-nq)[:,None])[None],float('-inf'))
  ref=(scores.softmax(-1)@vv).transpose(0,1);expected_lse=scores.logsumexp(-1).transpose(0,1)/math.log(2)
  actual=out[offset+ix].float();difference=actual-ref
  torch.testing.assert_close(actual,ref,atol=.005 if dtype=='float16' else .02,rtol=.02 if dtype=='float16' else .04)
  torch.testing.assert_close(lse[offset+ix],expected_lse,atol=.01,rtol=.01)
  absmax=max(absmax,float(difference.abs().max()));squared+=float(difference.square().sum());elements+=difference.numel();lsmax=max(lsmax,float((lse[offset+ix]-expected_lse).abs().max()));vectors+=len(ids)*32;offset+=nq
 return dict(max_abs=absmax,rmse=math.sqrt(squared/elements),lse_max_abs=lsmax,vectors=vectors,elements=elements,full_reference=(sum(qs)<=128))

class Runtime:
 def __init__(self,flashinfer,torch,layout,qi,ks,vs,k,v,ls,dtype,split,arm,candidate,paged_bundle=None):
  self.torch=torch;self.layout=layout;self.arm=arm;self.candidate=candidate
  self.ws=torch.empty(512*1024**2,device='cuda',dtype=torch.uint8);self.out=None;self.lse=None;self.graphs={}
  plan_kw=dict(causal=True,q_data_type=dtype,kv_data_type=dtype,disable_split_kv=(split=='unsplit'))
  if layout=='ragged':
   ki=torch.tensor([0]+list(itertools.accumulate(ls)),device='cuda',dtype=torch.int32)
   self.wrapper=flashinfer.BatchPrefillWithRaggedKVCacheWrapper(self.ws,backend='fa2')
   self._plan=lambda:self.wrapper.plan(qi,ki,32,8,128,**plan_kw)
   self._call=lambda q:self.wrapper.run(q,k,v,out=self.out,lse=self.lse,return_lse=True)
   self.keep=(ki,)
  else:
   if paged_bundle is None:raise RuntimeError('shared paged bundle required')
   page,pp,pi,last,kp,vp=paged_bundle
   self.wrapper=flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.ws,'NHD',backend='fa2')
   self._plan=lambda:self.wrapper.plan(qi,pp,pi,last,32,8,128,page,**plan_kw)
   self._call=lambda q:self.wrapper.run(q,(kp,vp),out=self.out,lse=self.lse,return_lse=True)
   self.keep=(pp,pi,last,kp,vp)
 def attach(self,q):
  self.out=self.torch.empty_like(q);self.lse=self.torch.empty((q.shape[0],32),device='cuda',dtype=self.torch.float32)
 def plan(self):
  self._plan();info=[int(x) for x in self.wrapper._plan_info]
  if self.candidate:
   if len(info)!=16:raise RuntimeError('candidate plan vector size')
   original=bool(info[15])
   if self.arm=='off':self.wrapper._plan_info[15]=0
   elif self.arm=='cap':self.wrapper._plan_info[15]=1
   elif self.arm!='guarded':raise ValueError(self.arm)
   info=[int(x) for x in self.wrapper._plan_info]
   return info,original
  if len(info)!=15:raise RuntimeError('pristine plan vector size')
  return info,False
 def call(self,q):return self._call(q)
 def capture(self,q,count):
  self.plan();g=self.torch.cuda.CUDAGraph()
  with self.torch.cuda.graph(g):
   for _ in range(count):self.call(q)
  self.graphs[count]=g
 def pointers(self,q,k,v):return [x.data_ptr() for x in (q,k,v,self.out,self.lse,self.ws,self.wrapper._int_workspace_buffer)]

def timed_eager(rt,q,torch):
 ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True)
 rt.plan();rt.call(q);torch.cuda.synchronize();gc.disable();tick=time.perf_counter_ns();ev0.record()
 for _ in range(16):rt.plan();rt.call(q)
 ev1.record();ev1.synchronize();wall=(time.perf_counter_ns()-tick)/1000/16;device=ev0.elapsed_time(ev1)*1000/16;gc.enable()
 return wall,device

def timed_graph(rt,count,replays,q,torch):
 g=rt.graphs[count];g.replay();torch.cuda.synchronize();ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True)
 gc.disable();tick=time.perf_counter_ns();ev0.record()
 for _ in range(replays):g.replay()
 ev1.record();ev1.synchronize();wall=(time.perf_counter_ns()-tick)/1000/(count*replays);device=ev0.elapsed_time(ev1)*1000/(count*replays);gc.enable()
 return wall,device

def timed(rt,execution,q,torch):
 if execution=='eager_full_call':return timed_eager(rt,q,torch)
 if execution=='graph1_replay':return timed_graph(rt,1,16,q,torch)
 if execution=='graph16_replay':return timed_graph(rt,16,1,q,torch)
 raise ValueError(execution)

def run(a):
 import torch,flashinfer
 if not torch.cuda.is_available() or flashinfer.__version__!='0.7.0':raise RuntimeError('real CUDA and pinned 0.7.0 required')
 if a.out.exists():raise FileExistsError('preserve evidence')
 if a.mode not in ('pristine','paired'):raise ValueError(a.mode)
 if not 0<=a.rep<3 or not 0<=a.shard<a.shards<=8:raise ValueError('bounded repeat/shard')
 m=load();wanted=('canary',) if a.stage=='canary' else ('release',);allcases=[c for c in m['cases'] if c['family'] in wanted];cases=[c for i,c in enumerate(allcases) if i%a.shards==a.shard]
 if not cases:raise ValueError('empty shard')
 pkg=Path(flashinfer.__file__).parent;root=pkg.parent;candidate=a.mode=='paired'
 if candidate:
  binding=json.loads((root/'RESOURCE_BINDING.json').read_text());
  if binding.get('mode')!='guarded' or binding.get('base_hashes')!=EXPECTED:raise RuntimeError('candidate binding mismatch')
  for rel,h in binding['modified_hashes'].items():
   if sha(root/rel)!=h:raise RuntimeError('candidate source drift '+rel)
 else:
  for rel,h in EXPECTED.items():
   if sha(root/rel)!=h:raise RuntimeError('pristine source drift '+rel)
 a.out.mkdir(parents=True);a.refs.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(affinity[:min(2,len(affinity))]));num_sm=torch.cuda.get_device_properties(0).multi_processor_count
 env=dict(gpu=torch.cuda.get_device_name(),num_sm=num_sm,torch=torch.__version__,cuda=torch.version.cuda,flashinfer=flashinfer.__version__,mode=a.mode,stage=a.stage,rep=a.rep,shard=a.shard,shards=a.shards,case_hash=m['case_hash'],cases=cases,
  source={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},hardware=command(['nvidia-smi','--query-gpu=name,uuid,driver_version,power.limit','--format=csv']),processes=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv']),affinity=list(os.sched_getaffinity(0)),
  partition=os.environ.get('SLURM_JOB_PARTITION'),profiled=False,profiling_mode='unprofiled-release-timing',nsight_compute_excluded=True,official_overlay_sha256=os.environ.get('SGI_OFFICIAL_MANIFEST_SHA256'),source_archive_sha256=os.environ.get('SGI_SOURCE_ARCHIVE_SHA256'),candidate_binary_sha256=backend_digest(root),measurement_contract='v3.2 paired candidate; eager full plan+run wall; graph replay wall/device',full_model=False,default_promotion=False)
 expected_gpu={'gpu_4090':'NVIDIA GeForce RTX 4090','gpu_5090':'NVIDIA GeForce RTX 5090'}.get(env['partition'])
 if env['hardware']['rc'] or expected_gpu is None or env['gpu']!=expected_gpu or not env['official_overlay_sha256'] or not env['source_archive_sha256']:raise RuntimeError('environment/provenance mismatch')
 save(a.out/'environment.json',env);rows=[];quals=[];started=time.monotonic()
 try:
  for case,dtype_name,layout,split in itertools.product(cases,m['dtypes'],m['layouts'],m['splits']):
   key=dict(case=case['id'],family=case['family'],dtype=dtype_name,layout=layout,split=split);cellid=digest(key);torch.manual_seed(int(cellid[:8],16));dtype=getattr(torch,dtype_name)
   qs=case['q'];ls=[q+k for q,k in zip(qs,case['cached'])];q=torch.randn((sum(qs),32,128),device='cuda',dtype=dtype);k=torch.randn((sum(ls),8,128),device='cuda',dtype=dtype);v=torch.randn_like(k);ks=list(k.split(ls));vs=list(v.split(ls));qi=torch.tensor([0]+list(itertools.accumulate(qs)),device='cuda',dtype=torch.int32)
   paged_bundle=None
   if layout=='paged':
    page=16;counts=[(L+page-1)//page for L in ls];npages=sum(counts);pp=torch.tensor([0]+list(itertools.accumulate(counts)),device='cuda',dtype=torch.int32)
    order=torch.randperm(npages,device='cuda');pi=order.to(torch.int32);last=torch.tensor([(L-1)%page+1 for L in ls],device='cuda',dtype=torch.int32)
    kp=torch.zeros((npages,page,8,128),device='cuda',dtype=dtype);vp=torch.zeros_like(kp);off=0
    for L,n,kr,vr in zip(ls,counts,ks,vs):
     kl=torch.zeros((n*page,8,128),device='cuda',dtype=dtype);vl=torch.zeros_like(kl);kl[:L].copy_(kr);vl[:L].copy_(vr);kp[order[off:off+n]]=kl.view(n,page,8,128);vp[order[off:off+n]]=vl.view(n,page,8,128);off+=n
    paged_bundle=(page,pp,pi,last,kp,vp)
   arm_names=('pristine',) if not candidate else ARMS;runtimes={arm:Runtime(flashinfer,torch,layout,qi,ks,vs,k,v,ls,dtype,split,arm,candidate,paged_bundle) for arm in arm_names}
   for rt in runtimes.values():rt.attach(q)
   plans={};original_guard={};
   for arm,rt in runtimes.items():
    info,orig=rt.plan();plans[arm]=info;original_guard[arm]=orig
    for _ in range(3):rt.call(q)
    torch.cuda.synchronize();rt.capture(q,1);rt.capture(q,16);rt.graphs[1].replay();rt.graphs[16].replay();torch.cuda.synchronize()
   if candidate:
    cores={tuple(info[:-1]) for info in plans.values()}
    if len(cores)!=1 or plans['off'][-1]!=0 or plans['cap'][-1]!=1:raise RuntimeError('candidate plan pairing contract')
    expected_guard=plan_guard(qs,case['cached'],plans['guarded'][0],32,8,num_sm,bool(plans['guarded'][14]))
    if bool(plans['guarded'][15])!=expected_guard or original_guard['guarded']!=expected_guard:raise RuntimeError('guard decision mismatch')
   else:expected_guard=False
   inputs=digest([thash(q),thash(k),thash(v)]);ref_path=a.refs/(cellid+'.pt')
   arm_quals={};baseline=None
   if not candidate:
    rt=runtimes['pristine'];rt.plan();rt.call(q);torch.cuda.synchronize();reference=oracle(q,ks,vs,qs,rt.out,rt.lse,dtype_name);baseline=dict(out=rt.out.cpu(),lse=rt.lse.cpu(),inputs=inputs,FP32=reference);torch.save(baseline,ref_path)
   else:
    if not ref_path.exists():raise RuntimeError('missing independent pristine reference')
    baseline=torch.load(ref_path,map_location='cpu',weights_only=True)
    if baseline['inputs']!=inputs:raise RuntimeError('cross-mode inputs differ')
    reference=baseline.get('FP32')
   for arm,rt in runtimes.items():
    rt.plan();rt.call(q);torch.cuda.synchronize();cpuout=rt.out.cpu();cpulse=rt.lse.cpu();exact=torch.equal(cpuout,baseline['out']) and torch.equal(cpulse,baseline['lse'])
    if not exact:raise RuntimeError('full output/LSE not bit-exact '+arm)
    for count,g in rt.graphs.items():g.replay();torch.cuda.synchronize();
    if not torch.equal(rt.out.cpu(),baseline['out']) or not torch.equal(rt.lse.cpu(),baseline['lse']):raise RuntimeError('graph parity '+arm)
    arm_quals[arm]=dict(out_sha256=thash(rt.out),lse_sha256=thash(rt.lse),pristine_exact=True,plan_info=plans[arm])
   hwline=env['hardware']['out'].strip().splitlines()[1].split(',');nvcc=command([os.path.join(os.environ.get('CUDA_HOME',''),'bin','nvcc'),'--version'])
   if nvcc['rc']:raise RuntimeError('missing nvcc identity')
   environment_identity=dict(gpu_name=hwline[0].strip(),gpu_uuid=hwline[1].strip(),driver=hwline[2].strip(),num_sms=num_sm,cuda=str(torch.version.cuda),torch=str(torch.__version__),flashinfer=str(flashinfer.__version__),python=sys.version.split()[0],cudnn=str(torch.backends.cudnn.version()),nvcc=nvcc['out'].strip(),backend_sha256=env['candidate_binary_sha256'],official_overlay_sha256=env['official_overlay_sha256'],source_archive_sha256=env['source_archive_sha256'],selector_version='3.2',measurement_policy='eager recurring full plan+run wall; graph replay wall/device; paired candidate arms',profiling_mode='unprofiled')
   ordered_pairs_sha256=digest(list(zip(qs,case['cached'])));identities={}
   for arm,info in plans.items():
    identities[arm]={}
    for execution in EXECUTIONS:
     identities[arm][execution]=tactic_identity(environment=environment_identity,operation=dict(execution_mode=execution,backend='fa2',causal=True,layout=layout,dtype=dtype_name,requested_split=split,actual_split='split' if bool(info[14]) else 'unsplit',num_qo_heads=32,num_kv_heads=8,head_dim_qk=128,head_dim_vo=128,page_size=1 if layout=='ragged' else 16,q=list(qs),cached=list(case['cached']),ordered_pairs_sha256=ordered_pairs_sha256,plan_signature=info,window_repeats=16 if execution!='graph16_replay' else 1,candidate_binary_sha256=env['candidate_binary_sha256']))
   qual=dict(key,inputs_sha256=inputs,ordered_pairs_sha256=ordered_pairs_sha256,expected_selector=case['expected_selector'],guard_expected=expected_guard,arms=arm_quals,tactic_identities=identities,FP32=reference,candidate_plan_core_equal=(len({tuple(x[:-1]) for x in plans.values()})==1 if candidate else None),eager_graph_exact=True)
   quals.append(qual);blocks=2 if a.stage=='canary' else m['blocks'];rng=random.Random(int(cellid[:8],16)+a.rep*1009)
   ptrs={arm:rt.pointers(q,k,v) for arm,rt in runtimes.items()}
   for block in range(blocks):
    executions=list(EXECUTIONS);rng.shuffle(executions)
    for execution in executions:
     if not candidate:
      roles=('a','b','b','a') if block%2==0 else ('b','a','a','b')
      for pos,role in enumerate(roles):
       wall,device=timed(runtimes['pristine'],execution,q,torch);rows.append(dict(key,mode='pristine',arm='pristine',comparison_group='position',rep=a.rep,block=block,execution_mode=execution,position=pos,role=role,wall_us=wall,device_us=device,kernel_calls=KERNEL_CALLS[execution],tactic_identity=identities['pristine'][execution]))
     else:
      groups=['guarded','cap'];
      if block%2:groups.reverse()
      for group in groups:
       other=group;seq=('off',other,other,'off') if block%2==0 else (other,'off','off',other)
       for pos,arm in enumerate(seq):
        wall,device=timed(runtimes[arm],execution,q,torch);rows.append(dict(key,mode='paired',arm=arm,comparison_group=group,rep=a.rep,block=block,execution_mode=execution,position=pos,role='A' if arm=='off' else 'B',wall_us=wall,device_us=device,kernel_calls=KERNEL_CALLS[execution],tactic_identity=identities[arm][execution]))
     for arm,rt in runtimes.items():
      if ptrs[arm]!=rt.pointers(q,k,v):raise RuntimeError('storage changed '+arm)
   save(a.out/'progress.json',dict(key,rows=len(rows),qualifications=len(quals),elapsed=time.monotonic()-started));print('CELL',a.mode,a.rep,key,len(rows),flush=True)
   del runtimes,q,k,v,ks,vs,baseline;torch.cuda.empty_cache()
  cells=len(cases)*2*2*2;expected=cells*(2 if a.stage=='canary' else m['blocks'])*3*(4 if not candidate else 8)
  if len(rows)!=expected:raise RuntimeError(f'incomplete matrix {len(rows)} != {expected}')
  save(a.out/'measurements.json',rows);save(a.out/'qualification.json',quals);save(a.out/'complete.json',dict(complete=True,rows=len(rows),expected=expected,qualifications=len(quals),files={p.name:sha(p) for p in a.out.glob('*.json')},performance_promotion=False,serving_promotion=False));print('COMPLETE',a.mode,a.stage,a.rep,len(rows),flush=True)
 except BaseException as exc:
  save(a.out/'partial_measurements.json',rows);save(a.out/'qualification.json',quals);save(a.out/'failure.json',dict(type=type(exc).__name__,message=str(exc)));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--refs',type=Path,required=True);p.add_argument('--mode',choices=['pristine','paired'],required=True);p.add_argument('--stage',choices=['canary','release'],required=True);p.add_argument('--rep',type=int,default=0);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);run(p.parse_args())
