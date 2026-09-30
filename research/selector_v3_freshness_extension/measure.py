"""Real native FA2 resource-policy evaluation, full-output parity and charged cycles."""
from __future__ import annotations
import argparse,hashlib,itertools,json,math,os,random,subprocess,sys,time
from pathlib import Path
from manifest import load,digest,plan_guard,tactic_identity
from prepare_guarded import EXPECTED

BACKEND_FILES=tuple(sorted(EXPECTED))
def backend_digest(root):
 return digest({rel:sha(Path(root)/rel) for rel in BACKEND_FILES})

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def command(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=15);return dict(rc=p.returncode,out=p.stdout,err=p.stderr)
def thash(t):return hashlib.sha256(t.contiguous().view(__import__('torch').uint8).cpu().numpy().tobytes()).hexdigest()

def oracle(q,ks,vs,qs,out,lse,dtype):
 import torch
 absmax=0.;squared=0.;elements=0;lsmax=0.;vectors=0;offset=0
 for nq,k,v in zip(qs,ks,vs):
  ids=list(range(nq)) if sum(qs)<=128 else sorted({0,nq//8,nq//4,nq//2,3*nq//4,7*nq//8,nq-1})
  ix=torch.tensor(ids,device=q.device);L=k.shape[0]
  qq=q[offset+ix].float().transpose(0,1)
  kk=k.float().repeat_interleave(4,dim=1).transpose(0,1)
  vv=v.float().repeat_interleave(4,dim=1).transpose(0,1)
  scores=(qq@kk.transpose(-1,-2))/math.sqrt(q.shape[-1])
  scores.masked_fill_((torch.arange(L,device=q.device)[None,:]>(ix+L-nq)[:,None])[None],float('-inf'))
  ref=(scores.softmax(-1)@vv).transpose(0,1);expected_lse=scores.logsumexp(-1).transpose(0,1)/math.log(2)
  actual=out[offset+ix].float();difference=actual-ref
  torch.testing.assert_close(actual,ref,atol=.005 if dtype=='float16' else .02,rtol=.02 if dtype=='float16' else .04)
  torch.testing.assert_close(lse[offset+ix],expected_lse,atol=.01,rtol=.01)
  absmax=max(absmax,float(difference.abs().max()));squared+=float(difference.square().sum());elements+=difference.numel()
  lsmax=max(lsmax,float((lse[offset+ix]-expected_lse).abs().max()));vectors+=len(ids)*32;offset+=nq
 return dict(max_abs=absmax,rmse=math.sqrt(squared/elements),lse_max_abs=lsmax,vectors=vectors,elements=elements,full_reference=(sum(qs)<=128))

def run(a):
 import torch,flashinfer
 if not torch.cuda.is_available() or flashinfer.__version__!='0.7.0':raise RuntimeError('real CUDA and pinned 0.7.0 required')
 if a.out.exists():raise FileExistsError('preserve all evidence')
 if not 0<=a.rep<3 or not 0<=a.shard<a.shards<=8:raise ValueError('invalid bounded repeat/shard')
 m=load();wanted=('canary',) if a.stage=='canary' else ('release',)
 cases=[c for c in m['cases'] if c['family'] in wanted]
 cases=[c for i,c in enumerate(cases) if i%a.shards==a.shard]
 if not cases:raise ValueError('empty shard')
 pkg=Path(flashinfer.__file__).parent;root=pkg.parent;head=pkg/'data/include/flashinfer/attention/prefill.cuh'
 if a.mode=='pristine':
  mismatched={rel:(EXPECTED[rel],sha(root/rel)) for rel in EXPECTED if sha(root/rel)!=EXPECTED[rel]}
  if mismatched:raise RuntimeError('pristine source mismatch '+json.dumps(mismatched,sort_keys=True))
 else:
  binding=json.loads((root/'RESOURCE_BINDING.json').read_text())
  if binding.get('mode')!=a.mode or binding.get('base_hashes')!=EXPECTED:raise RuntimeError('candidate binding identity mismatch')
  mismatched={rel:(expected,sha(root/rel)) for rel,expected in binding.get('modified_hashes',{}).items() if sha(root/rel)!=expected}
  if mismatched:raise RuntimeError('candidate source mismatch '+json.dumps(mismatched,sort_keys=True))
 a.out.mkdir(parents=True);a.refs.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(affinity[:min(2,len(affinity))]))
 num_sm=torch.cuda.get_device_properties(0).multi_processor_count
 env=dict(gpu=torch.cuda.get_device_name(),num_sm=num_sm,torch=torch.__version__,cuda=torch.version.cuda,flashinfer=flashinfer.__version__,
  mode=a.mode,stage=a.stage,rep=a.rep,shard=a.shard,shards=a.shards,case_hash=m['case_hash'],cases=cases,
  header_sha256=sha(head),source={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
  hardware=command(['nvidia-smi','--query-gpu=name,uuid,driver_version,power.limit','--format=csv']),
  processes=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv']),
  affinity=list(os.sched_getaffinity(0)),clocks_locked=False,whole_node_exclusive=False,
  partition=os.environ.get('SLURM_JOB_PARTITION'),profiled=False,profiling_mode='unprofiled-release-timing',nsight_compute_excluded=True,
  official_overlay_sha256=os.environ.get('SGI_OFFICIAL_MANIFEST_SHA256'),source_archive_sha256=os.environ.get('SGI_SOURCE_ARCHIVE_SHA256'),
  output_seed='case/layout/dtype/split fixed across modes and process repeats',full_model=False,default_promotion=False)
 if env['hardware']['rc']:raise RuntimeError('missing hardware binding')
 if not env['official_overlay_sha256'] or not env['source_archive_sha256']:raise RuntimeError('missing immutable provenance binding')
 expected_gpu={'gpu_4090':'NVIDIA GeForce RTX 4090','gpu_5090':'NVIDIA GeForce RTX 5090'}.get(env['partition'])
 if expected_gpu is None or env['gpu']!=expected_gpu:raise RuntimeError('GPU family/partition mismatch')
 save(a.out/'environment.json',env);rows=[];quals=[];trace_done=set();started=time.monotonic()
 try:
  for case,dtype_name,layout,split in itertools.product(cases,m['dtypes'],m['layouts'],m['splits']):
   key=dict(case=case['id'],family=case['family'],dtype=dtype_name,layout=layout,split=split)
   cellid=digest(key);torch.manual_seed(int(cellid[:8],16));dtype=getattr(torch,dtype_name)
   qs=case['q'];ls=[q+k for q,k in zip(qs,case['cached'])]
   q=torch.randn((sum(qs),32,128),device='cuda',dtype=dtype)
   k=torch.randn((sum(ls),8,128),device='cuda',dtype=dtype);v=torch.randn_like(k)
   ks=list(k.split(ls));vs=list(v.split(ls))
   qi=torch.tensor([0]+list(itertools.accumulate(qs)),device='cuda',dtype=torch.int32)
   ws=torch.empty(512*1024**2,device='cuda',dtype=torch.uint8)
   plan_kw=dict(causal=True,q_data_type=dtype,kv_data_type=dtype,disable_split_kv=(split=='unsplit'))
   if layout=='ragged':
    ki=torch.tensor([0]+list(itertools.accumulate(ls)),device='cuda',dtype=torch.int32)
    w=flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws,backend='fa2')
    def plan():w.plan(qi,ki,32,8,128,**plan_kw)
    paged=None
   else:
    page=16;counts=[(L+page-1)//page for L in ls];npages=sum(counts)
    pp=torch.tensor([0]+list(itertools.accumulate(counts)),device='cuda',dtype=torch.int32)
    order=torch.randperm(npages,device='cuda');pi=order.to(torch.int32)
    last=torch.tensor([(L-1)%page+1 for L in ls],device='cuda',dtype=torch.int32)
    kp=torch.zeros((npages,page,8,128),device='cuda',dtype=dtype);vp=torch.zeros_like(kp);off=0
    for L,n,kr,vr in zip(ls,counts,ks,vs):
     kl=torch.zeros((n*page,8,128),device='cuda',dtype=dtype);vl=torch.zeros_like(kl);kl[:L].copy_(kr);vl[:L].copy_(vr)
     kp[order[off:off+n]]=kl.view(n,page,8,128);vp[order[off:off+n]]=vl.view(n,page,8,128);off+=n
    paged=(kp,vp)
    w=flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws,'NHD',backend='fa2')
    def plan():w.plan(qi,pp,pi,last,32,8,128,page,**plan_kw)
   plan();out=torch.empty_like(q);lse=torch.empty((sum(qs),32),device='cuda',dtype=torch.float32)
   def call():
    return w.run(q,k,v,out=out,lse=lse,return_lse=True) if layout=='ragged' else w.run(q,paged,out=out,lse=lse,return_lse=True)
   for _ in range(5):call()
   torch.cuda.synchronize();eager=out.clone();el=lse.clone();reference=oracle(q,ks,vs,qs,out,lse,dtype_name)
   inputs=digest([thash(q),thash(k),thash(v)]);opath=a.refs/(cellid+'.pt')
   if a.mode=='pristine' and not opath.exists():torch.save(dict(out=out.cpu(),lse=lse.cpu(),inputs=inputs),opath)
   if not opath.exists():raise RuntimeError('missing independently run pristine reference')
   baseline=torch.load(opath,map_location='cpu',weights_only=True)
   if baseline['inputs']!=inputs:raise RuntimeError('inputs differ across modes')
   cpuout=out.cpu();cpulse=lse.cpu();exact=torch.equal(cpuout,baseline['out']) and torch.equal(cpulse,baseline['lse'])
   if not exact:raise RuntimeError('resource-only full output/LSE not bit-exact')
   torch.testing.assert_close(cpuout,baseline['out'],atol=.005 if dtype_name=='float16' else .02,rtol=.02 if dtype_name=='float16' else .04)
   torch.testing.assert_close(cpulse,baseline['lse'],atol=.01,rtol=.01)
   err=(cpuout.float()-baseline['out'].float()).abs();maximum=float(err.max())
   graphs={}
   for count in (1,16):
    gr=torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
     for _ in range(count):call()
    out.fill_(float('nan'));lse.fill_(float('nan'));gr.replay();torch.cuda.synchronize()
    torch.testing.assert_close(out,eager,atol=0,rtol=0);torch.testing.assert_close(lse,el,atol=0,rtol=0);graphs[count]=gr
   info=[int(x) for x in w._plan_info]
   expected_guard=plan_guard(qs,case['cached'],info[0],32,8,num_sm,bool(info[14]))
   hwline=env['hardware']['out'].strip().splitlines()[1].split(',')
   nvcc=command([os.path.join(os.environ.get('CUDA_HOME',''), 'bin', 'nvcc'),'--version'])
   if nvcc['rc']:raise RuntimeError('missing nvcc identity')
   environment_identity=dict(gpu_name=hwline[0].strip(),gpu_uuid=hwline[1].strip(),driver=hwline[2].strip(),num_sms=num_sm,
    cuda=str(torch.version.cuda),torch=str(torch.__version__),flashinfer=str(flashinfer.__version__),python=sys.version.split()[0],
    cudnn=str(torch.backends.cudnn.version()),nvcc=nvcc['out'].strip(),backend_sha256=backend_digest(pkg.parent),
    official_overlay_sha256=env['official_overlay_sha256'],source_archive_sha256=env['source_archive_sha256'],selector_version=3,
    measurement_policy='unprofiled full eager-call wall/device timing and deployment-matched CUDA-graph replay; no cross-mode reuse',profiling_mode='unprofiled')
   ordered_pairs_sha256=digest(list(zip(qs,case['cached'])))
   identities={}
   for execution_mode in ('eager','graph1','graph16'):
    identities[execution_mode]=tactic_identity(environment=environment_identity,operation=dict(execution_mode=execution_mode,backend='fa2',causal=True,layout=layout,dtype=dtype_name,
     requested_split=split,actual_split='split' if bool(info[14]) else 'unsplit',num_qo_heads=32,num_kv_heads=8,head_dim_qk=128,head_dim_vo=128,
     page_size=1 if layout=='ragged' else 16,q=list(qs),cached=list(case['cached']),ordered_pairs_sha256=ordered_pairs_sha256,plan_signature=info))
   qual=dict(key,guard_expected=expected_guard,resource_cap_plan=bool(info[15]) if len(info)>15 else False,inputs_sha256=inputs,out_sha256=thash(out),lse_sha256=thash(lse),pristine_exact=bool(exact),
    pristine_full_max_abs=maximum,pristine_full_mean_abs=float(err.mean()),FP32=reference,eager_graph_exact=True,plan_info=info,
    environment_identity=environment_identity,tactic_identities=identities,ordered_pairs_sha256=ordered_pairs_sha256)
   quals.append(qual);ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True)
   ev0.record();ev1.record();ev1.synchronize();ptrs=[x.data_ptr() for x in (q,k,v,out,lse,ws,w._int_workspace_buffer)]
   blocks=2 if a.stage=='canary' else m['blocks'];rng=random.Random(int(cellid[:8],16)+a.rep)
   for b in range(blocks):
    execution=[0,1,16];rng.shuffle(execution)
    for n in execution:
     runner=call if n==0 else graphs[n].replay
     execution_mode='eager' if n==0 else ('graph1' if n==1 else 'graph16')
     labels=('main','repeat','repeat','main') if b%2==0 else ('repeat','main','main','repeat')
     for pos,label in enumerate(labels):
      runner();torch.cuda.synchronize();tick=time.perf_counter();ev0.record();runner();ev1.record();ev1.synchronize()
      wall=(time.perf_counter()-tick)*1e6;device=ev0.elapsed_time(ev1)*1000
      tick=time.perf_counter();plan();runner();torch.cuda.synchronize();cycle=(time.perf_counter()-tick)*1e6
      if not all(math.isfinite(x) and x>0 for x in (wall,device,cycle)):raise RuntimeError('invalid duration')
      rows.append(dict(key,mode=a.mode,rep=a.rep,block=b,calls=n,execution_mode=execution_mode,label=label,position=pos,run_device_us=device,run_wall_us=wall,cycle_us=cycle,tactic_identity=identities[execution_mode]))
     torch.testing.assert_close(out,eager,atol=0,rtol=0);torch.testing.assert_close(lse,el,atol=0,rtol=0)
     if ptrs!=[x.data_ptr() for x in (q,k,v,out,lse,ws,w._int_workspace_buffer)]:raise RuntimeError('plan storage changed')
   save(a.out/'progress.json',dict(key,rows=len(rows),qualifications=len(quals),elapsed=time.monotonic()-started))
   print('CELL',a.mode,a.rep,key,len(rows),flush=True)
   del graphs,gr,w,ws,q,k,v,ks,vs,out,lse,eager,el,paged,baseline,cpuout,cpulse,err
   torch.cuda.empty_cache()
  expected=len(cases)*2*2*2*(2 if a.stage=='canary' else m['blocks'])*3*4
  if len(rows)!=expected:raise RuntimeError('incomplete matrix')
  save(a.out/'measurements.json',rows);save(a.out/'qualification.json',quals)
  save(a.out/'complete.json',dict(complete=True,rows=len(rows),expected=expected,qualifications=len(quals),
    files={p.name:sha(p) for p in a.out.glob('*.json')},performance_promotion=False,serving_promotion=False))
  print('COMPLETE',a.mode,a.stage,a.rep,len(rows),flush=True)
 except BaseException as exc:
  save(a.out/'partial_measurements.json',rows);save(a.out/'qualification.json',quals);save(a.out/'failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--refs',type=Path,required=True)
 p.add_argument('--mode',choices=['pristine','off','cap','guarded'],required=True);p.add_argument('--stage',choices=['canary','release'],required=True)
 p.add_argument('--rep',type=int,default=0);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);run(p.parse_args())
