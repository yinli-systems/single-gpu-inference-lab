"""Numerical lifecycle gate; real paged cache, repeated graphs, independent streams.
No performance timings. Each concurrent wrapper owns independent storage.
"""
import argparse,hashlib,itertools,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from measure import sha,save,thash,oracle
from prepare import EXPECTED

def run(a):
 import torch,flashinfer
 if a.out.exists():raise FileExistsError('preserve qualification')
 if not torch.cuda.is_available() or flashinfer.__version__!='0.7.0':raise RuntimeError('wrong environment')
 a.out.mkdir(parents=True);a.refs.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 pkg=Path(flashinfer.__file__).parent;h=sha(pkg/'data/include/flashinfer/attention/prefill.cuh')
 if a.mode=='pristine':
  if h!=EXPECTED:raise RuntimeError('unqualified pristine')
 else:
  b=json.loads((pkg.parent/'RESOURCE_BINDING.json').read_text())
  if b['mode']!=a.mode or h!=b['modified_sha256']:raise RuntimeError('unqualified candidate')
 records=[]
 try:
  for dtype_name,shared in itertools.product(['float16','bfloat16'],[False,True]):
   contexts=[]
   for sid in range(2):
    torch.manual_seed(963301+sid+100*int(shared));dtype=getattr(torch,dtype_name);qs=[256,128];page=16
    q=torch.randn((384,32,128),device='cuda',dtype=dtype)
    kv=torch.randn((512,2,page,8,128),device='cuda',dtype=dtype)
    order=torch.randperm(512,device='cuda').to(torch.int32)
    ws=torch.empty(512*1024**2,device='cuda',dtype=torch.uint8)
    qb=torch.empty(3,device='cuda',dtype=torch.int32);pb=torch.empty_like(qb);ib=torch.empty(512,device='cuda',dtype=torch.int32);lb=torch.empty(2,device='cuda',dtype=torch.int32)
    w=flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws,'NHD',use_cuda_graph=True,
     qo_indptr_buf=qb,paged_kv_indptr_buf=pb,paged_kv_indices_buf=ib,paged_kv_last_page_len_buf=lb,backend='fa2')
    out=torch.empty_like(q);lse=torch.empty((384,32),device='cuda',dtype=torch.float32)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    context=dict(sid=sid,q=q,kv=kv,ws=ws,w=w,out=out,lse=lse,stream=stream,order=order,qs=qs,
      ptrs=[t.data_ptr() for t in [q,kv,ws,out,lse,qb,pb,ib,lb,w._int_workspace_buffer]])
    contexts.append(context)
   def plan(c,Ls):
    counts=[(L+15)//16 for L in Ls]
    ids=torch.cat([c['order'][:counts[0]],c['order'][0 if shared else 256:(0 if shared else 256)+counts[1]]])
    qi=torch.tensor([0,256,384],device='cuda',dtype=torch.int32)
    pp=torch.tensor([0,counts[0],sum(counts)],device='cuda',dtype=torch.int32)
    last=torch.tensor([(L-1)%16+1 for L in Ls],device='cuda',dtype=torch.int32)
    c['w'].plan(qi,pp,ids,last,32,8,128,16,causal=True,q_data_type=getattr(torch,dtype_name),kv_data_type=getattr(torch,dtype_name),disable_split_kv=True)
    c['Ls']=Ls;c['indices']=ids
   def call(c):return c['w'].run(c['q'],c['kv'],out=c['out'],lse=c['lse'],return_lse=True)
   for c in contexts:
    with torch.cuda.stream(c['stream']):
     plan(c,[2048,4096])
     for _ in range(4):call(c)
    c['stream'].synchronize();gr=torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr,stream=c['stream']):call(c)
    c['graph']=gr
   for transition,Ls in enumerate(([2048,4096],[4096,2048],[1025,3073],[2048,4096])):
    for c in contexts:
     with torch.cuda.stream(c['stream']):
      plan(c,Ls);c['out'].fill_(float('nan'));c['lse'].fill_(float('nan'));c['graph'].replay()
    for c in contexts:c['stream'].synchronize()
    for c in contexts:
     saved=c['out'].clone();slse=c['lse'].clone()
     with torch.cuda.stream(c['stream']):call(c)
     c['stream'].synchronize();torch.testing.assert_close(c['out'],saved,atol=0,rtol=0);torch.testing.assert_close(c['lse'],slse,atol=0,rtol=0)
     ks=[];vs=[];start=0
     for L in Ls:
      n=(L+15)//16;ids=c['indices'][start:start+n].long();ks.append(c['kv'][ids,0].reshape(-1,8,128)[:L]);vs.append(c['kv'][ids,1].reshape(-1,8,128)[:L]);start+=n
     ref=oracle(c['q'],ks,vs,c['qs'],saved,slse,dtype_name)
     key=f'{dtype_name}-shared{shared}-stream{c["sid"]}-transition{transition}'
     path=a.refs/(key+'.pt');inputs=thash(c['q'])+thash(c['kv'])
     if a.mode=='pristine':
      if path.exists():raise FileExistsError('duplicate pristine reference')
      torch.save(dict(out=saved.cpu(),lse=slse.cpu(),inputs=inputs),path)
     expected=torch.load(path,map_location='cpu',weights_only=True)
     if expected['inputs']!=inputs:raise RuntimeError('changed inputs')
     exact=torch.equal(saved.cpu(),expected['out']) and torch.equal(slse.cpu(),expected['lse'])
     if a.mode in ['cap','off'] and not exact:raise RuntimeError('nonexact primary lifecycle')
     torch.testing.assert_close(saved.cpu(),expected['out'],atol=.005 if dtype_name=='float16' else .02,rtol=.02 if dtype_name=='float16' else .04)
     torch.testing.assert_close(slse.cpu(),expected['lse'],atol=.01,rtol=.01)
     records.append(dict(key=key,graph_eager_exact=True,pristine_exact=bool(exact),FP32=ref,
       full_max_abs=float((saved.cpu().float()-expected['out'].float()).abs().max())))
   for c in contexts:del c['graph']
   del contexts,c,ks,vs,saved,slse,expected;torch.cuda.empty_cache()
  if len(records)!=32:raise RuntimeError('incomplete lifecycle')
  save(a.out/'qualification.json',records);save(a.out/'complete.json',dict(complete=True,mode=a.mode,checks=len(records),header_sha256=h,source_sha256=sha(__file__),files={'qualification.json':sha(a.out/'qualification.json')},performance_claim=False))
  print('LIFECYCLE_PASS',a.mode,len(records),flush=True)
 except BaseException as exc:
  save(a.out/'partial.json',records);save(a.out/'failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--refs',type=Path,required=True);p.add_argument('--mode',choices=['pristine','off','cap','wide'],required=True);run(p.parse_args())
