"""Real GPU numerical robustness; no performance measurements or model-quality claim."""
from __future__ import annotations
import argparse,hashlib,itertools,json,math,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from measure import save,sha,thash
from prepare import EXPECTED

def cases():
 result=[]
 for heads,(name,qs,ls),pattern,causal in itertools.product([(8,8),(16,4),(32,4),(32,1)],
    [('full-small',[16,32],[48,96]),('long-kv',[127,33],[32768,16384]),('tile-tail',[513,63],[4096,8192])],
    ['zero','unit','peaked'],[True,False]):
  result.append(dict(id=len(result),hq=heads[0],hkv=heads[1],name=name,q=qs,kv=ls,pattern=pattern,causal=causal))
 return result

def fp32(q,k,v,qs,ls,causal,out,lse):
 import torch
 group=q.shape[1]//k.shape[1];qo=ko=0;maxabs=0.;rmse=0.;nel=0;lmax=0.;vectors=0
 for nq,nk in zip(qs,ls):
  ids=list(range(nq)) if sum(qs)<=64 else sorted({0,1,nq//4,nq//2,3*nq//4,nq-2,nq-1})
  ix=torch.tensor(ids,device=q.device);qq=q[qo+ix].float().transpose(0,1)
  kk=k[ko:ko+nk].float().repeat_interleave(group,dim=1).transpose(0,1)
  vv=v[ko:ko+nk].float().repeat_interleave(group,dim=1).transpose(0,1)
  scores=(qq@kk.transpose(-1,-2))/math.sqrt(128)
  if causal:scores.masked_fill_((torch.arange(nk,device=q.device)[None,:]>(ix+nk-nq)[:,None])[None],float('-inf'))
  ref=(scores.softmax(-1)@vv).transpose(0,1);lr=scores.logsumexp(-1).transpose(0,1)/math.log(2)
  difference=out[qo+ix].float()-ref;maximum=float(difference.abs().max())
  torch.testing.assert_close(out[qo+ix].float(),ref,atol=.005 if q.dtype==torch.float16 else .02,rtol=.02 if q.dtype==torch.float16 else .04)
  torch.testing.assert_close(lse[qo+ix],lr,atol=.01,rtol=.01)
  maxabs=max(maxabs,maximum);rmse+=float(difference.square().sum());nel+=difference.numel();lmax=max(lmax,float((lse[qo+ix]-lr).abs().max()));vectors+=len(ids)*q.shape[1];qo+=nq;ko+=nk
 return dict(max_abs=maxabs,rmse=math.sqrt(rmse/nel),lse_max_abs=lmax,evaluated_elements=nel,evaluated_vectors=vectors,full_reference=sum(qs)<=64)

def run(a):
 import torch,flashinfer
 if a.out.exists():raise FileExistsError('preserve robustness results')
 if not torch.cuda.is_available() or flashinfer.__version__!='0.7.0':raise RuntimeError('wrong real CUDA environment')
 a.out.mkdir(parents=True);a.refs.mkdir(exist_ok=True,parents=True);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 pkg=Path(flashinfer.__file__).parent;h=sha(pkg/'data/include/flashinfer/attention/prefill.cuh')
 if a.mode=='pristine':
  if h!=EXPECTED:raise RuntimeError('pristine mismatch')
 else:
  binding=json.loads((pkg.parent/'RESOURCE_BINDING.json').read_text())
  if binding['mode']!=a.mode or h!=binding['modified_sha256']:raise RuntimeError('source mismatch')
 rows=[]
 save(a.out/'environment.json',dict(mode=a.mode,gpu=torch.cuda.get_device_name(),header_sha256=h,source_sha256=sha(__file__),torch=torch.__version__,flashinfer=flashinfer.__version__,cases=cases(),performance_measured=False))
 try:
  for c,dtype_name in itertools.product(cases(),['float16','bfloat16']):
   torch.manual_seed(941001+c['id']);dtype=getattr(torch,dtype_name);qs=c['q'];ls=c['kv'];hq=c['hq'];hkv=c['hkv']
   q=torch.randn((sum(qs),hq,128),device='cuda',dtype=dtype);k=torch.randn((sum(ls),hkv,128),device='cuda',dtype=dtype);v=torch.randn_like(k)
   if c['pattern']=='zero':q.zero_();k.zero_()
   if c['pattern']=='peaked':q.mul_(4);k.mul_(4)
   qi=torch.tensor([0]+list(itertools.accumulate(qs)),device='cuda',dtype=torch.int32);ki=torch.tensor([0]+list(itertools.accumulate(ls)),device='cuda',dtype=torch.int32)
   ws=torch.empty(512*1024**2,device='cuda',dtype=torch.uint8);w=flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws,backend='fa2')
   w.plan(qi,ki,hq,hkv,128,causal=c['causal'],q_data_type=dtype,kv_data_type=dtype,disable_split_kv=True)
   out=torch.full_like(q,float('nan'));lse=torch.full((sum(qs),hq),float('nan'),device='cuda',dtype=torch.float32)
   w.run(q,k,v,out=out,lse=lse,return_lse=True);torch.cuda.synchronize()
   reference=fp32(q,k,v,qs,ls,c['causal'],out,lse);inputs=thash(q)+thash(k)+thash(v)
   file=a.refs/f'{c["id"]}-{dtype_name}.pt'
   if a.mode=='pristine':
    if file.exists():raise FileExistsError('duplicate pristine record')
    torch.save(dict(out=out.cpu(),lse=lse.cpu(),inputs=inputs),file)
   base=torch.load(file,weights_only=True,map_location='cpu')
   if base['inputs']!=inputs:raise RuntimeError('data mismatch')
   exact=torch.equal(base['out'],out.cpu()) and torch.equal(base['lse'],lse.cpu())
   if a.mode in ['off','cap'] and not exact:raise RuntimeError('resource-only numerical change')
   torch.testing.assert_close(out.cpu(),base['out'],atol=.005 if dtype_name=='float16' else .02,rtol=.02 if dtype_name=='float16' else .04)
   torch.testing.assert_close(lse.cpu(),base['lse'],atol=.01,rtol=.01)
   rows.append(dict(case=c,dtype=dtype_name,pristine_exact=bool(exact),full_max_abs=float((out.cpu().float()-base['out'].float()).abs().max()),reference=reference,output_sha256=thash(out),lse_sha256=thash(lse)))
   if len(rows)%12==0:print('NUMERICS',a.mode,len(rows),flush=True)
   del q,k,v,ws,w,out,lse,base;torch.cuda.empty_cache()
  if len(rows)!=144:raise RuntimeError('incomplete robustness matrix')
  save(a.out/'qualification.json',rows);save(a.out/'complete.json',dict(complete=True,mode=a.mode,checks=len(rows),file_sha256=sha(a.out/'qualification.json'),performance_claim=False))
 except BaseException as exc:
  save(a.out/'partial.json',rows);save(a.out/'failure.json',dict(type=type(exc).__name__,message=str(exc),passed_checks=len(rows)));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--refs',type=Path,required=True);p.add_argument('--mode',choices=['pristine','off','cap','wide'],required=True);run(p.parse_args())
