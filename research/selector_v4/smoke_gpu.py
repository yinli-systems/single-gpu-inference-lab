"""Compile/run smoke for pristine and v4 isolated tactics; development evidence only."""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, itertools, json, os
from policy import candidate_pool


def thash(t) -> str:
    import torch
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def save(path: Path, value) -> None:
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def run_layout(torch,flashinfer,layout,policy,ref_dir):
    torch.manual_seed(44017)
    qlens=[5,41,109,177,277,345,477]
    cached=[28672,22016,14592,8960,4352,640,64]
    lengths=[q+c for q,c in zip(qlens,cached)]
    dtype=torch.bfloat16
    q=torch.randn((sum(qlens),32,128),device='cuda',dtype=dtype)
    k=torch.randn((sum(lengths),8,128),device='cuda',dtype=dtype);v=torch.randn_like(k)
    ks=list(k.split(lengths));vs=list(v.split(lengths));qi=torch.tensor([0]+list(itertools.accumulate(qlens)),device='cuda',dtype=torch.int32)
    ws=torch.empty(512*1024**2,device='cuda',dtype=torch.uint8)
    if layout=='ragged':
        ki=torch.tensor([0]+list(itertools.accumulate(lengths)),device='cuda',dtype=torch.int32)
        wrapper=flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws,backend='fa2')
        plan=lambda:wrapper.plan(qi,ki,32,8,128,causal=True,q_data_type=dtype,kv_data_type=dtype,disable_split_kv=True)
        call=lambda out,lse:wrapper.run(q,k,v,out=out,lse=lse,return_lse=True)
    else:
        page=16;counts=[(n+page-1)//page for n in lengths];pp=torch.tensor([0]+list(itertools.accumulate(counts)),device='cuda',dtype=torch.int32)
        npages=sum(counts);pi=torch.arange(npages,device='cuda',dtype=torch.int32);last=torch.tensor([(n-1)%page+1 for n in lengths],device='cuda',dtype=torch.int32)
        kp=torch.zeros((npages,page,8,128),device='cuda',dtype=dtype);vp=torch.zeros_like(kp);off=0
        for n,pages,kr,vr in zip(lengths,counts,ks,vs):
            kp[off:off+pages].view(-1,8,128)[:n].copy_(kr);vp[off:off+pages].view(-1,8,128)[:n].copy_(vr);off+=pages
        wrapper=flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws,'NHD',backend='fa2')
        plan=lambda:wrapper.plan(qi,pp,pi,last,32,8,128,page,causal=True,q_data_type=dtype,kv_data_type=dtype,disable_split_kv=True)
        call=lambda out,lse:wrapper.run(q,(kp,vp),out=out,lse=lse,return_lse=True)
    if policy is not None: wrapper._sgi_resource_policy=policy
    plan(); info=[int(x) for x in wrapper._plan_info]
    if policy in (0,1) and info[-1]!=policy:raise RuntimeError('forced policy bit')
    if policy==2:
        expected=candidate_pool(q=qlens,cached=cached,padded_batch_size=info[0],num_qo_heads=32,num_kv_heads=8,num_sms=torch.cuda.get_device_properties(0).multi_processor_count,split_kv=bool(info[14])).eligible
        if bool(info[-1])!=expected:raise RuntimeError('candidate-pool mismatch')
    out=torch.empty_like(q);lse=torch.empty((q.shape[0],32),device='cuda',dtype=torch.float32)
    call(out,lse);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph();
    with torch.cuda.graph(graph): call(out,lse)
    out.fill_(float('nan'));lse.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
    result={'layout':layout,'policy':policy,'plan_info':info,'out_sha256':thash(out),'lse_sha256':thash(lse),'gpu':torch.cuda.get_device_name(),'sm_count':torch.cuda.get_device_properties(0).multi_processor_count}
    key=f'{layout}.json'
    if policy is None:
        save(ref_dir/key,result)
    else:
        ref=json.loads((ref_dir/key).read_text())
        if result['out_sha256']!=ref['out_sha256'] or result['lse_sha256']!=ref['lse_sha256']:
            raise RuntimeError(f'numerical mismatch {layout} policy={policy}')
    return result


def main(args):
    import torch,flashinfer
    if not torch.cuda.is_available() or flashinfer.__version__!='0.7.0':raise RuntimeError('CUDA/FlashInfer')
    args.out.mkdir(parents=True,exist_ok=True);args.refs.mkdir(parents=True,exist_ok=True)
    results=[]
    policies=(None,) if args.mode=='pristine' else (0,1,2)
    for layout in ('ragged','paged'):
        for policy in policies:results.append(run_layout(torch,flashinfer,layout,policy,args.refs))
    save(args.out/'results.json',{'mode':args.mode,'results':results,'complete':True,'release_evidence':False,'default_promotion':False})
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pristine','v4'],required=True);p.add_argument('--refs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
