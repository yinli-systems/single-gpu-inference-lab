"""Development-only GPU smoke for native/cap symbol isolation and exact graph replay."""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, itertools, json, time


def thash(t) -> str:
    import torch
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

def save(path: Path, value) -> None:
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def graph_time(graph, torch, replays=32) -> dict:
    graph.replay();torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    tick=time.perf_counter_ns();a.record()
    for _ in range(replays):graph.replay()
    b.record();b.synchronize()
    return {'wall_us_per_replay':(time.perf_counter_ns()-tick)/1000/replays,'device_us_per_replay':a.elapsed_time(b)*1000/replays,'replays':replays}

def make_wrapper(torch,flashinfer,layout,policy,q,qi,k,v,lengths):
    dtype=q.dtype;ws=torch.empty(512*1024**2,device='cuda',dtype=torch.uint8)
    if layout=='ragged':
        ki=torch.tensor([0]+list(itertools.accumulate(lengths)),device='cuda',dtype=torch.int32)
        wrapper=flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws,backend='fa2')
        plan=lambda:wrapper.plan(qi,ki,32,8,128,causal=True,q_data_type=dtype,kv_data_type=dtype,disable_split_kv=True)
        call=lambda out,lse:wrapper.run(q,k,v,out=out,lse=lse,return_lse=True);keep=(ki,)
    else:
        page=16;counts=[(n+page-1)//page for n in lengths];pp=torch.tensor([0]+list(itertools.accumulate(counts)),device='cuda',dtype=torch.int32)
        npages=sum(counts);pi=torch.arange(npages,device='cuda',dtype=torch.int32);last=torch.tensor([(n-1)%page+1 for n in lengths],device='cuda',dtype=torch.int32)
        kp=torch.zeros((npages,page,8,128),device='cuda',dtype=dtype);vp=torch.zeros_like(kp);off=0
        for n,pages,kr,vr in zip(lengths,counts,k.split(lengths),v.split(lengths)):
            kp[off:off+pages].view(-1,8,128)[:n].copy_(kr);vp[off:off+pages].view(-1,8,128)[:n].copy_(vr);off+=pages
        wrapper=flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws,'NHD',backend='fa2')
        plan=lambda:wrapper.plan(qi,pp,pi,last,32,8,128,page,causal=True,q_data_type=dtype,kv_data_type=dtype,disable_split_kv=True)
        call=lambda out,lse:wrapper.run(q,(kp,vp),out=out,lse=lse,return_lse=True);keep=(pp,pi,last,kp,vp)
    if policy is not None:wrapper._sgi_resource_policy=policy
    return wrapper,plan,call,ws,keep

def run_layout(torch,flashinfer,layout,mode,ref_dir):
    torch.manual_seed(44017);qlens=[5,41,109,177,277,345,477];cached=[28672,22016,14592,8960,4352,640,64]
    lengths=[q+c for q,c in zip(qlens,cached)];dtype=torch.bfloat16
    q=torch.randn((sum(qlens),32,128),device='cuda',dtype=dtype);k=torch.randn((sum(lengths),8,128),device='cuda',dtype=dtype);v=torch.randn_like(k)
    qi=torch.tensor([0]+list(itertools.accumulate(qlens)),device='cuda',dtype=torch.int32)
    sequence=[('pristine',None)] if mode=='pristine' else [('native_before',0),('cap',1),('native_after',0)]
    results=[];ref_path=ref_dir/f'{layout}.json';reference=None if not ref_path.exists() else json.loads(ref_path.read_text())
    for label,policy in sequence:
        wrapper,plan,call,ws,keep=make_wrapper(torch,flashinfer,layout,policy,q,qi,k,v,lengths)
        plan();info=[int(x) for x in wrapper._plan_info]
        expected=15 if policy is None else 16
        if len(info)!=expected:raise RuntimeError('plan vector size')
        if policy in (0,1) and info[-1]!=policy:raise RuntimeError('forced policy bit')
        out=torch.empty_like(q);lse=torch.empty((q.shape[0],32),device='cuda',dtype=torch.float32)
        for _ in range(3):call(out,lse)
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):call(out,lse)
        out.fill_(float('nan'));lse.fill_(float('nan'));g.replay();torch.cuda.synchronize()
        item={'layout':layout,'label':label,'policy':policy,'plan_info':info,'out_sha256':thash(out),'lse_sha256':thash(lse),'timing':graph_time(g,torch),'pointers':[x.data_ptr() for x in (q,k,v,out,lse,ws)],'gpu':torch.cuda.get_device_name(),'sm_count':torch.cuda.get_device_properties(0).multi_processor_count}
        if mode=='pristine':reference=item;save(ref_path,item)
        else:
            if reference is None:raise RuntimeError('missing pristine reference')
            if item['out_sha256']!=reference['out_sha256'] or item['lse_sha256']!=reference['lse_sha256']:raise RuntimeError(f'numerical mismatch {layout} {label}')
        results.append(item)
    if mode=='candidate':
        before,next_cap,after=results
        if before['plan_info'][:-1]!=next_cap['plan_info'][:-1] or before['plan_info'][:-1]!=after['plan_info'][:-1]:raise RuntimeError('plan core drift')
        if before['out_sha256']!=after['out_sha256'] or before['lse_sha256']!=after['lse_sha256']:raise RuntimeError('native after cap changed')
        results.append({'layout':layout,'label':'native_after_over_before','wall_ratio':before['timing']['wall_us_per_replay']/after['timing']['wall_us_per_replay'],'device_ratio':before['timing']['device_us_per_replay']/after['timing']['device_us_per_replay'],'diagnostic_only':True})
    return results

def main(args):
    import torch,flashinfer
    if not torch.cuda.is_available() or flashinfer.__version__!='0.7.0':raise RuntimeError('CUDA/FlashInfer')
    args.out.mkdir(parents=True,exist_ok=True);args.refs.mkdir(parents=True,exist_ok=True);results=[]
    for layout in ('ragged','paged'):results.extend(run_layout(torch,flashinfer,layout,args.mode,args.refs))
    save(args.out/'results.json',{'mode':args.mode,'results':results,'complete':True,'release_evidence':False,'default_promotion':False,'serving_promotion':False})
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pristine','candidate'],required=True);p.add_argument('--refs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
