"""Physical-buffer-fixed, metadata-only FA2 schedule intervention.

Candidate: heavy_first, declared before timing. Other policies are explanatory.
Real FlashInfer calls only. No full-model, HTTP, or serving-gain claim.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from plan_contract import PlanSnapshot, POLICIES, canonical_hash
from frozen_geometry import CONFIGS

def write(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def cases(stage):
    result=[]
    names=['n2','n4'] if stage=='canary' else list(CONFIGS)
    for name in names:
        k,q,eq=CONFIGS[name]
        states=[('same',q),('opposite',q[::-1])]
        if eq: states += [('eqC-a',eq[0]),('eqC-b',eq[1])]
        for state,pairing in states:
            result.append(dict(case='discovery-'+name,state=state,query=list(pairing),
                               cached=list(k),holdout=False))
    if stage=='test':
        rng=random.Random(20261103)
        for j,n in enumerate([3,3,5,5,7,7,9,9]):
            q=sorted(rng.sample(range(48,769,16),n))
            k=sorted(rng.sample(range(0,24577,512),n))
            for label,qs in [('same',q),('opposite',q[::-1])]:
                result.append(dict(case='holdout-%02d'%j,state=label,query=qs,cached=k,holdout=True))
    return result

def oracle(q,k,v,out,lse,qs,ls):
    import torch
    err=0.;lerr=0.;count=0;qo=ko=0
    for nq,nk in zip(qs,ls):
        ix=sorted({0,nq//2,nq-1});idx=torch.tensor(ix,device=q.device)
        qq=q[qo+idx].float().transpose(0,1)
        kk=k[ko:ko+nk].float().repeat_interleave(4,dim=1).transpose(0,1)
        vv=v[ko:ko+nk].float().repeat_interleave(4,dim=1).transpose(0,1)
        scores=qq@kk.transpose(-1,-2)/math.sqrt(128)
        scores.masked_fill_((torch.arange(nk,device=q.device)[None,:]>(idx+nk-nq)[:,None])[None],float('-inf'))
        ref=(scores.softmax(-1)@vv).transpose(0,1)
        actual=out[qo+idx].float()
        torch.testing.assert_close(actual,ref,atol=.005 if q.dtype==torch.float16 else .02,rtol=.02 if q.dtype==torch.float16 else .04)
        # FlashInfer prefill LSE uses base-2 logs in this pinned kernel.
        expected_lse=scores.logsumexp(-1).transpose(0,1)/math.log(2)
        torch.testing.assert_close(lse[qo+idx],expected_lse,atol=.01,rtol=.01)
        err=max(err,float((actual-ref).abs().max()));lerr=max(lerr,float((lse[qo+idx]-expected_lse).abs().max()))
        count+=len(ix)*32;qo+=nq;ko+=nk
    return dict(selected_fp32_vectors=count,max_abs=err,lse_max_abs=lerr)

def run(a):
    import torch
    import flashinfer
    torch.set_num_threads(1);torch.manual_seed(20261119+a.rep)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    a.out.mkdir(parents=True,exist_ok=False)
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    cs=cases(a.stage);dtypes=['float16'] if a.stage=='canary' else ['float16','bfloat16']
    blocks=2 if a.stage=='canary' else 6;inner=3 if a.stage=='canary' else 6
    properties=torch.cuda.get_device_properties(0)
    env=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),job=os.getenv('SLURM_JOB_ID'),rep=a.rep,
             gpu=properties.name,sm=[properties.major,properties.minor],sm_count=properties.multi_processor_count,
             torch=torch.__version__,flashinfer=flashinfer.__version__,cuda=torch.version.cuda,
             stage=a.stage,source=source,case_manifest=cs,case_hash=canonical_hash(cs),dtype=dtypes,
             blocks=blocks,inner=inner,primary_policy='heavy_first',scratch_MiB=512,
             plan_creation='eager FA2; fixed-shape CUDA graph replay, not dynamic graph plans',
             full_model=False,HTTP=False,timing='run only; all Python validation/setup outside timing, setup cost separately recorded')
    for key,cmd in [('hardware',['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,power.limit','--format=csv']),
                    ('occupancy',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv'])]:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=10);env[key]=dict(rc=p.returncode,out=p.stdout,err=p.stderr)
    write(a.out/'environment.json',env)
    rows=[];checks=[];plans=[];start=time.monotonic();rng=random.Random(12081+a.rep)
    try:
        for dtype_name,case,split in itertools.product(dtypes,cs,['auto','unsplit']):
            dtype=getattr(torch,dtype_name);qs=case['query'];ls=[q+k for q,k in zip(qs,case['cached'])]
            q=torch.randn((sum(qs),32,128),device='cuda',dtype=dtype)
            k=torch.randn((sum(ls),8,128),device='cuda',dtype=dtype);v=torch.randn_like(k)
            qp=torch.tensor([0]+list(itertools.accumulate(qs)),device='cuda',dtype=torch.int32)
            kp=torch.tensor([0]+list(itertools.accumulate(ls)),device='cuda',dtype=torch.int32)
            space=torch.empty(512*1024**2,device='cuda',dtype=torch.uint8)
            w=flashinfer.BatchPrefillWithRaggedKVCacheWrapper(space,backend='fa2')
            torch.cuda.synchronize();tick=time.perf_counter()
            w.plan(qp,kp,32,8,128,causal=True,q_data_type=dtype,kv_data_type=dtype,disable_split_kv=split=='unsplit')
            torch.cuda.synchronize();plan_us=(time.perf_counter()-tick)*1e6
            snapshot=PlanSnapshot(w,qs,ls)
            out=torch.empty_like(q);lse=torch.empty((sum(qs),32),device='cuda',dtype=torch.float32)
            def call(): return w.run(q,k,v,out=out,lse=lse,return_lse=True)
            for _ in range(3):call()
            torch.cuda.synchronize();ref=out.clone();ref_lse=lse.clone()
            refcheck=oracle(q,k,v,ref,ref_lse,qs,ls)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(4):call()
            graph.replay();torch.cuda.synchronize()
            torch.testing.assert_close(out,ref,atol=0,rtol=0);torch.testing.assert_close(lse,ref_lse,atol=0,rtol=0)
            ptrs=[q.data_ptr(),k.data_ptr(),v.data_ptr(),out.data_ptr(),space.data_ptr()]
            key=dict(case=case['case'],state=case['state'],holdout=case['holdout'],dtype=dtype_name,split=split)
            plans.append(dict(key,plan=snapshot.report,buffer_ids=ptrs,plan_us=plan_us,reference=refcheck))
            for policy in POLICIES:
                signature=snapshot.apply(policy)
                out.fill_(float('nan'));lse.fill_(float('nan'));call();torch.cuda.synchronize()
                torch.testing.assert_close(out,ref,atol=0,rtol=0);torch.testing.assert_close(lse,ref_lse,atol=0,rtol=0)
                out.fill_(float('nan'));lse.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
                torch.testing.assert_close(out,ref,atol=0,rtol=0);torch.testing.assert_close(lse,ref_lse,atol=0,rtol=0)
                checks.append(dict(key,policy=policy,descriptor_hash=signature,full_output_exact=True,lse_exact=True,graph_exact=True))
            for block in range(-1,blocks):
                order=list(POLICIES);rng.shuffle(order)
                for policy in order:
                    tick=time.perf_counter();signature=snapshot.apply(policy);apply_us=(time.perf_counter()-tick)*1e6
                    if [q.data_ptr(),k.data_ptr(),v.data_ptr(),out.data_ptr(),space.data_ptr()]!=ptrs:
                        raise ValueError('physical buffer moved')
                    graph.replay();torch.cuda.synchronize()
                    torch.testing.assert_close(out,ref,atol=0,rtol=0);torch.testing.assert_close(lse,ref_lse,atol=0,rtol=0)
                    modes=['eager','graph'];rng.shuffle(modes)
                    for mode in modes:
                        e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize();tick=time.perf_counter();e0.record()
                        for _ in range(inner):
                            if mode=='eager':call()
                            else:graph.replay()
                        e1.record();e1.synchronize();count=inner*(4 if mode=='graph' else 1)
                        wall_us=(time.perf_counter()-tick)*1e6/count;device_us=e0.elapsed_time(e1)*1000/count
                        if not (math.isfinite(device_us) and device_us>0 and math.isfinite(wall_us) and wall_us>0):raise ValueError('invalid timing')
                        if block>=0:
                            row=dict(key,rep=a.rep,block=block,policy=policy,mode=mode,device_us=device_us,wall_us=wall_us,
                                     apply_us=apply_us,plan_us=plan_us,calls=count,descriptor_hash=signature,
                                     trial_id='/'.join(map(str,[a.rep,dtype_name,case['case'],case['state'],split,block,policy,mode])))
                            rows.append(row)
            snapshot.apply('identity');call();torch.cuda.synchronize();torch.testing.assert_close(out,ref,atol=0,rtol=0)
            write(a.out/'progress.json',dict(key,rows=len(rows),checks=len(checks),seconds=time.monotonic()-start))
            print('CELL',dtype_name,case['case'],case['state'],split,len(rows),flush=True)
            del graph,snapshot,w,space,q,k,v,qp,kp,ref,ref_lse,out,lse
            torch.cuda.empty_cache()
        expected=len(cs)*len(dtypes)*2*len(POLICIES)*2*blocks
        if len(rows)!=expected or len({r['trial_id'] for r in rows})!=expected:raise ValueError('incomplete trial matrix')
        write(a.out/'measurements.json',rows);write(a.out/'qualification.json',checks);write(a.out/'plans.json',plans)
        write(a.out/'complete.json',dict(complete=True,stage=a.stage,rep=a.rep,rows=len(rows),expected=expected,
              qualifications=len(checks),source=source,case_hash=env['case_hash'],full_model=False,
              serving_promotion=False,seconds=time.monotonic()-start,
              files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.glob('*.json')}))
        print('COMPLETE',a.stage,a.rep,len(rows),flush=True)
    except BaseException as exc:
        write(a.out/'partial_measurements.json',rows);write(a.out/'plans.json',plans);write(a.out/'qualification.json',checks)
        write(a.out/'failure.json',dict(type=type(exc).__name__,message=str(exc),rows=len(rows),complete=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=['canary','test'],required=True);p.add_argument('--rep',type=int,default=0)
    run(p.parse_args())
