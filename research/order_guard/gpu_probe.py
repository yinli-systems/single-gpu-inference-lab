"""Real-GPU-only fixed-buffer probe. Does not submit jobs or change live runtimes.

Run-only diagnostic: Python setup is measured separately. No serving promotion.
The flush label describes a 128 MiB write, not a proof of complete L2 eviction.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys
import subprocess
import time

from order_guard import Geometry, digest, propose

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'benchmarks/results/plan-order-mechanism'
MANIFEST='fb6c14b78cc03f8d02b7b16ff8efa9f382ce313b13e46781f5506a78be7cfa37'


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def apply_order(snapshot, order, torch):
    # Restore through the original audited generation and invariant checks first.
    snapshot.apply('identity')
    if sorted(order)!=list(range(len(snapshot.desc))):raise ValueError('invalid proposal')
    fields=('request_indices_offset','qo_tile_indices_offset','kv_tile_indices_offset')
    for column,field in enumerate(fields):
        values=torch.tensor([snapshot.desc[i][column] for i in order],device=snapshot.device,dtype=torch.int32)
        snapshot.buffer.view(torch.uint8).narrow(0,snapshot.info[field],4*len(order)).view(torch.int32).copy_(values)
    torch.cuda.synchronize()
    actual=list(zip(*(snapshot._read(snapshot.info[f],len(order)) for f in fields)))
    expected=[snapshot.desc[i] for i in order]
    if actual!=expected:raise ValueError('descriptor transfer mismatch')
    snapshot.last_descriptor_hash=digest(actual)
    return snapshot.last_descriptor_hash


def run(a):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('real CUDA GPU unavailable; no mock fallback')
    import flashinfer
    if flashinfer.__version__ != '0.6.18':
        raise RuntimeError('this adapter is qualified only for FlashInfer 0.6.18; port separately')
    sys.path.insert(0,str(BASE))
    from plan_contract import PlanSnapshot, order_indices
    from measure_order import oracle
    from manifest import build
    manifest=build()
    if digest(manifest['cases'])!=MANIFEST or manifest['cases_sha256']!=MANIFEST:raise ValueError('manifest changed')
    if a.out.exists():raise FileExistsError('refuse overwriting evidence')
    if not 0<=a.shard<a.shards<=8 or not 0<=a.rep<3:raise ValueError('invalid bounded shard/repeat')
    selected=[r for i,r in enumerate(manifest['cases']) if i%a.shards==a.shard]
    if a.stage=='canary':selected=[manifest['cases'][0],next(r for r in manifest['cases'] if r['kind']=='old-regression')]
    dtypes=['float16'] if a.stage=='canary' else manifest['dtypes']
    caches=['warm'] if a.stage=='canary' else manifest['cache_regimes']
    policies=manifest['policies'];blocks=6
    torch.set_num_threads(1);torch.manual_seed(20260929134631+a.rep)
    torch.backends.cuda.matmul.allow_tf32=False
    rng=random.Random(293421+a.rep)
    a.out.mkdir(parents=True)
    env=dict(torch=torch.__version__,cuda=torch.version.cuda,flashinfer=flashinfer.__version__,
             gpu=torch.cuda.get_device_name(),manifest_sha256=MANIFEST,cases=selected,rep=a.rep,stage=a.stage,
             source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
             setup_excluded_from_run=True,full_model=False,serving_promotion=False,
             kernel_symbol_verified=False,timing='single attention invocation, wall and CUDA event reported separately')
    properties=torch.cuda.get_device_properties(0)
    env.update(sm=[properties.major,properties.minor],sm_count=properties.multi_processor_count,
               clocks_locked=False,exclusive_allocation_verified=False,shard=a.shard,shards=a.shards)
    for label,command in [('hardware',['nvidia-smi','--query-gpu=name,uuid,driver_version,power.limit','--format=csv']),
                          ('processes',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv'])]:
        result=subprocess.run(command,capture_output=True,text=True,timeout=10)
        env[label]=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        if result.returncode:raise RuntimeError('hardware evidence unavailable')
    save(a.out/'environment.json',env);rows=[];quals=[];plans=[]
    try:
        for dtype_name,case,split in itertools.product(dtypes,selected,manifest['split_modes']):
            dtype=getattr(torch,dtype_name);qs=case['query'];ls=[q+k for q,k in zip(qs,case['cached'])]
            q=torch.randn((sum(qs),32,128),device='cuda',dtype=dtype)
            k=torch.randn((sum(ls),8,128),device='cuda',dtype=dtype);v=torch.randn_like(k)
            qp=torch.tensor([0]+list(itertools.accumulate(qs)),device='cuda',dtype=torch.int32)
            kp=torch.tensor([0]+list(itertools.accumulate(ls)),device='cuda',dtype=torch.int32)
            space=torch.empty(512*1024**2,device='cuda',dtype=torch.uint8)
            w=flashinfer.BatchPrefillWithRaggedKVCacheWrapper(space,backend='fa2')
            w.plan(qp,kp,32,8,128,causal=True,q_data_type=dtype,kv_data_type=dtype,disable_split_kv=split=='unsplit')
            snapshot=PlanSnapshot(w,qs,ls)
            g=Geometry(tuple(qs),tuple(ls),4,snapshot.info['cta_tile_q'],snapshot.chunk,bool(snapshot.info['split_kv']))
            out=torch.empty_like(q);lse=torch.empty((sum(qs),32),device='cuda',dtype=torch.float32)
            def call():return w.run(q,k,v,out=out,lse=lse,return_lse=True)
            for _ in range(3):call()
            torch.cuda.synchronize();ref=out.clone();rlse=lse.clone();reference=oracle(q,k,v,ref,rlse,qs,ls)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):call()
            flush=torch.empty(128*1024**2,device='cuda',dtype=torch.uint8)
            key=dict(case=case['id'],kind=case['kind'],dtype=dtype_name,split=split,rep=a.rep)
            plans.append(dict(key,plan=snapshot.report,FP32_reference=reference))
            ptrs=[x.data_ptr() for x in (q,k,v,out,lse,space)]
            orders={}
            for policy in policies:
                orders[policy]=propose(g,snapshot.desc,policy) if policy in ('causal_heavy','locality_packet8') else \
                    order_indices(snapshot.desc,qs,ls,g.tile,g.chunk,g.split,g.group,policy)
                signature=apply_order(snapshot,orders[policy],torch)
                for mode in ('eager','graph'):
                    out.fill_(float('nan'));lse.fill_(float('nan'))
                    call() if mode=='eager' else graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(out,ref,atol=0,rtol=0);torch.testing.assert_close(lse,rlse,atol=0,rtol=0)
                quals.append(dict(key,policy=policy,descriptor_sha256=signature,exact_output=True,exact_lse=True,graph_exact=True))
            for block in range(-1,blocks):
                order=list(policies);rng.shuffle(order)
                for policy in order:
                    start=time.perf_counter();signature=apply_order(snapshot,orders[policy],torch)
                    setup_us=(time.perf_counter()-start)*1e6
                    if ptrs!=[x.data_ptr() for x in (q,k,v,out,lse,space)]:raise ValueError('buffer identity changed')
                    conditions=list(itertools.product(('eager','graph'),caches));rng.shuffle(conditions)
                    for mode,cache in conditions:
                        graph.replay();torch.cuda.synchronize()
                        if cache=='flush128MiB':flush.zero_();torch.cuda.synchronize()
                        e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start=time.perf_counter();e0.record()
                        call() if mode=='eager' else graph.replay()
                        e1.record();e1.synchronize();wall_us=(time.perf_counter()-start)*1e6;device_us=e0.elapsed_time(e1)*1000
                        if not all(math.isfinite(x) and x>0 for x in (wall_us,device_us,setup_us)):raise ValueError('invalid timing')
                        torch.testing.assert_close(out,ref,atol=0,rtol=0);torch.testing.assert_close(lse,rlse,atol=0,rtol=0)
                        if block>=0:rows.append(dict(key,block=block,policy=policy,mode=mode,cache=cache,
                            setup_us=setup_us,wall_us=wall_us,device_us=device_us,calls=1,descriptor_sha256=signature))
            snapshot.apply('identity')
            save(a.out/'progress.json',dict(case=case['id'],rows=len(rows),qualifications=len(quals)))
            del graph,snapshot,w,space,q,k,v,qp,kp,out,lse,ref,rlse,flush
            torch.cuda.empty_cache()
        expected=len(selected)*len(dtypes)*2*len(policies)*2*len(caches)*blocks
        if len(rows)!=expected:raise ValueError('incomplete matrix')
        save(a.out/'measurements.json',rows);save(a.out/'qualification.json',quals);save(a.out/'plans.json',plans)
        save(a.out/'complete.json',dict(complete=True,rows=len(rows),expected_rows=expected,GPU_executed=True,
             source_manifest=MANIFEST,serving_promotion=False,
             files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.glob('*.json')}))
    except BaseException as exc:
        save(a.out/'partial_measurements.json',rows);save(a.out/'qualification.json',quals);save(a.out/'plans.json',plans)
        save(a.out/'failure.json',dict(type=type(exc).__name__,message=str(exc),complete=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=('canary','formal'),default='canary');p.add_argument('--rep',type=int,default=0)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);run(p.parse_args())
