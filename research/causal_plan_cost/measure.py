"""Real GPU experiment. No synthetic timings; FA2 operator results only."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
import traceback
from geometry import POLICIES, corpus, corpus_json, plan


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--replicate',type=int,default=0)
    p.add_argument('--stage',choices=['canary','full'],default='full')
    p.add_argument('--blocks',type=int,default=6)
    p.add_argument('--inner',type=int,default=3)
    a=p.parse_args()
    if min(a.blocks,a.inner)<=0:raise ValueError('positive measurement counts required')
    a.out.mkdir(parents=True,exist_ok=False)
    import torch
    import flashinfer
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    seed=20260928+a.replicate
    torch.manual_seed(seed)
    hw=torch.cuda.get_device_properties(0);sms=hw.multi_processor_count
    all_shapes=corpus()
    if a.stage=='canary':
        all_shapes=[s for s in all_shapes if s.name in
            ('discovery-eq255-A','discovery-eq255-B','Q512-K1536-eq63-A','Q512-K1536-n8-A')]
    suite=[(s,hq,hkv) for s in all_shapes for hq,hkv in ((16,4),(32,8))]
    random.Random(seed).shuffle(suite)
    metadata=dict(kind='causal_plan_operator_study',stage=a.stage,seed=seed,
        replicate=a.replicate,gpu=hw.name,sms=sms,sm=[hw.major,hw.minor],
        torch=torch.__version__,cuda=torch.version.cuda,flashinfer=flashinfer.__version__,
        backend='fa2',dtype='float16',head_dim=128,causal=True,cuda_graph=False,
        blocks=a.blocks,inner=a.inner,planned_shape_head_pairs=len(suite),
        planned_records=len(suite)*len(POLICIES),policies=POLICIES,
        corpus_sha256=hashlib.sha256(corpus_json().encode()).hexdigest(),
        source_sha256={n:sha(Path(__file__).with_name(n)) for n in ('geometry.py','measure.py')},
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        numerical_equivalence='tolerance-based, not bitwise',end_to_end_serving=False)
    (a.out/'manifest.json').write_text(json.dumps(metadata,indent=2))
    records=[];start_all=time.perf_counter();failures=[]
    for shape,hq,hkv in suite:
        qoff=[0];koff=[0]
        for ql,kl in zip(shape.q,shape.k):
            qoff.append(qoff[-1]+ql);koff.append(koff[-1]+ql+kl)
        q=torch.randn((qoff[-1],hq,128),device='cuda',dtype=torch.float16)
        k=torch.randn((koff[-1],hkv,128),device='cuda',dtype=torch.float16)
        v=torch.randn_like(k);out=torch.empty_like(q)
        qptr=torch.tensor(qoff,device='cuda',dtype=torch.int32)
        kptr=torch.tensor(koff,device='cuda',dtype=torch.int32)
        wrappers={};permode={};auto_output=None
        # Auto first only for correctness; all timing orders are randomized.
        for policy in POLICIES:
            rec=dict(shape=shape.record(),hq=hq,hkv=hkv,sms=sms,policy=policy,
                     gpu=hw.name,replicate=a.replicate,status='starting')
            try:
                workspace=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
                w=flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace,kv_layout='NHD',backend='fa2')
                opts=dict(causal=True,q_data_type=torch.float16,kv_data_type=torch.float16)
                if policy=='none':opts['disable_split_kv']=True
                elif policy!='auto':opts['fixed_split_size']=int(policy[1:])
                def do_plan():w.plan(qptr,kptr,hq,hkv,128,**opts)
                do_plan();torch.cuda.synchronize()
                pred=plan(shape,hq,hkv,sms,policy)
                info=list(w._plan_info)
                if len(info)!=15:raise RuntimeError('unsupported planner ABI')
                assert int(info[0])==pred['grid_x'],(info,pred)
                assert int(info[3])==pred['tile'],(info,pred)
                assert bool(info[14])==pred['split'],(info,pred)
                observed_chunk=None
                if pred['split']:
                    off=int(info[9])
                    observed_chunk=int(w._int_workspace_buffer[off:off+4].cpu().view(torch.int32).item())
                    assert observed_chunk==pred['kv_chunk'],(observed_chunk,pred)
                rec['planner_match']=True
                rec['plan_summary']={z:pred[z] for z in ('grid_x','tile','kv_chunk','split','active','merge_rows')}
                rec['observed_kv_chunk']=observed_chunk
                got=w.run(q,k,v,out=out).clone();torch.cuda.synchronize()
                if auto_output is None:
                    auto_output=got.clone();max_ref=0.;nref=0
                    for ri,(ql,depth) in enumerate(zip(shape.q,shape.k)):
                        for qi in sorted(set((0,ql-1))):
                            for h in (0,hq-1):
                                end=koff[ri]+depth+qi+1;kh=h//(hq//hkv)
                                qr=q[qoff[ri]+qi,h].float()
                                kr=k[koff[ri]:end,kh].float();vr=v[koff[ri]:end,kh].float()
                                ref=torch.softmax((kr@qr)/math.sqrt(128),dim=0)@vr
                                actual=got[qoff[ri]+qi,h].float()
                                torch.testing.assert_close(actual,ref,atol=.005,rtol=.02)
                                max_ref=max(max_ref,float((actual-ref).abs().max()));nref+=1
                    rec['fp32_reference']={'vectors':nref,'max_abs':max_ref,'passed':True}
                torch.testing.assert_close(got,auto_output,atol=.005,rtol=.02)
                rec['full_output_comparison']={'passed':True,'elements':got.numel(),
                    'max_abs':float((got-auto_output).abs().max())}
                for _ in range(5):w.run(q,k,v,out=out)
                torch.cuda.synchronize()
                plan_ms=[]
                for _ in range(3):
                    tick=time.perf_counter();do_plan();torch.cuda.synchronize()
                    plan_ms.append((time.perf_counter()-tick)*1000)
                rec['plan_wall_ms']=plan_ms
                rec['cuda_ms']=[];rec['status']='ready';wrappers[policy]=w
            except Exception as exc:
                rec.update(status='error',error=type(exc).__name__+': '+str(exc)[:1500])
                failures.append(dict(name=shape.name,hq=hq,policy=policy,error=rec['error']))
            permode[policy]=rec
        for block in range(a.blocks):
            order=list(POLICIES);random.Random(f'{seed}:{shape.name}:{hq}:{block}').shuffle(order)
            # Forward/reverse pair cancels linear drift within the block.
            values={m:[] for m in wrappers}
            for mode in order+order[::-1]:
                if mode not in wrappers:continue
                w=wrappers[mode];beg=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                beg.record()
                for _ in range(a.inner):w.run(q,k,v,out=out)
                end.record();end.synchronize()
                values[mode].append(beg.elapsed_time(end)/a.inner)
            for mode,xs in values.items():permode[mode]['cuda_ms'].append(statistics.mean(xs))
        for mode,rec in permode.items():
            if rec['status']=='ready':
                rec.update(status='complete',median_ms=statistics.median(rec['cuda_ms']),
                           mean_plan_ms=statistics.mean(rec['plan_wall_ms']))
            records.append(rec)
            with (a.out/'measurements.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        print('CASE',shape.name,hq,{m:round(r.get('median_ms',-1),5) for m,r in permode.items()},flush=True)
        del wrappers,permode,w,workspace,q,k,v,out,got,auto_output
        torch.cuda.empty_cache()
    summary=dict(metadata,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        elapsed_seconds=time.perf_counter()-start_all,records=len(records),
        completed_records=sum(r['status']=='complete' for r in records),failures=failures,
        measurements_sha256=sha(a.out/'measurements.jsonl'))
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('SUMMARY',json.dumps(summary),flush=True)
    if failures:raise SystemExit(2)


if __name__=='__main__':main()
