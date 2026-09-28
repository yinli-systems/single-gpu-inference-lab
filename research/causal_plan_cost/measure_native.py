"""Fresh live cycle validation of frozen policies. NOT full-model serving.

Primary: native_cycle vs the training-selected hardware/head fixed policy.
Secondary: native_run, stock auto. Do not choose the better endpoint after test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from geometry import corpus, POLICIES, plan, workspace_bytes
from native import Native, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--native',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--replicate',type=int,default=0);p.add_argument('--blocks',type=int,default=8)
    p.add_argument('--stage',choices=['canary','test'],default='test');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    import torch,flashinfer
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    seed=20261001+a.replicate;torch.manual_seed(seed)
    hw=torch.cuda.get_device_properties(0);sms=hw.multi_processor_count
    native=Native(a.native);sel=json.loads((a.analysis/'model_selection.json').read_text())
    if sha(a.analysis/'model_selection.json')!=native.info['model_selection_sha256']:
        raise ValueError('Different frozen model selection')
    validation=json.loads((a.native/'validation.json').read_text())
    if not validation['all_policies_identical']:raise ValueError('Native equivalence not passed')
    selected=[s for s in corpus() if s.split=='test']
    if a.stage=='canary':
        names={'discovery-eq255-A','discovery-eq255-B','Q512-K1536-n8-A'}
        selected=[s for s in corpus() if s.name in names]
        assert len(selected)==3
    suite=[(s,hq,hkv) for s in selected for hq,hkv in ((16,4),(32,8))]
    random.Random(seed).shuffle(suite)
    arms=('auto','fixed_head','native_run','native_cycle')
    manifest={'stage':a.stage,'kind':'frozen_native_plan_run_validation','replicate':a.replicate,
        'seed':seed,'gpu':hw.name,'sms':sms,'torch':torch.__version__,'cuda':torch.version.cuda,
        'flashinfer':flashinfer.__version__,'backend':'fa2','dtype':'float16','head_dim':128,
        'cuda_graph':False,'model_selection_sha256':sha(a.analysis/'model_selection.json'),
        'native_manifest_sha256':sha(a.native/'manifest.json'),'blocks':a.blocks,
        'arms':arms,'planned_cells':len(suite),'full_vllm_serving':False,
        'primary':'native_cycle versus train-selected fixed_head',
        'timing':'CPU selection + public plan + run + GPU completion; no selector cache',
        'source_sha256':{f:sha(Path(__file__).with_name(f)) for f in
            ('measure_native.py','native.py','native_selector.cpp','geometry.py')},
        'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[];started=time.perf_counter()
    for shape,hq,hkv in suite:
        qoffset=[0];koffset=[0]
        for ql,kl in zip(shape.q,shape.k):
            qoffset.append(qoffset[-1]+ql);koffset.append(koffset[-1]+ql+kl)
        q=torch.randn((qoffset[-1],hq,128),device='cuda',dtype=torch.float16)
        k=torch.randn((koffset[-1],hkv,128),device='cuda',dtype=torch.float16);v=torch.randn_like(k)
        qptr=torch.tensor(qoffset,device='cuda',dtype=torch.int32)
        kptr=torch.tensor(koffset,device='cuda',dtype=torch.int32)
        out=torch.empty_like(q)
        reserved=max(1024*1024,max(workspace_bytes(shape,hq,hkv,sms,p) for p in POLICIES))
        scratch=torch.empty(reserved,device='cuda',dtype=torch.uint8)
        wrappers={arm:flashinfer.BatchPrefillWithRaggedKVCacheWrapper(scratch,kv_layout='NHD',backend='fa2') for arm in arms}
        fixed=sel['fixed'][hw.name+'|median_cycle_ms']['per_head'][str(hq)]
        def execute(arm):
            if arm=='auto':mode='auto'
            elif arm=='fixed_head':mode=fixed
            else:
                target='median_ms' if arm=='native_run' else 'median_cycle_ms'
                mode=native.choose(hw.name+'|'+target+'|causal',shape,hq,hkv,sms)
            opts=dict(causal=True,q_data_type=torch.float16,kv_data_type=torch.float16)
            if mode=='none':opts['disable_split_kv']=True
            elif mode!='auto':opts['fixed_split_size']=int(mode[1:])
            w=wrappers[arm];w.plan(qptr,kptr,hq,hkv,128,**opts);w.run(q,k,v,out=out)
            return mode
        policies={};diffs={};ref=None;maxref=0.;nref=0
        for arm in arms:
            policies[arm]=execute(arm);torch.cuda.synchronize();got=out.clone()
            expected=plan(shape,hq,hkv,sms,policies[arm]);info=list(wrappers[arm]._plan_info)
            assert int(info[0])==expected['grid_x'] and int(info[3])==expected['tile'] and bool(info[14])==expected['split']
            if ref is None:
                ref=got.clone()
                for ri,(ql,depth) in enumerate(zip(shape.q,shape.k)):
                    for qi in sorted(set((0,ql-1))):
                        for h in (0,hq-1):
                            end=koffset[ri]+depth+qi+1;kh=h//(hq//hkv)
                            qq=q[qoffset[ri]+qi,h].float();kk=k[koffset[ri]:end,kh].float();vv=v[koffset[ri]:end,kh].float()
                            value=torch.softmax((kk@qq)/math.sqrt(128),dim=0)@vv
                            actual=got[qoffset[ri]+qi,h].float()
                            torch.testing.assert_close(actual,value,atol=.005,rtol=.02)
                            maxref=max(maxref,float((actual-value).abs().max()));nref+=1
            torch.testing.assert_close(got,ref,atol=.005,rtol=.02)
            diffs[arm]=float((got-ref).abs().max())
            for _ in range(3):execute(arm)
            torch.cuda.synchronize()
        block_values={arm:[] for arm in arms}
        for block in range(a.blocks):
            order=list(arms);random.Random(f'{seed}:{shape.name}:{hq}:{block}').shuffle(order)
            paired={arm:[] for arm in arms}
            for arm in order+order[::-1]:
                torch.cuda.synchronize();tick=time.perf_counter_ns()
                mode=execute(arm);torch.cuda.synchronize()
                elapsed=(time.perf_counter_ns()-tick)/1e6
                if mode!=policies[arm]:raise ValueError('Non-deterministic frozen selector')
                paired[arm].append(elapsed)
            for arm in arms:block_values[arm].append(statistics.mean(paired[arm]))
        row={'shape':shape.record(),'hq':hq,'hkv':hkv,'gpu':hw.name,'replicate':a.replicate,
             'selected_policy':policies,'cycle_ms':block_values,
             'median_ms':{arm:statistics.median(x) for arm,x in block_values.items()},
             'full_output_max_abs':diffs,'fp32_vectors':nref,'fp32_max_abs':maxref,
             'scratch_reserved_bytes':reserved,'correctness_pass':True,'planner_match':True}
        rows.append(row)
        with (a.out/'measurements.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('NATIVE_CASE',shape.name,hq,row['median_ms'],flush=True)
        del wrappers,scratch,q,k,v,out,got,ref
        torch.cuda.empty_cache()
    summary=dict(manifest,completed_cells=len(rows),correctness_all=True,
        elapsed_seconds=time.perf_counter()-started,measurements_sha256=sha(a.out/'measurements.jsonl'))
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':main()
