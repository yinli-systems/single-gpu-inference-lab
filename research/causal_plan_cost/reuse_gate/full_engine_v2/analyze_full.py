"""Evaluate complete full-engine runs; never silently drop failed requests/jobs.

The one-job qualification screen has no inferential confidence interval.
Six counterbalanced triplets form the units for conditional test inference.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from measure_engine import summarize

ARMS=('auto','fixed_budget','native_cycle')
WORKLOADS=('burst','staggered')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confidence(log_ratios):
    import numpy as np
    if len(log_ratios)!=6:raise ValueError('six matched independent triplets required')
    data=np.asarray(log_ratios,dtype=float)
    if not np.isfinite(data).all():raise ValueError('finite positive timing ratios required')
    draws=np.random.default_rng(20260928).choice(data,size=(20000,6),replace=True).mean(1)
    return dict(ratio=float(np.exp(data.mean())),CI95=list(map(float,np.exp(np.quantile(draws,[.025,.975])))),
                units=6,draws=20000,seed=20260928)


def load(root,stage):
    paths=sorted((root/'runs').glob(stage+'-*'))
    expected=1 if stage=='screen' else 6
    if len(paths)!=expected:raise ValueError(f'Expected {expected} complete {stage} job directories; found {len(paths)}')
    jobs=[];source_identities=set();replicas=set()
    orders=list(itertools.permutations(ARMS))
    for path in paths:
        done=json.loads((path/'complete.json').read_text())
        if not done['all_arms_complete'] or done['stage']!=stage:raise ValueError('incomplete job')
        rep=done['replicate']
        if rep in replicas:raise ValueError('duplicate process-order replicate')
        replicas.add(rep)
        order=tuple(json.loads((path/'arm_order.json').read_text()))
        if order!=orders[rep%6]:raise ValueError('unregistered arm order')
        receipts=json.loads((path/'receipts.json').read_text())
        if [r['arm'] for r in receipts]!=list(order) or any(r['returncode']!=0 for r in receipts):raise ValueError('failed or missing model arm')
        job=dict(path=str(path),replicate=rep,order=order,arms={},raw_hashes={})
        for arm in ARMS:
            directory=path/arm;metadata=json.loads((directory/'summary.json').read_text())
            if not metadata['complete'] or metadata['arm']!=arm or metadata['replicate']!=rep:raise ValueError('wrong model-arm metadata')
            if (metadata['requested_KV_blocks'],metadata['page_size'],metadata['scratch_MiB'],metadata['dtype'],metadata['eager'])!=(2048,16,512,'float16',True):raise ValueError('changed resource/precision configuration')
            source_identities.add(json.dumps({Path(k).name:v for k,v in metadata['source_sha256'].items()},sort_keys=True))
            data={}
            for name in WORKLOADS:
                f=directory/('measured-'+name+'.json');r=json.loads(f.read_text())
                wf=json.loads((directory/('warmup-'+name+'.json')).read_text())
                if not r['complete'] or not wf['complete'] or r['workload_sha256']!=wf['workload_sha256']:raise ValueError('warmup/measurement mismatch')
                if r['router']['counts'].get('qualified_plans',0)<=0:raise ValueError('target adapter was not executed')
                from engine_screen import workload
                cells=workload(name)
                if r['workload_sha256']!=hashlib.sha256(json.dumps(cells,sort_keys=True).encode()).hexdigest():raise ValueError('unregistered workload')
                records={x['id']:x for x in r['requests_detail']}
                if len(records)!=len(r['requests_detail']):raise ValueError('duplicate output request')
                recomputed=summarize(cells,records,r['elapsed_seconds'])
                for key in ('output_tokens','requests','tokens_per_second','strict_slo_count','lenient_slo_count','strict_slo_goodput','lenient_slo_goodput'):
                    if abs(r[key]-recomputed[key])>1e-9:raise ValueError('summary not reproduced from request evidence: '+key)
                data[name]=r;job['raw_hashes'][str(f)]=sha(f)
            job['arms'][arm]=dict(metadata=metadata,workloads=data)
        job['parity']={}
        for name in WORKLOADS:
            ref={r['id']:r['token_ids'] for r in job['arms']['auto']['workloads'][name]['requests_detail']}
            job['parity'][name]={arm:{r['id']:r['token_ids'] for r in job['arms'][arm]['workloads'][name]['requests_detail']}==ref for arm in ARMS}
        verified=all(v for w in job['parity'].values() for v in w.values())
        if verified!=done['greedy_outputs_identical']:raise ValueError('parity summary differs from token evidence')
        jobs.append(job)
    if len(source_identities)!=1:raise ValueError('mixed runtime/measurement source hashes')
    if stage=='test' and replicas!=set(range(6)):raise ValueError('all six arm orders required')
    return jobs


def analyze(root,out,stage):
    if out.exists():raise FileExistsError('preserve prior results')
    jobs=load(root,stage);flat=[]
    for job in jobs:
        for arm in ARMS:
            for name in WORKLOADS:
                r=job['arms'][arm]['workloads'][name]
                flat.append(dict(replicate=job['replicate'],arm=arm,workload=name,
                    elapsed_seconds=r['elapsed_seconds'],tokens_per_second=r['tokens_per_second'],
                    offered_ttft_p95=r['offered_ttft_seconds']['p95'],tpot_p95=r['tpot_seconds']['p95'],
                    latency_p95=r['request_latency_seconds']['p95'],latency_p99=r['request_latency_seconds']['p99'],
                    strict_slo_count=r['strict_slo_count'],strict_slo_goodput=r['strict_slo_goodput'],
                    lenient_slo_count=r['lenient_slo_count'],lenient_slo_goodput=r['lenient_slo_goodput'],
                    plans=r['router']['counts'],cache_hits=r['router']['cache_hits'],
                    cache_misses=r['router']['cache_misses'],greedy_parity=job['parity'][name][arm]))
    comparisons={}
    for baseline in ('auto','fixed_budget'):
        logs=[];slo=[]
        for job in jobs:
            a=job['arms']['native_cycle']['workloads'];b=job['arms'][baseline]['workloads']
            logs.append(statistics.mean(math.log(a[n]['tokens_per_second']/b[n]['tokens_per_second']) for n in WORKLOADS))
            slo.append(sum(a[n]['strict_slo_goodput']-b[n]['strict_slo_goodput'] for n in WORKLOADS))
        comparisons[baseline]=confidence(logs) if stage=='test' else dict(ratio=math.exp(logs[0]),CI95=None,qualification_only=True)
        comparisons[baseline]['strict_slo_goodput_delta_sum']=sum(slo)
        comparisons[baseline]['paired_log_ratios']=logs
    parity=all(v for job in jobs for w in job['parity'].values() for v in w.values())
    capacity_values=[job['arms'][arm]['metadata']['actual_config_KV_blocks'] for job in jobs for arm in ARMS]
    capacity=all(v==2048 for v in capacity_values)
    strict_events=sum(r['strict_slo_count'] for r in flat)
    promote=(stage=='test' and parity and capacity and strict_events>0 and
             all(c['CI95'][0]>1 and c['strict_slo_goodput_delta_sum']>=0 for c in comparisons.values()))
    result=dict(stage=stage,complete=True,jobs=len(jobs),model_processes=len(jobs)*3,
        measured_requests=len(jobs)*60,warmup_requests=len(jobs)*60,full_model=True,HTTP_network=False,
        greedy_parity=parity,actual_KV_blocks=capacity_values,capacity_verified=capacity,
        rows=flat,comparisons=comparisons,strict_slo_events=strict_events,promoted=promote,
        inference='Qualification only; no statistical superiority claim.' if stage=='screen' else
            'Six matched process-order triplets, two fixed synthetic workloads. Conditional bootstrap, not workload-population evidence.',
        raw_hashes={k:v for job in jobs for k,v in job['raw_hashes'].items()},
        source_sha256=sha(__file__))
    out.mkdir(parents=True);(out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('FULL_ENGINE_ANALYSIS',json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=['screen','test'],required=True);a=p.parse_args();analyze(a.root,a.out,a.stage)
