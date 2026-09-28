"""Recompute milestone summaries from complete, hash-checked measured evidence."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import pickle
import statistics
import numpy as np
from analyze import load, design, policy_metrics, sha, write_json


def prediction_intervals(root,out):
    analysis=root/'analysis';rows,_,_=load(root,'test')
    selection=json.loads((analysis/'model_selection.json').read_text())
    frozen=json.loads((analysis/'freeze.json').read_text())
    if sha(analysis/'model_selection.json')!=frozen['model_selection_sha256']:
        raise ValueError('Changed model selection')
    results={}
    for gpu in sorted({r['gpu'] for r in rows}):
        rs=[r for r in rows if r['gpu']==gpu]
        groups=np.array([r['shape']['family'] for r in rs]);families=sorted(set(groups))
        for target in ('median_ms','median_cycle_ms'):
            key=gpu+'|'+target
            baseline=selection['strong_nonplan'][key];errors={}
            for kind in (baseline,'plan','causal'):
                item=selection['models'][key+'|'+kind];p=analysis/item['file']
                if sha(p)!=item['sha256']:raise ValueError('Changed model')
                model=pickle.loads(p.read_bytes())
                pred=np.exp(model.predict(design(rs,kind)))
                errors[kind]=abs(pred-np.array([r[target] for r in rs]))
            for comparator in (baseline,'plan'):
                left=np.array([errors[comparator][groups==g].mean() for g in families])
                right=np.array([errors['causal'][groups==g].mean() for g in families])
                idx=np.random.default_rng(20260928).integers(0,len(families),size=(20000,len(families)))
                improvement=1-right[idx].mean(1)/left[idx].mean(1)
                results[key+'|vs_'+comparator]={
                    'comparator':comparator,'causal_mae_ms':float(right.mean()),
                    'comparator_mae_ms':float(left.mean()),
                    'relative_mae_reduction':float(1-right.mean()/left.mean()),
                    'family_bootstrap_95pct_CI':list(map(float,np.quantile(improvement,[.025,.975]))),
                    'families':families,'scope':'Fixed-predictor heldout-family uncertainty; not a selection-adjusted or serving result'}
    write_json(out/'prediction_uncertainty.json',results)


def completed_rows(root,pattern,expected_records,expected_runs):
    rows=[];provenance={};runs=[]
    for p in sorted((root/'runs').glob(pattern+'/summary.json')):
        m=json.loads(p.read_text());raw=p.with_name('measurements.jsonl')
        if sha(raw)!=m['measurements_sha256']:raise ValueError('Raw hash mismatch')
        rr=[json.loads(line) for line in raw.read_text().splitlines()]
        if len(rr)!=expected_records:raise ValueError(f'Incomplete run {p}')
        if m.get('failures') or m.get('correctness_all') is False:raise ValueError('Failed run')
        if len({(r['shape']['name'],r['hq'],r.get('policy')) for r in rr})!=len(rr):
            raise ValueError('Duplicate cells')
        for r in rr:
            if r.get('status','complete')!='complete' or not r.get('planner_match',False):
                raise ValueError('Invalid measured cell')
            if r.get('correctness_pass') is False:raise ValueError('Numerical failure')
        rows.extend(rr);provenance[str(raw)]=sha(raw);runs.append(m)
    if len(runs)!=expected_runs:raise ValueError(f'Expected {expected_runs} runs, got {len(runs)}')
    groups=collections.defaultdict(set)
    for m in runs:
        if m['replicate'] in groups[m['gpu']]:raise ValueError('Duplicate run')
        groups[m['gpu']].add(m['replicate'])
    if len(groups)!=2 or any(v!={0,1,2} for v in groups.values()):
        raise ValueError('Incomplete hardware/seed matrix')
    return rows,provenance,runs


def wave(root,out):
    rows,provenance,runs=completed_rows(root,'full-*',144,6)
    groups=collections.defaultdict(dict)
    for r in rows:
        key=(r['gpu'],r['shape']['family'],r['hq'],r['policy'],r['replicate'])
        groups[key][r['shape']['name'][-1]]=r
    combined=collections.defaultdict(list)
    for (gpu,fam,h,mode,rep),pair in sorted(groups.items()):
        if set(pair)!={'A','B'}:raise ValueError('Unpaired intervention')
        a,b=pair['A'],pair['B'];sa,sb=a['shape'],b['shape']
        work=lambda s:sum(q*k+q*(q+1)//2 for q,k in zip(s['q'],s['k']))
        if work(sa)!=work(sb) or sum(sa['q'])!=sum(sb['q']) or sum(sa['k'])!=sum(sb['k']):
            raise ValueError('Broken equal-work identity')
        if sorted(q+k for q,k in zip(sa['q'],sa['k']))!=sorted(q+k for q,k in zip(sb['q'],sb['k'])):
            raise ValueError('Different total-KV marginals')
        combined[(gpu,fam,h,mode)].append({'replicate':rep,'A_ms':a['median_ms'],
            'B_ms':b['median_ms'],'ratio_A_over_B':a['median_ms']/b['median_ms'],
            'cycle_A_over_B':a['median_cycle_ms']/b['median_cycle_ms'],
            'grid_A':a['plan_summary']['grid_x'],'grid_B':b['plan_summary']['grid_x'],
            'kv_chunk_A':a['plan_summary']['kv_chunk'],'kv_chunk_B':b['plan_summary']['kv_chunk']})
    table=[]
    for (gpu,fam,h,mode),rr in sorted(combined.items()):
        rr.sort(key=lambda r:r['replicate']);ratios=[r['ratio_A_over_B'] for r in rr]
        table.append({'gpu':gpu,'family':fam,'hq':h,'policy':mode,'replicates':rr,
                      'median_ratio':statistics.median(ratios),'replicate_range':[min(ratios),max(ratios)]})
    lookup={(r['gpu'],r['family'],r['hq'],r['policy']):r for r in table}
    tests=[]
    for Q in (1344,2720):
        for h in (16,32):
            a=lookup[('NVIDIA GeForce RTX 5090',f'wave-Q{Q}',h,'none')]['median_ratio']
            b=lookup[('NVIDIA GeForce RTX 4090',f'wave-Q{Q}',h,'none')]['median_ratio']
            tests.append({'Q':Q,'hq':h,'ratio_5090':a,'ratio_4090':b,
                          'registered_directional_gate_pass':a>1.1 and a>b})
    result={'kind':'prospective_hardware_boundary_diagnostic','records':len(rows),'runs':len(runs),
        'new_threshold_predictions':tests,'all_four_predictions_pass':all(t['registered_directional_gate_pass'] for t in tests),
        'cells':table,'raw_provenance':provenance,
        'caveat':'Shape latency ratios, NOT optimization speedups. Three process repetitions; ranges are NOT confidence intervals. Wave quantization and split-KV are prior art.'}
    write_json(out/'wave_results.json',result)


def native(root,out):
    rows,provenance,runs=completed_rows(root,'test-*',144,6)
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['gpu'],r['shape']['name'],r['hq'])].append(r)
    by_gpu=collections.defaultdict(list)
    for (gpu,name,h),rs in sorted(groups.items()):
        if len(rs)!=3:raise ValueError('Missing repetitions')
        if len({json.dumps(r['selected_policy'],sort_keys=True) for r in rs})!=1:
            raise ValueError('Frozen policy differed between repetitions')
        by_gpu[gpu].append({'shape':name,'hq':h,'family':rs[0]['shape']['family'],
            'selected_policy':rs[0]['selected_policy'],
            'median_ms':{arm:statistics.median(r['median_ms'][arm] for r in rs)
                         for arm in ('auto','fixed_head','native_run','native_cycle')}})
    report={'kind':'fresh_frozen_native_cycle_validation','raw_provenance':provenance,
            'runs':len(runs),'geometry_head_repetitions':len(rows),'hardware':{},
            'primary':'native_cycle vs fixed_head','full_model_serving':False,
            'test_geometry_reused':True,'weights_or_policy_retuned':False}
    for gpu,rs in sorted(by_gpu.items()):
        base=[r['median_ms']['fixed_head'] for r in rs];fam=[r['family'] for r in rs]
        oracle=[min(r['median_ms'].values()) for r in rs];metrics={}
        for arm in ('auto','fixed_head','native_run','native_cycle'):
            values=[r['median_ms'][arm] for r in rs]
            metrics[arm]=policy_metrics(values,base,fam,oracle)
        passed=metrics['native_cycle']['family_cluster_bootstrap_95pct_CI'][0]>1
        report['hardware'][gpu]={'metrics':metrics,'primary_promotion_gate_pass':passed,'cells':rs}
        print('NATIVE_LIVE_RESULT',gpu,metrics['native_cycle'],flush=True)
    report['both_hardware_primary_pass']=all(r['primary_promotion_gate_pass'] for r in report['hardware'].values())
    write_json(out/'native_live_results.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prediction','wave','native'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    {'prediction':prediction_intervals,'wave':wave,'native':native}[a.mode](a.root,a.out)
