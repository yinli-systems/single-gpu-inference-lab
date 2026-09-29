"""Fail-closed offline analysis; CPU-only. GPU timings are never synthesized."""
from __future__ import annotations
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from measure_order import cases
from plan_contract import POLICIES,canonical_hash,validate_descriptors,order_indices

KEYS=('case','state','dtype','split','block','policy','mode')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def validate_rows(rows,env):
    case_map={(c['case'],c['state']):c for c in env['case_manifest']}
    expected=set(itertools.product(case_map,env['dtype'],('auto','unsplit'),range(env['blocks']),POLICIES,('eager','graph')))
    seen={};trial_ids=set()
    for r in rows:
        key=((r['case'],r['state']),r['dtype'],r['split'],r['block'],r['policy'],r['mode'])
        if key not in expected or key in seen:raise ValueError('unexpected or duplicate timing cell')
        if type(r['rep']) is not int or r['rep']!=env['rep']:raise ValueError('wrong repeat')
        if type(r['block']) is not int:raise ValueError('invalid block')
        if r['holdout'] is not case_map[key[0]]['holdout']:raise ValueError('changed holdout designation')
        for f in ('device_us','wall_us','apply_us','plan_us'):
            if type(r[f]) not in (int,float) or not math.isfinite(r[f]) or r[f]<=0:raise ValueError('nonpositive/nonfinite timing')
        if r['trial_id'] in trial_ids:raise ValueError('duplicate trial id')
        trial_ids.add(r['trial_id']);seen[key]=r
    if set(seen)!=expected:raise ValueError('incomplete matrix; no favorable subset')
    return seen

def load(root,stage):
    paths=sorted((root/'runs').glob(stage+'-*'));expected_runs=2 if stage=='canary' else 6
    if len(paths)!=expected_runs:raise ValueError('expected %d complete runs, found %d'%(expected_runs,len(paths)))
    result=[];sources=set();commits=set()
    for path in paths:
        if (path/'failure.json').exists():raise ValueError('failed run retained: '+str(path))
        complete=json.loads((path/'complete.json').read_text());env=json.loads((path/'environment.json').read_text())
        if complete['complete'] is not True or complete['stage']!=stage or env['stage']!=stage:raise ValueError('incomplete stage')
        if not (path/'completed_utc.txt').exists():raise ValueError('launcher not complete')
        for f,digest in complete['files'].items():
            if Path(f).name!=f or sha(path/f)!=digest:raise ValueError('raw evidence hash mismatch')
        if env['case_manifest']!=cases(stage) or env['case_hash']!=canonical_hash(cases(stage)):raise ValueError('changed case manifest')
        if complete['source']!=env['source'] or complete['case_hash']!=env['case_hash']:raise ValueError('mixed source/data')
        if env['flashinfer']!='0.6.18' or env['scratch_MiB']!=512 or env['full_model'] is not False:raise ValueError('wrong experiment')
        sources.add(canonical_hash(env['source']));commits.add((path/'source_commit.txt').read_text().strip())
        rows=json.loads((path/'measurements.json').read_text());index=validate_rows(rows,env)
        if complete['rows']!=len(rows) or complete['expected']!=len(rows):raise ValueError('completion count mismatch')
        checks=json.loads((path/'qualification.json').read_text())
        expected_keys=set(itertools.product([(c['case'],c['state']) for c in cases(stage)],env['dtype'],('auto','unsplit'),POLICIES))
        seen=set()
        for c in checks:
            k=((c['case'],c['state']),c['dtype'],c['split'],c['policy'])
            if k in seen or k not in expected_keys or any(c[n] is not True for n in ('full_output_exact','lse_exact','graph_exact')):
                raise ValueError('invalid qualification')
            seen.add(k)
        if seen!=expected_keys or complete['qualifications']!=len(checks):raise ValueError('qualification coverage gap')
        plans=json.loads((path/'plans.json').read_text());plan_keys=set();policy_hashes={}
        cm={(c['case'],c['state']):c for c in cases(stage)}
        for p in plans:
            pk=((p['case'],p['state']),p['dtype'],p['split'])
            if pk in plan_keys:raise ValueError('duplicate plan record')
            plan_keys.add(pk);pc=p['plan'];case=cm[pk[0]]
            L=[q+k for q,k in zip(case['query'],case['cached'])]
            if pc['query']!=case['query'] or pc['total_kv']!=L:raise ValueError('plan geometry mismatch')
            validate_descriptors(pc['descriptors'],pc['query'],L,pc['info'],pc['chunk'],4,pc['output_indptr'],pc['merge_indptr'])
            ref=p['reference']
            if ref['selected_fp32_vectors']!=sum(len({0,q//2,q-1})*32 for q in case['query']):raise ValueError('reference coverage mismatch')
            if any(not math.isfinite(ref[k]) or ref[k]<0 for k in ('max_abs','lse_max_abs')):raise ValueError('bad reference error')
            for policy in POLICIES:
                ids=order_indices(pc['descriptors'],pc['query'],L,pc['info']['cta_tile_q'],pc['chunk'],bool(pc['info']['split_kv']),4,policy)
                policy_hashes[pk+(policy,)]=canonical_hash([pc['descriptors'][i] for i in ids])
        expected_plans=set(itertools.product(cm,env['dtype'],('auto','unsplit')))
        if plan_keys!=expected_plans:raise ValueError('plan coverage mismatch')
        for r in rows+checks:
            pk=((r['case'],r['state']),r['dtype'],r['split'],r['policy'])
            if r['descriptor_hash']!=policy_hashes[pk]:raise ValueError('timing/qualification descriptor mismatch')

        result.append(dict(path=str(path),env=env,rows=rows,index=index,hashes=complete['files']))
    if len(sources)!=1 or len(commits)!=1:raise ValueError('mixed measurement sources')
    grouped=defaultdict(list)
    for r in result:grouped[r['env']['gpu']].append(r)
    if len(grouped)!=2 or not any('RTX 4090' in n for n in grouped) or not any('RTX 5090' in n for n in grouped):raise ValueError('missing GPU family')
    for gpu,rs in grouped.items():
        target={0} if stage=='canary' else {0,1,2}
        if len(rs)!=len(target) or {r['env']['rep'] for r in rs}!=target:raise ValueError('missing/duplicate process repeat')
        rs.sort(key=lambda r:r['env']['rep'])
    return grouped,commits.pop()

@lru_cache(maxsize=1)
def weights():
    rng=random.Random(20261201);draws=[]
    for _ in range(10000):
        w=[0]*18
        for j in range(3):
            p=rng.randrange(3)
            for b in range(6):w[p*6+rng.randrange(6)]+=1
        draws.append(tuple(w))
    return draws

def quantile(sorted_values,p):
    x=(len(sorted_values)-1)*p;i=int(x);f=x-i
    return sorted_values[i]*(1-f)+sorted_values[min(i+1,len(sorted_values)-1)]*f

def interval(grid,confidence=.95,log=False):
    if len(grid)!=3 or any(len(row)!=6 for row in grid):raise ValueError('three process repeats and six blocks required')
    values=[x for row in grid for x in row]
    if not all(math.isfinite(x) for x in values):raise ValueError('nonfinite paired value')
    draws=sorted(sum(n*x for n,x in zip(w,values))/18 for w in weights())
    tail=(1-confidence)/2
    out=dict(mean=statistics.mean(values),CI=[quantile(draws,tail),quantile(draws,1-tail)],confidence=confidence,
             units='3 process repeats x 6 matched blocks',draws=10000,seed=20261201)
    if log:out['ratio']=math.exp(out.pop('mean'));out['CI']=[math.exp(x) for x in out['CI']]
    return out

def paired_grid(runs,cells,policy,mode,metric='device_us',difference=False):
    grid=[]
    for run in runs:
        row=[]
        for block in range(run['env']['blocks']):
            values=[]
            for case,dtype,split in cells:
                base=run['index'][(case,dtype,split,block,'identity',mode)][metric]
                alt=run['index'][(case,dtype,split,block,policy,mode)][metric]
                values.append(alt-base if difference else math.log(base/alt))
            row.append(statistics.mean(values))
        grid.append(row)
    return grid

def analyze(root,stage,out):
    grouped,commit=load(root,stage)
    if out.exists():raise FileExistsError('preserve existing result')
    report=dict(stage=stage,complete=True,measurement_commit=commit,analysis_sha256=sha(__file__),
                serving_promotion=False,full_model=False,setup_in_run_metric=False,GPUs={})
    md=['# Metadata-only ordering results','',
        'Real FlashInfer runs. Run-only ratios exclude metadata setup; these are not serving speedups.',
        'All declared cells, failed controls and process repeats are retained.','',
        '|GPU|set|mode|policy|native/candidate ratio [95% CI]|','|---|---|---|---|---|']
    for gpu,runs in sorted(grouped.items()):
        env=runs[0]['env'];allcells=list(itertools.product([(c['case'],c['state']) for c in cases(stage)],env['dtype'],('auto','unsplit')))
        holdout=[c for c in allcells if c[0][0].startswith('holdout-')]
        summaries=[];control_failures=[];cell_results=[]
        sets=[('discovery',[c for c in allcells if c not in holdout])]+([('holdout',holdout)] if holdout else [])
        for set_name,cells in sets:
            for mode,policy in itertools.product(('eager','graph'),POLICIES[1:]):
                grid=paired_grid(runs,cells,policy,mode)
                stats=interval(grid,log=True) if stage=='test' else dict(ratio=math.exp(statistics.mean(x for row in grid for x in row)),CI=None,qualification_only=True)
                item=dict(set=set_name,mode=mode,policy=policy,primary=(set_name=='holdout' and mode=='graph' and policy=='heavy_first'),**stats)
                summaries.append(item)
                ci='unscored canary' if stats['CI'] is None else '%.5f, %.5f'%tuple(stats['CI'])
                md.append('|%s|%s|%s|%s|%.5f [%s]|'%(gpu,set_name,mode,policy,stats['ratio'],ci))
        for cell in allcells:
            for mode in ('eager','graph'):
                base=[run['index'][(cell[0],cell[1],cell[2],b,'identity',mode)]['device_us'] for run in runs for b in range(env['blocks'])]
                row=dict(case=cell[0][0],state=cell[0][1],dtype=cell[1],split=cell[2],mode=mode,
                         native_median_us=statistics.median(base),holdout=cell in holdout,ratios={})
                for policy in POLICIES[1:]:
                    grid=paired_grid(runs,[cell],policy,mode)
                    row['ratios'][policy]=math.exp(statistics.mean(x for line in grid for x in line))
                delta=paired_grid(runs,[cell],'identity_repeat',mode,difference=True)
                if stage=='test':
                    st=interval(delta,confidence=.90);epsilon=max(3.,.02*row['native_median_us'])
                    row['AA']=dict(st,epsilon_us=epsilon,equivalent=st['CI'][0]>=-epsilon and st['CI'][1]<=epsilon)
                    if not row['AA']['equivalent']:control_failures.append({k:row[k] for k in ('case','state','dtype','split','mode','holdout','AA')})
                cell_results.append(row)
        allrows=[r for run in runs for r in run['rows']]
        report['GPUs'][gpu]=dict(runs=len(runs),timing_rows=len(allrows),summaries=summaries,cells=cell_results,
             AA_failed=control_failures,policy_apply_median_us=statistics.median(r['apply_us'] for r in allrows),
             qualification_records=sum(len(json.loads((Path(run['path'])/'qualification.json').read_text())) for run in runs),
             environments=[run['env'] for run in runs],raw_hashes={Path(run['path']).name:run['hashes'] for run in runs})
        md+=['','A/A failures on %s: %d. Median prototype metadata setup %.1f us (NOT included in run-only ratios).'%(gpu,len(control_failures),report['GPUs'][gpu]['policy_apply_median_us']),'']
    out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out/'RESULTS.md').write_text('\n'.join(md)+'\n')
    print('\n'.join(md),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['canary','test'],required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.root,a.stage,a.out)
