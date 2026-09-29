"""Retrospective development only; rescoring existing timings, not new GPU runs.

Select on one process, confirm on another, score on the third. All states were
previously exposed. Folds share data and are not independent unseen workloads.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path,PurePosixPath
import statistics
import tarfile

from order_guard import Context, Geometry, Probe, admit, digest, propose, select

ARCHIVE_SHA='70ba29c0e70e1178cc296996bfbc98cefbc52953744d727aabb4545f0c96410b'
HISTORICAL_POLICIES=('identity','identity_repeat','request_reverse','heavy_first','interleave')
ROOT=Path(__file__).resolve().parents[2]


def read_archive(path):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ARCHIVE_SHA:raise ValueError('archive hash mismatch')
    contents={}
    with tarfile.open(path,'r:gz') as t:
        for m in t:
            p=PurePosixPath(m.name)
            if p.is_absolute() or '..' in p.parts or m.name in contents:raise ValueError('unsafe archive path')
            if not m.isfile():raise ValueError('regular files only')
            if m.size>64*1024**2:raise ValueError('oversized member')
            contents[m.name]=t.extractfile(m).read()
    if len(contents)!=140:raise ValueError('archive inventory changed')
    for name,data in contents.items():
        if PurePosixPath(name).name!='complete.json':continue
        c=json.loads(data)
        if c['complete'] is not True:raise ValueError('incomplete run')
        parent=str(PurePosixPath(name).parent)
        for fname,h in c['files'].items():
            if PurePosixPath(fname).name!=fname or hashlib.sha256(contents[parent+'/'+fname]).hexdigest()!=h:
                raise ValueError('run evidence hash mismatch')
    return contents


def load(contents):
    grouped=defaultdict(list)
    for name,data in sorted(contents.items()):
        if not name.startswith('runs/test-') or not name.endswith('/environment.json'):continue
        path=str(PurePosixPath(name).parent);env=json.loads(data)
        rows=json.loads(contents[path+'/measurements.json']);plans=json.loads(contents[path+'/plans.json'])
        checks=json.loads(contents[path+'/qualification.json'])
        if len(rows)!=7200 or len(checks)!=600 or len(plans)!=120:raise ValueError('incomplete matrix')
        if any(not all(c[k] is True for k in ('full_output_exact','lse_exact','graph_exact')) for c in checks):
            raise ValueError('numerical qualification failed')
        if env['blocks']!=6 or env['rep'] not in (0,1,2):raise ValueError('wrong repeat design')
        index={}
        for r in rows:
            k=(r['case'],r['state'],r['dtype'],r['split'],r['mode'],r['block'],r['policy'])
            if k in index:raise ValueError('duplicate row')
            if r['policy'] not in HISTORICAL_POLICIES or r['mode'] not in ('graph','eager'):raise ValueError('unexpected arm')
            if not 0<=r['block']<6 or not all(math.isfinite(r[f]) and r[f]>0 for f in ('device_us','wall_us','apply_us')):
                raise ValueError('invalid timing')
            index[k]=r
        driver=list(csv.DictReader(io.StringIO(env['hardware']['out'])))[0][' driver_version'].strip()
        grouped[env['gpu']].append(dict(path=path,env=env,index=index,rows=rows,plans=plans,driver=driver,plan_index={(p['case'],p['state'],p['dtype'],p['split']):p['plan'] for p in plans}))
    if len(grouped)!=2 or sum(len(rs) for rs in grouped.values())!=6:raise ValueError('missing run')
    for rs in grouped.values():
        rs.sort(key=lambda r:r['env']['rep'])
        if [r['env']['rep'] for r in rs]!=[0,1,2]:raise ValueError('repeat coverage')
    return grouped


def ctx(run,key,relaxed=False):
    e=run['env']
    return Context(e['gpu'], 'DRIVER_IGNORED_DIAGNOSTIC_ONLY' if relaxed else run['driver'],
                   e['torch'],e['cuda'],digest(e['source']),digest(HISTORICAL_POLICIES),digest(run['plan_index'][key[:4]]),
                   key[2],key[4],'historical-warm-fixed-buffers','run_only',1)


def paired(run,key,policy,c):
    out=[]
    for b in range(6):
        x=run['index'][key+(b,'identity')];y=run['index'][key+(b,policy)]
        out.append(Probe(run['path']+'/'+y['trial_id'],c.key,policy,x['device_us'],y['device_us'],True,True))
    return out


def summarize(rows):
    ratios=[r['ratio'] for r in rows]
    return dict(cells=len(rows),post_selection_run_only_ratio=True,geomean=math.exp(statistics.mean(map(math.log,ratios))),
                worst_ratio=min(ratios),slowdown_over_1pct=sum(x<1/1.01 for x in ratios),
                policy_counts=dict(Counter(r['policy'] for r in rows)),
                gate_reasons=dict(Counter(r['reason'] for r in rows)))


def analyze(archive):
    contents=read_archive(archive);groups=load(contents)
    report=dict(archive_sha256=ARCHIVE_SHA,hashed_files=len(contents),formal_rows=43200,GPU_rerun=False,
                exposed_development_only=True,production_promotion=False,
                note='Offline counterfactual on historical warm run-only timings. Not an observed dispatcher, not unseen shape validation.',GPUs={})
    for gpu,runs in groups.items():
        keys=sorted({k[:5] for k in runs[0]['index'] if k[4]=='graph'})
        variants={};adverse=[]
        for relaxed in (False,True):
            for budget in (None,36,1000,10000):
                rows=[]
                for fold in range(3):
                    a,b,c=runs[fold],runs[(fold+1)%3],runs[(fold+2)%3]
                    for key in keys:
                        ca,cb,cc=ctx(a,key,relaxed),ctx(b,key,relaxed),ctx(c,key,relaxed)
                        selection=sum((paired(a,key,p,ca) for p in HISTORICAL_POLICIES[2:]),[])
                        policy=select(ca,selection)
                        # Budget=None is a deliberately cost-free diagnostic; not deployment admission.
                        sample_cost=sum(a['index'][key+(j,p)]['calls']*a['index'][key+(j,p)]['wall_us']+
                                        a['index'][key+(j,p)]['apply_us'] for j in range(6) for p in HISTORICAL_POLICIES)
                        sample_cost+=sum(b['index'][key+(j,p)]['calls']*b['index'][key+(j,p)]['wall_us']+
                                         b['index'][key+(j,p)]['apply_us'] for j in range(6) for p in
                                         ('identity','identity_repeat',policy if policy!='identity' else 'heavy_first'))
                        conf=paired(b,key,policy,cb) if policy!='identity' else []
                        gate=admit(ca,policy,{p.trial_id for p in selection},conf,paired(b,key,'identity_repeat',cb),
                                   probe_cost_us=0. if budget is None else sample_cost,extra_setup_us=0.,dispatch_us=0.,
                                   expected_reuses=2**31-1 if budget is None else budget)
                        if cc.key!=ca.key:gate=dict(policy='identity',reason='evaluation_context_mismatch')
                        chosen=gate['policy']
                        rr=math.exp(statistics.mean(math.log(c['index'][key+(j,'identity')]['device_us']/
                                                           c['index'][key+(j,chosen)]['device_us']) for j in range(6)))
                        row=dict(fold=fold,key=key,policy=chosen,ratio=rr,reason=gate['reason'],
                                 probe_cost_lower_bound_us=sample_cost,break_even_reuses=gate.get('break_even_reuses'))
                        if budget is not None:
                            base_time=statistics.mean(c['index'][key+(j,'identity')]['device_us'] for j in range(6))
                            chosen_time=statistics.mean(c['index'][key+(j,chosen)]['device_us'] for j in range(6))
                            row['optimistic_total_cost_ratio']=budget*base_time/(sample_cost+budget*chosen_time)
                            row['sunk_probe_cost_charged_on_fallback']=True
                        rows.append(row)
                name=('ignore_driver_diagnostic' if relaxed else 'strict_environment')+'/'+('uncharged' if budget is None else str(budget))
                summary=summarize(rows)
                if budget is not None:
                    net=[r['optimistic_total_cost_ratio'] for r in rows]
                    summary['optimistic_total_cost_geomean']=math.exp(statistics.mean(map(math.log,net)))
                    summary['optimistic_total_cost_worst']=min(net)
                variants[name]=dict(summary=summary,rows=rows,
                    overhead_note='Probe cost is an optimistic lower bound from existing charged calls/setup, excluding JIT, independent-reference checks and warmup; dispatch cost is assumed zero, not measured.')
        for run in runs:
            key=('holdout-01','opposite','float16','unsplit','graph')
            m={p:statistics.median(run['index'][key+(b,p)]['device_us'] for b in range(6)) for p in HISTORICAL_POLICIES}
            adverse.append(dict(rep=run['env']['rep'],driver=run['driver'],median_us=m,heavy_ratio=m['identity']/m['heavy_first']))
        proposals=Counter();logical_checks=0
        for p in runs[0]['plans']:
            plan=p['plan'];g=Geometry(tuple(plan['query']),tuple(plan['total_kv']),4,plan['info']['cta_tile_q'],plan['chunk'],bool(plan['info']['split_kv']))
            desc=list(map(tuple,plan['descriptors']))
            for policy in ('causal_heavy','locality_packet8'):
                order=propose(g,desc,policy);logical_checks+=1
                if order!=tuple(range(len(desc))):proposals[policy]+=1
        report['GPUs'][gpu]=dict(drivers=[r['driver'] for r in runs],adverse_case=adverse,variants=variants,
              new_proposals_structural_checks=logical_checks,changed_plans=dict(proposals),new_proposals_GPU_tested=False)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,default=ROOT/'benchmarks/results/plan-order-mechanism/raw-evidence.tar.gz')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('preserve previous analysis')
    result=analyze(a.archive);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({g:{k:v['summary'] for k,v in r['variants'].items()} for g,r in result['GPUs'].items()},indent=2))
