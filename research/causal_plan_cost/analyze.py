"""Train-only selection, sealed test evaluation, and family-level uncertainty.

Usage: analyze.py fit --root ROOT --out OUT
       analyze.py evaluate --root ROOT --out OUT
Only six complete stage=full runs are accepted. Freeze models before evaluate.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics
import time
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn
from geometry import Shape, POLICIES, corpus, corpus_json, features

KINDS = ('marginal', 'joint', 'plan', 'causal')
TARGETS = ('median_ms', 'median_cycle_ms')
SEED = 20260928
SPECS = ([{'class': 'ridge', 'alpha': a} for a in (.01, 1., 100.)] +
         [{'class': 'extra_trees', 'leaf': n} for n in (2, 8)])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def shape_of(row):
    s = row['shape']
    return Shape(s['name'], s['family'], s['split'], tuple(s['q']), tuple(s['k']))


def load(root, split):
    """Select requested split; other split timings never enter model fitting."""
    expected = {s.name: s.record() for s in corpus() if s.split == split}
    corpus_hash = hashlib.sha256(corpus_json().encode()).hexdigest()
    groups, provenance, manifests = collections.defaultdict(list), {}, []
    for summary in sorted((root/'runs').glob('full-*/summary.json')):
        meta = json.loads(summary.read_text())
        if meta.get('stage') != 'full':
            continue
        if meta.get('failures') or meta['completed_records'] != meta['planned_records']:
            raise ValueError(f'Incomplete or failed registered run: {summary}')
        if meta['corpus_sha256'] != corpus_hash:
            raise ValueError('Corpus changed after registration')
        raw = summary.with_name('measurements.jsonl')
        digest = sha(raw)
        if digest != meta['measurements_sha256']:
            raise ValueError(f'Modified raw data: {raw}')
        provenance[str(raw)] = digest
        manifests.append(meta)
        seen = set()
        with raw.open() as f:
            for line in f:
                row = json.loads(line)
                if row['shape']['split'] != split:
                    continue
                if row['shape'] != expected.get(row['shape']['name']):
                    raise ValueError('Unregistered geometry')
                if (row['hq'], row['hkv']) not in ((16,4),(32,8)) or row['policy'] not in POLICIES:
                    raise ValueError('Unexpected head shape or policy')
                key = (row['gpu'], row['shape']['name'], row['hq'], row['policy'])
                if key in seen:
                    raise ValueError('Duplicate geometry/head/policy in one run')
                seen.add(key)
                if row['status'] != 'complete' or not row['planner_match']:
                    raise ValueError('Invalid row cannot be silently dropped')
                if not row['full_output_comparison']['passed']:
                    raise ValueError('Failed numerical equivalence')
                if row['policy']=='auto' and not row['fp32_reference']['passed']:
                    raise ValueError('Failed independent reference')
                if row['gpu'] != meta['gpu'] or row['replicate'] != meta['replicate']:
                    raise ValueError('Run metadata mismatch')
                groups[key].append(row)
        if len(seen) != len(expected)*2*len(POLICIES):
            raise ValueError(f'Missing selected-split cells in {raw}')
    if len(manifests) != 6:
        raise ValueError(f'Need six registered full runs, got {len(manifests)}')
    gpu_reps = collections.defaultdict(set)
    for m in manifests:
        if m['replicate'] in gpu_reps[m['gpu']]:
            raise ValueError('Duplicate hardware replicate')
        gpu_reps[m['gpu']].add(m['replicate'])
    if len(gpu_reps) != 2 or any(v != {0,1,2} for v in gpu_reps.values()):
        raise ValueError('Hardware/replicate coverage mismatch')
    if len({json.dumps(m['source_sha256'],sort_keys=True) for m in manifests}) != 1:
        raise ValueError('Mixed measurement source versions')
    rows=[]
    for key, rs in sorted(groups.items()):
        if len(rs) != 3:
            raise ValueError(f'Incomplete repeat group {key}')
        rs.sort(key=lambda r:r['replicate'])
        r = {k:rs[0][k] for k in ('gpu','shape','hq','hkv','sms','policy')}
        for target in TARGETS:
            xs=[x[target] for x in rs]
            if any(not math.isfinite(x) or x <= 0 for x in xs):
                raise ValueError('Nonpositive/invalid timing')
            r[target]=statistics.median(xs)
            r[target+'_replicates']=xs
        r['plan_wall_ms']=statistics.median(x['mean_plan_ms'] for x in rs)
        rows.append(r)
    return rows, provenance, manifests


def design(rows, kind):
    return np.asarray([features(shape_of(r),r['hq'],r['hkv'],r['sms'],r['policy'],kind)
                       for r in rows], dtype=np.float64)


def estimator(spec):
    if spec['class']=='ridge':
        return make_pipeline(StandardScaler(),Ridge(alpha=spec['alpha']))
    return ExtraTreesRegressor(n_estimators=128,min_samples_leaf=spec['leaf'],
                               random_state=SEED,n_jobs=1)


def select_model(rows, kind, target):
    if any(r['shape']['split']!='train' for r in rows):
        raise ValueError('Fitting must only use TRAIN families')
    X=design(rows,kind); y=np.log([r[target] for r in rows])
    groups=np.array([r['shape']['family'] for r in rows]); families=sorted(set(groups))
    table=[]
    for spec in SPECS:
        fold=[]
        for family in families:
            valid=groups==family
            model=estimator(spec).fit(X[~valid],y[~valid])
            fold.append(float(np.mean(np.abs(model.predict(X[valid])-y[valid]))))
        table.append({'spec':spec,'cv_mean_abs_log_error':float(np.mean(fold)),
                      'cv_family_errors':dict(zip(families,fold))})
    best=min(range(len(table)),key=lambda i:(table[i]['cv_mean_abs_log_error'],i))
    choice=table[best]
    model=estimator(choice['spec']).fit(X,y)
    return model,choice,table


def best_fixed(rows,target):
    costs={p:float(np.mean([math.log(r[target]) for r in rows if r['policy']==p]))
           for p in POLICIES}
    return min(POLICIES,key=lambda p:(costs[p],POLICIES.index(p))),costs


def fit(root,out):
    out.mkdir(parents=True,exist_ok=False)
    rows,provenance,manifests=load(root,'train')
    selections={'stage':'train_only_frozen','seed':SEED,'sklearn':sklearn.__version__,
                'numpy':np.__version__,'raw_provenance':provenance,
                'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                'training_families':sorted({r['shape']['family'] for r in rows}),
                'models':{},'fixed':{},'strong_nonplan':{}}
    for gpu in sorted({r['gpu'] for r in rows}):
        rr=[r for r in rows if r['gpu']==gpu]
        for target in TARGETS:
            key=gpu+'|'+target
            fixed,scores=best_fixed(rr,target)
            by_head={str(h):best_fixed([r for r in rr if r['hq']==h],target)[0]
                     for h in sorted({r['hq'] for r in rr})}
            selections['fixed'][key]={'global':fixed,'per_head':by_head,
                                      'training_mean_log_costs':scores}
            for kind in KINDS:
                model,choice,table=select_model(rr,kind,target)
                name=gpu.replace(' ','_')+'-'+target+'-'+kind+'.pkl'
                path=out/name
                path.write_bytes(pickle.dumps(model,protocol=4))
                selections['models'][key+'|'+kind]={
                    'file':name,'sha256':sha(path),'choice':choice,'cv_search':table}
                print('TRAIN_SELECTION',gpu,target,kind,choice,flush=True)
            strong=min(('marginal','joint'),key=lambda kind:
                selections['models'][key+'|'+kind]['choice']['cv_mean_abs_log_error'])
            selections['strong_nonplan'][key]=strong
    write_json(out/'model_selection.json',selections)
    write_json(out/'freeze.json',{'model_selection_sha256':sha(out/'model_selection.json'),
                                 'test_metrics_computed':False})
    print('TRAIN_FREEZE',sha(out/'model_selection.json'),flush=True)


def cost_metrics(actual,pred):
    actual=np.asarray(actual);pred=np.asarray(pred);error=pred-actual
    return {'mae_ms':float(np.mean(abs(error))),
            'p95_abs_ms':float(np.quantile(abs(error),.95)),
            'p95_underprediction_ms':float(np.quantile(np.maximum(-error,0),.95)),
            'mean_abs_log_error':float(np.mean(abs(np.log(pred/actual))))}


def policy_metrics(cost,reference,families,oracle):
    cost=np.asarray(cost);reference=np.asarray(reference);oracle=np.asarray(oracle)
    fam=np.asarray(families)
    groups=sorted(set(families))
    per_family=np.array([np.mean(np.log(reference[fam==f]/cost[fam==f])) for f in groups])
    rng=np.random.default_rng(SEED)
    means=rng.choice(per_family,size=(20000,len(groups)),replace=True).mean(1)
    return {'family_geomean_speed_ratio':float(np.exp(per_family.mean())),
            'family_cluster_bootstrap_95pct_CI':list(map(float,np.exp(np.quantile(means,[.025,.975])))),
            'clusters':len(groups),'geometry_head_cells':len(cost),
            'regressions_over_5pct':int(np.sum(cost>1.05*reference)),
            'max_slowdown_ratio':float(np.max(cost/reference)),
            'median_cost_ms':float(np.median(cost)),
            'mean_regret_to_oracle_ms':float(np.mean(cost-oracle)),
            'geomean_regret_ratio':float(np.exp(np.mean(np.log(cost/oracle))))}


def overhead(model,cell,kind,repeats=15):
    # Only metadata is used. This CPU timing is not GPU-side serving evidence.
    s=shape_of(cell)
    def choose():
        xx=np.array([features(s,cell['hq'],cell['hkv'],cell['sms'],p,kind) for p in POLICIES])
        return int(np.argmin(model.predict(xx)))
    choose();values=[]
    for _ in range(repeats):
        start=time.perf_counter_ns();choose();values.append((time.perf_counter_ns()-start)/1e6)
    return float(statistics.median(values))


def evaluate(root,out):
    if (out/'test_results.json').exists() or (out/'test_decisions.json').exists():
        raise FileExistsError('Existing test results must not be silently overwritten')
    selection_path=out/'model_selection.json'
    frozen=json.loads((out/'freeze.json').read_text())
    if sha(selection_path)!=frozen['model_selection_sha256']:
        raise ValueError('Model choices modified after freeze')
    selected=json.loads(selection_path.read_text())
    rows,provenance,manifests=load(root,'test')
    if provenance!=selected['raw_provenance']:
        raise ValueError('Input evidence changed between fit and evaluate')
    if set(selected['training_families']) & {r['shape']['family'] for r in rows}:
        raise ValueError('Train/test family leakage')
    result={'stage':'heldout_geometry_evaluation','frozen_selection_sha256':sha(selection_path),
            'seed':SEED,'bootstrap_replicates':20000,'hardware':{},
            'raw_provenance':provenance,'novelty_confirmed':False,'full_vllm_evaluated':False,
            'selector_measurement_host':__import__('socket').gethostname(),
            'warning':'Six heldout families per hardware. Inner timings are not independent. '
                      'Additive selector overhead is not integrated production serving latency.'}
    decisions=[]
    for gpu in sorted({r['gpu'] for r in rows}):
        rr=[r for r in rows if r['gpu']==gpu]
        by_cell=collections.defaultdict(dict)
        for r in rr:by_cell[(r['shape']['name'],r['hq'])][r['policy']]=r
        keys=sorted(by_cell);fam=[by_cell[k]['auto']['shape']['family'] for k in keys]
        for target in TARGETS:
            key=gpu+'|'+target
            costs=np.array([[by_cell[k][p][target] for p in POLICIES] for k in keys])
            oracle=costs.min(1)
            global_fixed=selected['fixed'][key]['global']
            hp=selected['fixed'][key]['per_head']
            headfixed=np.array([costs[i,POLICIES.index(hp[str(k[1])])] for i,k in enumerate(keys)])
            section={'prediction':{},'dispatch':{},'selected_global_fixed':global_fixed,
                     'selected_head_fixed':hp,'nonplan_chosen_on_train':selected['strong_nonplan'][key]}
            for i,p in enumerate(POLICIES):
                section['dispatch'][p]=policy_metrics(costs[:,i],headfixed,fam,oracle)
            section['dispatch']['train_fixed_head']=policy_metrics(headfixed,headfixed,fam,oracle)
            section['dispatch']['oracle_diagnostic']=policy_metrics(oracle,headfixed,fam,oracle)
            for kind in KINDS:
                modelinfo=selected['models'][key+'|'+kind];mp=out/modelinfo['file']
                if sha(mp)!=modelinfo['sha256']:raise ValueError('Model bytes changed')
                model=pickle.loads(mp.read_bytes()) # Own locally trained, hash-checked artifact.
                pred=np.exp(model.predict(design(rr,kind)))
                section['prediction'][kind]=cost_metrics([r[target] for r in rr],pred)
                predictions={(r['shape']['name'],r['hq'],r['policy']):v for r,v in zip(rr,pred)}
                pi=np.array([int(np.argmin([predictions[k+(p,)] for p in POLICIES])) for k in keys])
                chosen=costs[np.arange(len(keys)),pi]
                ms=np.array([overhead(model,by_cell[k]['auto'],kind) for k in keys])
                d=policy_metrics(chosen,headfixed,fam,oracle)
                d['selected_policy_counts']=dict(collections.Counter(POLICIES[i] for i in pi))
                d['selector_ms_median']=float(np.median(ms));d['selector_ms_p95']=float(np.quantile(ms,.95))
                d['with_python_selector_additive']=policy_metrics(chosen+ms,headfixed,fam,oracle)
                section['dispatch'][kind]=d
                for i,k in enumerate(keys):
                    decisions.append({'gpu':gpu,'target':target,'kind':kind,'shape':k[0],'hq':k[1],
                        'family':fam[i],'policy':POLICIES[pi[i]],'cost_ms':float(chosen[i]),
                        'fixed_cost_ms':float(headfixed[i]),'oracle_ms':float(oracle[i]),
                        'selector_ms':float(ms[i])})
                other=next(g for g in sorted({r['gpu'] for r in rows}) if g!=gpu)
                otherinfo=selected['models'][other+'|'+target+'|'+kind]
                op=out/otherinfo['file']
                if sha(op)!=otherinfo['sha256']:raise ValueError('Transfer model bytes changed')
                om=pickle.loads(op.read_bytes())
                opred=om.predict(design(rr,kind))
                od={(r['shape']['name'],r['hq'],r['policy']):v for r,v in zip(rr,opred)}
                oi=np.array([int(np.argmin([od[k+(p,)] for p in POLICIES])) for k in keys])
                section['dispatch'][kind]['no_refit_cross_hardware']=policy_metrics(
                    costs[np.arange(len(keys)),oi],headfixed,fam,oracle)
            baseline=section['nonplan_chosen_on_train']
            improvement=1-section['prediction']['causal']['mae_ms']/section['prediction'][baseline]['mae_ms']
            section['causal_mae_improvement_vs_selected_nonplan']=improvement
            section['predictive_gate_pass']=bool(improvement>=.2)
            d=section['dispatch']['causal']
            section['raw_dispatch_gate_pass']=bool(d['family_cluster_bootstrap_95pct_CI'][0]>1)
            section['python_net_dispatch_gate_pass']=bool(d['with_python_selector_additive']['family_cluster_bootstrap_95pct_CI'][0]>1)
            result['hardware'][key]=section
            print('HELDOUT',key,'mae_gain',improvement,'causal_ratio',d['family_geomean_speed_ratio'],
                  'CI',d['family_cluster_bootstrap_95pct_CI'],'python_net',d['with_python_selector_additive']['family_geomean_speed_ratio'],flush=True)
    write_json(out/'test_results.json',result)
    write_json(out/'test_decisions.json',decisions)
    print('TEST_RESULTS_SHA256',sha(out/'test_results.json'),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['fit','evaluate'])
    p.add_argument('--root',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args(); {'fit':fit,'evaluate':evaluate}[a.command](a.root,a.out)


if __name__=='__main__':main()
