"""Complete-matrix analysis of direct eager stack wall times. No model fitting."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np
from contracts import POLICIES,fresh_corpus,corpus_hash,scratch_bytes

ARMS=('auto','fixed_budget','native_cycle','native_run')
REPEATS=(1,2,4,8,16,36)
SEQUENCES={'unchanged':(0,0,0,0),'alternating':(0,1,0,1),'eviction':(0,1,2,0)}
EXPECTED_COUNTERS={'unchanged':(1,3,1,0),'alternating':(4,2,2,0),'eviction':(4,0,4,2)}
SEED=20261011


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def ratio_stats(cost,reference,families):
    x=np.asarray(cost,dtype=float);b=np.asarray(reference,dtype=float);f=np.asarray(families)
    if len(x)!=len(b) or len(f)!=len(x) or not len(x):raise ValueError('aligned nonempty inputs required')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(b)) or min(x.min(),b.min())<=0:raise ValueError('positive finite costs required')
    groups=sorted(set(families));logs=np.asarray([np.mean(np.log(b[f==g]/x[f==g])) for g in groups])
    bootstrap=np.random.default_rng(SEED).choice(logs,size=(20000,len(logs)),replace=True).mean(1)
    return dict(ratio=float(np.exp(logs.mean())),CI95=list(map(float,np.exp(np.quantile(bootstrap,[.025,.975])))),
        families=len(groups),base_geometries=len(x),regressions_gt5pct=int(np.sum(x>1.05*b)),
        max_slowdown=float(np.max(x/b)),median_episode_ms=float(np.median(x)),
        family_log_ratios=dict(zip(groups,map(float,logs))))


def check_counters(c,repeat,sequence):
    expected=EXPECTED_COUNTERS[sequence]
    actual=tuple(c[k] for k in ('plans','hits','misses','evictions'))
    if actual!=expected or c['runs']!=4*repeat:raise ValueError('cache/plan lifetime invariant violated')
    if len(c['policies'])!=4 or any(p not in POLICIES for p in c['policies']):raise ValueError('invalid actual policy trace')


def load(root):
    directories=sorted((root/'runs').glob('test-*'))
    if len(directories)!=6:raise ValueError(f'need six full runs; got {len(directories)}')
    bases={r['name']:{k:r[k] for k in ('name','family','split','q','k')} for r in fresh_corpus() if r['state']==0}
    states=defaultdict(dict)
    for r in fresh_corpus():states[r['family'],r['n']][r['state']]=r
    provenance=[];raw=[];coverage=set();correctness={}
    for directory in directories:
        m=json.loads((directory/'summary.json').read_text())
        if not m['complete'] or m['failures'] or m['stage']!='test':raise ValueError('incomplete run cannot be excluded silently')
        if m['rows']!=648 or m['corpus_sha256']!=corpus_hash():raise ValueError('registered corpus/coverage mismatch')
        if (m['mode'],m['flashinfer'],m['heads'],m['head_dim'])!=('eager','0.6.18',[32,8],128):raise ValueError('wrong execution path')
        identity=(m['gpu'],m['replicate'])
        if identity in coverage:raise ValueError('duplicate process replicate')
        coverage.add(identity)
        f=directory/'measurements.jsonl'
        if sha(f)!=m['measurements_sha256']:raise ValueError('modified raw observations')
        rr=[json.loads(x) for x in f.read_text().splitlines()]
        if len(rr)!=648:raise ValueError('missing raw rows')
        seen=set()
        for r in rr:
            s=r['base'];repeat=r['layers'];sequence=r['sequence'];budget=r['scratch_ceiling']
            if bases.get(s['name'])!=s or repeat not in REPEATS or sequence not in SEQUENCES or budget not in (134217728,536870912):raise ValueError('unregistered cell')
            key=(s['name'],repeat,sequence,budget)
            if key in seen:raise ValueError('duplicated cell')
            seen.add(key)
            if (r['gpu'],r['replicate'])!=identity or not r['correctness_pass']:raise ValueError('metadata/correctness failure')
            for arm in ARMS:
                c=r['counters'][arm];check_counters(c,repeat,sequence)
                for si,mode in zip(SEQUENCES[sequence],c['policies']):
                    st=states[s['family'],len(s['q'])][si]
                    if scratch_bytes(tuple(st['q']),tuple(st['k']),32,8,m['sms'],mode)>budget:raise ValueError('scratch ceiling violated')
                values=r['arms_ms'][arm]
                if len(values)!=4 or any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('invalid timing blocks')
                if abs(statistics.median(values)-r['median_ms'][arm])>1e-10:raise ValueError('summary timing differs')
            # The checks were executed ONCE before all repeat/sequence measurements.
            ck=(m['gpu'],m['replicate'],s['name'],budget)
            item={k:r[k] for k in ('selected_modes_checked','full_tensor_checks','max_abs','fp32_vectors','fp32_max_abs','layer_tensor_bytes','peak_cuda_allocated','peak_cuda_reserved')}
            if ck in correctness:
                for k in ('selected_modes_checked','full_tensor_checks','max_abs','fp32_vectors','fp32_max_abs','layer_tensor_bytes'):
                    if item[k]!=correctness[ck][k]:raise ValueError('duplicated correctness annotation changed')
            else:correctness[ck]=item
            raw.append(r)
        provenance.append(dict(run=directory.name,path=str(f),sha256=sha(f),summary=m,
                               TRAIN_fixed_sha256=sha(directory/'TRAIN_fixed.json')))
    gpu_names=sorted({g for g,r in coverage})
    if len(gpu_names)!=2 or coverage!={(g,i) for g in gpu_names for i in range(3)}:raise ValueError('hardware replicate matrix incomplete')
    if len({json.dumps(p['summary']['source_sha256'],sort_keys=True) for p in provenance})!=1:raise ValueError('measurement sources changed')
    grouped=defaultdict(list)
    for r in raw:grouped[r['gpu'],r['base']['name'],r['layers'],r['sequence'],r['scratch_ceiling']].append(r)
    reduced=[]
    for key,rr in sorted(grouped.items()):
        if len(rr)!=3:raise ValueError('missing independent repeat')
        if len({json.dumps(r['counters'],sort_keys=True) for r in rr})!=1:raise ValueError('choices or cache behavior changed across replicas')
        reduced.append(dict(gpu=key[0],base=key[1],layers=key[2],sequence=key[3],scratch_ceiling=key[4],
            family=rr[0]['base']['family'],episode_ms={arm:statistics.median(r['median_ms'][arm] for r in rr) for arm in ARMS},
            replicate_ms={arm:[r['median_ms'][arm] for r in sorted(rr,key=lambda r:r['replicate'])] for arm in ARMS},
            counters=rr[0]['counters']))
    return raw,reduced,provenance,correctness


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('never overwrite a frozen result')
    raw,reduced,provenance,checks=load(a.root)
    results={}
    for gpu in sorted({r['gpu'] for r in reduced}):
        for budget in (134217728,536870912):
            for sequence in SEQUENCES:
                for repeat in REPEATS:
                    rr=[r for r in reduced if (r['gpu'],r['scratch_ceiling'],r['sequence'],r['layers'])==(gpu,budget,sequence,repeat)]
                    if len(rr)!=18:raise ValueError('18 fresh base geometries required')
                    family=[r['family'] for r in rr]
                    fixed=[r['episode_ms']['fixed_budget'] for r in rr];auto=[r['episode_ms']['auto'] for r in rr]
                    section=dict(gpu=gpu,scratch_MiB=budget//1024**2,sequence=sequence,layers_per_segment=repeat,segments=4,arms={})
                    for arm in ARMS:
                        cost=[r['episode_ms'][arm] for r in rr]
                        section['arms'][arm]={'vs_fixed':ratio_stats(cost,fixed,family),'vs_auto':ratio_stats(cost,auto,family)}
                    section['registered_primary_gate']=(repeat==36 and section['arms']['native_cycle']['vs_fixed']['CI95'][0]>1)
                    section['beats_both_controls_gate']=(section['registered_primary_gate'] and section['arms']['native_cycle']['vs_auto']['CI95'][0]>1)
                    key=f'{gpu}|{budget}|{sequence}|{repeat}';results[key]=section
                    if repeat==36:print('PRIMARY_REUSE',json.dumps(section),flush=True)
    a.out.mkdir(parents=True)
    correctness=dict(unique_base_budget_process_checks=len(checks),
        all_full_tensor_checks_including_auto=sum(c['full_tensor_checks'] for c in checks.values()),
        nonauto_full_tensor_checks=sum(sum((len(m)-1)*36 for m in c['selected_modes_checked']) for c in checks.values()),
        independent_selected_FP32_vectors=sum(c['fp32_vectors'] for c in checks.values()),
        max_abs_vs_auto=max(c['max_abs'] for c in checks.values()),max_abs_FP32=max(c['fp32_max_abs'] for c in checks.values()),
        peak_tensor_allocation_bytes=max(r['peak_cuda_allocated'] for r in raw),
        peak_cuda_allocator_reserved_bytes=max(r['peak_cuda_reserved'] for r in raw),
        max_distinct_layer_tensor_bytes=max(c['layer_tensor_bytes'] for c in checks.values()),
        cache_counter_checks=len(raw)*len(ARMS),
        caveat='All states and36 layer tensors checked per selected mode before timing; every timed output is NOT copied back and scored. Cache validity additionally checked by expected plan/hit/miss/eviction traces.')
    result=dict(kind='direct_matched_eager_attention_stack_reuse',raw_rows=len(raw),reduced_rows=len(reduced),
        policy_arms=len(ARMS),timed_layer_invocations=sum(r['layers']*4*4*2*len(ARMS) for r in raw),
        full_model=False,full_vllm=False,corpus_sha256=corpus_hash(),provenance=provenance,
        correctness=correctness,sections=results,
        uncertainty='Three process repeats reduced by cell median; nominal20k bootstrap over9 whole fresh geometry families, seed20261011; no multiplicity/population guarantee.',
        previous_unamortized_verdict='NOT_PROMOTED',
        baseline_note='TRAIN-only budget/head/reuse fixed uses old cycle+(r-1)*run proxy; all reported outcomes are newly measured direct wall times. Default auto also receives identical reuse.',
        timing_note='An episode has4 segments.36 means36 distinct layer tensors per segment:144 total calls. Unchanged geometry reuses one plan; alternating/eviction replans each segment. No TFLOPS-to-serving extrapolation.')
    save(a.out/'metrics.json',result);save(a.out/'per_cell.json',reduced)
    print('DIRECT_REUSE_RESULT_SHA256',sha(a.out/'metrics.json'))


if __name__=='__main__':main()
