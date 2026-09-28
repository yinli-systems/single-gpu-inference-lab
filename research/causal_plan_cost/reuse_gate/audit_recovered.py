"""Reproduce old-data recovery and a LABELLED additive break-even diagnostic.

No fitting, kernel launch, synthetic timings or fresh-test selection occurs.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from contracts import profitable_interval


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def verified_runs(root):
    allrows=[];receipts=[]
    for f in sorted((root/'runs').glob('*/measurements.jsonl')):
        summary=json.loads(f.with_name('summary.json').read_text())
        if sha(f)!=summary['measurements_sha256']:raise ValueError('modified raw measurements')
        rows=[json.loads(x) for x in f.read_text().splitlines()]
        if any(r.get('status','complete')!='complete' for r in rows):raise ValueError('failed record')
        allrows.extend(rows);receipts.append(dict(path=str(f),sha256=sha(f),records=len(rows),summary=summary))
    return allrows,receipts


def wave_audit(root):
    rows,receipts=verified_runs(root)
    if len(rows)!=864 or len(receipts)!=6:raise ValueError('full recovered wave coverage required')
    index={}
    for r in rows:
        key=(r['gpu'],r['shape']['name'],r['hq'],r['policy'],r['replicate'])
        if key in index:raise ValueError('duplicate wave record')
        index[key]=r
    result=[]
    for Q in (960,1024,1088,1280,1344,1408,2656,2720,2784):
        for hq in (16,32):
            cell=dict(Q=Q,hq=hq,by_gpu={})
            for gpu in sorted({r['gpu'] for r in rows}):
                modes={}
                for mode in ('auto','none','s512','s1024'):
                    ratios=[]
                    for rep in range(3):
                        A=index[gpu,f'wave-Q{Q}-A',hq,mode,rep]
                        B=index[gpu,f'wave-Q{Q}-B',hq,mode,rep]
                        qa,ka=A['shape']['q'],A['shape']['k'];qb,kb=B['shape']['q'],B['shape']['k']
                        work=lambda q,k:sum(a*b+a*(a+1)//2 for a,b in zip(q,k))
                        if work(qa,ka)!=work(qb,kb) or sum(qa)!=sum(qb) or sum(ka)!=sum(kb):raise ValueError('unequal-work control')
                        if sorted(a+b for a,b in zip(qa,ka))!=sorted(a+b for a,b in zip(qb,kb)):raise ValueError('total-KV marginals differ')
                        ratios.append(A['median_ms']/B['median_ms'])
                    modes[mode]=dict(ratios=ratios,median=statistics.median(ratios),range=[min(ratios),max(ratios)])
                cell['by_gpu'][gpu]=modes
            if Q in (1344,2720):
                a=cell['by_gpu']['NVIDIA GeForce RTX 4090']['none']['median']
                b=cell['by_gpu']['NVIDIA GeForce RTX 5090']['none']['median']
                cell['preregistered_crossover_pass']=bool(b>1.1 and b>a)
            result.append(cell)
    return dict(kind='recovered_preregistered_crossover',records=864,receipts=receipts,table=result,
                fresh_targets=4,passed=sum(c.get('preregistered_crossover_pass',False) for c in result),
                uncertainty='Three process/seed realizations; ranges are not confidence intervals. All neighbors retained.',
                claim='Supports a specific hardware-boundary mechanism; not invention of wave quantization, complete causality, or implemented speedup.')


def break_even_audit(campaign):
    audit=campaign/'independent-audit-20260928T0000Z'
    native=json.loads((audit/'native_validation.json').read_text())
    selection=json.loads((audit/'model_selection.json').read_text())
    rows,receipts=verified_runs(campaign)
    groups=defaultdict(list)
    # verified_runs includes old canaries; only the final full-stage rows enter this diagnostic.
    groups.clear()
    for receipt in receipts:
        if receipt['summary'].get('stage')!='full':continue
        for line in Path(receipt['path']).read_text().splitlines():
            r=json.loads(line)
            if r['shape']['split']=='test':groups[(r['gpu'],r['shape']['name'],r['hq'],r['policy'])].append(r)
    lookup={}
    for key,rs in groups.items():
        if len(rs)!=3:raise ValueError('missing replicate')
        lookup[key]={v:statistics.median(r[v] for r in rs) for v in ('median_ms','median_cycle_ms')}
    output=[]
    for r in native['native_timing_rows']:
        gpu,target,kind=r['key'].split('|')
        chosen=lookup[gpu,r['shape'],r['hq'],r['policy']]
        fixed_policy=selection['fixed'][gpu+'|'+target]['per_head'][str(r['hq'])]
        fixed=lookup[gpu,r['shape'],r['hq'],fixed_policy]
        saving=fixed['median_ms']-chosen['median_ms']
        setup_delta=(chosen['median_cycle_ms']-chosen['median_ms'])-(fixed['median_cycle_ms']-fixed['median_ms'])
        C=r['native_median_ms']; interval=profitable_interval(C+setup_delta,saving)
        output.append(dict(gpu=gpu,target=target,shape=r['shape'],family=r['family'],hq=r['hq'],
            candidate=r['policy'],fixed=fixed_policy,CPU_selector_ms=C,setup_delta_proxy_ms=setup_delta,
            per_call_GPU_saving_ms=saving,profitable_integer_interval=interval))
    summary={}
    for key in sorted({(r['gpu'],r['target']) for r in output}):
        rr=[r for r in output if (r['gpu'],r['target'])==key]
        finite_first=[r['profitable_integer_interval'][0] for r in rr if r['profitable_integer_interval'] is not None and r['profitable_integer_interval'][1] is None]
        summaries=dict(cells=len(rr),eventual_break_even_cells=len(finite_first),never_profitable=sum(r['profitable_integer_interval'] is None for r in rr),
                       finite_profitable_windows=sum(r['profitable_integer_interval'] is not None and r['profitable_integer_interval'][1] is not None for r in rr),
                       median_first_reuse_among_eventual=statistics.median(finite_first) if finite_first else None)
        summaries['positive_at_reuse']={str(n):sum(r['profitable_integer_interval'] is not None and r['profitable_integer_interval'][0]<=n and (r['profitable_integer_interval'][1] is None or n<=r['profitable_integer_interval'][1]) for r in rr) for n in (1,2,4,8,16,36)}
        summary['|'.join(key)]=summaries
    return dict(kind='old_measurement_additive_diagnostic_NOT_direct_reuse_result',summary=summary,cells=output,
        warning='CPU cost measured on ln01 with previous ctypes path. cycle-minus-CUDA is an approximate setup difference, not an isolated planning measurement. No new cache/guard cost, changed memory ceiling, per-layer data traffic, clock variability or full-model work is included. No deployment gate can pass from these estimates.',
        formula='r*(U_baseline-U_candidate)>CPU_selector+(P_candidate-P_baseline)',
        provenance={str(audit/'native_validation.json'):sha(audit/'native_validation.json'),str(audit/'model_selection.json'):sha(audit/'model_selection.json')})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    wave=wave_audit(a.root/'campaigns/wave-threshold-v1')
    save(a.out/'wave_recovered.json',wave)
    calc=break_even_audit(a.root/'campaigns/resume-20260928T2348Z')
    save(a.out/'break_even_diagnostic.json',calc)
    print('WAVE_TARGETS',wave['passed'],'/',wave['fresh_targets'])
    print('BREAK_EVEN_DIAGNOSTIC',json.dumps(calc['summary']))


if __name__=='__main__':main()
