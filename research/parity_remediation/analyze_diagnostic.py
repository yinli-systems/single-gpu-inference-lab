"""Diagnose repeatability without converting diagnostic samples to performance claims."""
import argparse,ast,collections,json,math
from pathlib import Path
from evidence_contract import compare_batches,first_difference,read_verified,file_hash

def resolved_options(path):
    for line in path.read_text(errors='replace').splitlines():
        if 'server_args=' in line:
            try:
                opts=ast.literal_eval(line.split('server_args=',1)[1])
                return {k:opts.get(k) for k in ('random_seed','enable_deterministic_inference','sampling_backend','disable_overlap_schedule','cuda_graph_config','page_size','disable_radix_cache')}
            except (ValueError,SyntaxError):return {'parse_failed':True}
    return {'missing':True}

def inspect(root):
    report=dict(campaign=str(root),runs=[],complete=False,diagnostic_only=True,performance_claim=False,serving_promoted=False)
    for job in sorted((root/'runs').iterdir()):
        if not job.is_dir():continue
        hardware=(job/'hardware.csv').read_text() if (job/'hardware.csv').exists() else None
        modes={};failed=[];missing=[];options={};snapshots={}
        for mode in ('pristine','cap'):
            for rep in (0,1):
                key=f'{mode}-{rep}';path=job/key
                if (path/'failure.json').exists():failed.append(dict(run=key,failure=json.loads((path/'failure.json').read_text())))
                if not (path/'complete.json').exists():missing.append(key);continue
                env=read_verified(path/'environment.json');options[key]=resolved_options(path/'server.log')
                modes[key]={}
                for work in ('prefill','decode','mixed'):
                    for b in range(3):modes[key][f'{work}-b{b}']=read_verified(path/f'{work}-b{b}.json')
                snapshots[key]={}
                for rid in ('decode-11','decode-13'):
                    for trial in (0,1):
                        for kind in ('isolated','prefix'):
                            name=f'{kind}-{rid}-{trial}.json';snapshots[key][name]=read_verified(path/name)
        comparisons=[]
        for rep in (0,1):
            bk,ck=f'pristine-{rep}',f'cap-{rep}'
            if bk in modes and ck in modes:
                for name,b in modes[bk].items():
                    comparisons.append(dict(rep=rep,cell=name,**compare_batches(b,modes[ck][name])))
        variation={}
        for mode in ('pristine','cap'):
            seqs=collections.defaultdict(set)
            for key,batches in modes.items():
                if key.startswith(mode+'-'):
                    for name,batch in batches.items():
                        for r in batch['requests']:seqs[(name.split('-b')[0],r['id'])].add(tuple(r['tokens']))
            variation[mode]=[dict(workload=k[0],id=k[1],unique_outputs=len(v)) for k,v in seqs.items() if len(v)>1]
        probes=[]
        for rep in (0,1):
            bk,ck=f'pristine-{rep}',f'cap-{rep}'
            if bk not in snapshots or ck not in snapshots:continue
            for name,b in snapshots[bk].items():
                c=snapshots[ck][name]
                if name.startswith('isolated'):
                    probes.append(dict(rep=rep,file=name,kind='single-request-autoregressive',**compare_batches(b,c)))
                else:
                    if b['prefix_sha256']!=c['prefix_sha256']:raise ValueError('teacher-forced prefix mismatch')
                    x=b['response'];y=c['response']
                    # Keep the complete top-logprob readout, not an invented logit-equality check.
                    probes.append(dict(rep=rep,file=name,kind='teacher-forced-prefill',historical_choices=b['historical_choices'],
                        output_equal=x.get('output_ids')==y.get('output_ids'),
                        pristine=x,cap=y,identical_prefix=True))
        report['runs'].append(dict(job=job.name,hardware=hardware,completed_modes=list(modes),missing=missing,
            failures=failed,options=options,within_arm_variation=variation,
            paired_candidate_requests=sum(c['candidate_requests'] for c in comparisons),
            paired_mismatches=sum(len(c['mismatches']) for c in comparisons),comparisons=comparisons,probes=probes,
            old_failure_resolved=False))
    report['complete']=bool(report['runs']) and all(not r['missing'] and not r['failures'] for r in report['runs'])
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('preserve audit')
    d=inspect(a.root);a.out.mkdir(parents=True);(a.out/'summary.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(complete=d['complete'],runs=[{k:r[k] for k in ('job','completed_modes','missing','failures','within_arm_variation','paired_candidate_requests','paired_mismatches','old_failure_resolved')} for r in d['runs']]),indent=2))
