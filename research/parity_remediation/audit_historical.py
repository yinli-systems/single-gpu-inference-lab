"""Independent frozen-archive audit; no extra GPU trials or filtering."""
import argparse,collections,hashlib,json,math
from pathlib import Path
from evidence_contract import compare_batches,read_verified,file_hash

def run(root,out):
    if out.exists():raise FileExistsError('preserve evidence')
    modes=('pristine','off','cap');all_records={m:[] for m in modes};comparisons=[];hashes={}
    for job in (1639805,1639806,1639807):
        for work in ('prefill','decode','mixed'):
            for b in range(3):
                records={}
                for mode in modes:
                    path=root/f'serving-runs/{job}/{mode}/{work}-b{b}.json'
                    records[mode]=read_verified(path);hashes[str(path.relative_to(root))]=file_hash(path)
                    all_records[mode].append((work,job,b,records[mode]))
                for mode in modes[1:]:
                    comparisons.append(dict(job=job,workload=work,block=b,mode=mode,
                        **compare_batches(records['pristine'],records[mode])))
    per_arm={};variation={}
    for mode,records in all_records.items():
        per_arm[mode]=sum(len(x[3]['requests']) for x in records)
        sets=collections.defaultdict(set)
        for work,j,b,batch in records:
            for r in batch['requests']:sets[(work,r['id'])].add(tuple(r['tokens']))
        variation[mode]=[dict(workload=k[0],id=k[1],unique_outputs=len(v)) for k,v in sets.items() if len(v)>1]
    report=dict(successful_requests_by_arm=per_arm,within_arm_variation=variation,
        cap_comparisons=sum(x['candidate_requests'] for x in comparisons if x['mode']=='cap'),
        cap_mismatch_requests=sum(len(x['mismatches']) for x in comparisons if x['mode']=='cap'),
        compared_blocks=3,all_blocks_included=True,comparisons=comparisons,source_files=hashes,
        units=dict(output_tokens_per_second='tokens/second',elapsed='seconds',TTFT='seconds',TPOT='seconds'),
        observations_not_proof_of_root_cause=True,serving_promoted=False)
    out.mkdir(parents=True);(out/'historical-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('comparisons','source_files')},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.root,a.out)
