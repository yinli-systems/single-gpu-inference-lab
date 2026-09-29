"""Match complete logical histories; do not infer hidden/KV equality from token history."""
from __future__ import annotations
import argparse,collections,hashlib,json,math
from pathlib import Path
from evidence_contract import file_hash,read_verified

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

def read_steps(root):
    files=sorted(root.glob('steps-*.jsonl'))
    if not files:raise ValueError('missing trace')
    rows=[];bindings={}
    for path in files:
        bindings[path.name]=file_hash(path)
        for line in path.read_text().splitlines():
            r=json.loads(line);n=r['batch_size']
            if n<1 or len(r['history_sha256'])!=n or len(r['seq_lens'])!=n:raise ValueError('trace dimensions')
            if any(len(r[k])!=n for k in ('top8_values','top8_ids','logit_sha256')):raise ValueError('trace logits dimension')
            if any(not math.isfinite(v) for xs in r['top8_values'] for v in xs):raise ValueError('nonfinite trace')
            identity={k:r[k] for k in ('forward_mode','seq_lens','input_ids','history_sha256')}
            if r['batch_signature']!=digest(identity):raise ValueError('trace signature mismatch')
            rows.append(r)
    return rows,bindings

def compare(base,candidate):
    maps=[];unknown=[]
    for rows in (base,candidate):
        index=collections.defaultdict(list);missing=0
        for r in rows:
            if not r['all_histories_complete'] or any(h is None for h in r['history_sha256']):missing+=1;continue
            index[r['batch_signature']].append(r)
        maps.append(index);unknown.append(missing)
    keys=sorted(set(maps[0])&set(maps[1]));same=0;different=[];ambiguous=[];set_relations=[]
    for key in keys:
        a,b=maps[0][key],maps[1][key]
        aa={tuple(x['logit_sha256']) for x in a};bb={tuple(x['logit_sha256']) for x in b}
        set_relations.append(dict(signature=key,shared_variants=len(aa&bb),pristine_only=len(aa-bb),cap_only=len(bb-aa)))
        if len(aa)==len(bb)==1 and aa==bb:same+=1
        elif len(aa)>1 or len(bb)>1:ambiguous.append(dict(signature=key,pristine_variants=len(aa),cap_variants=len(bb),seq_lens=a[0]['seq_lens']))
        else:
            different.append(dict(signature=key,seq_lens=a[0]['seq_lens'],history_sha256=a[0]['history_sha256'],
                pristine_top8=a[0]['top8_values'],cap_top8=b[0]['top8_values'],
                pristine_ids=a[0]['top8_ids'],cap_ids=b[0]['top8_ids'],
                pristine_sampled=a[0]['sampled_ids'],cap_sampled=b[0]['sampled_ids']))
    return dict(matched_unique_batch_signatures=len(keys),exact_logit_signature_matches=same,
        cross_arm_logit_differences=different,within_context_variation=ambiguous,
        cross_arm_variant_sets=set_relations,
        disjoint_cross_arm_signatures=sum(x['shared_variants']==0 for x in set_relations),
        signatures_with_cap_only_variants=sum(x['cap_only']>0 for x in set_relations),
        unknown_history_steps=unknown,total_steps=[len(base),len(candidate)],
        unmatched_signatures=[len(set(maps[i])-set(keys)) for i in range(2)],
        physical_KV_or_hidden_state_equality_established=False,first_operator_attributed=False,
        historical_failure_resolved=False,performance_promoted=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('preserve trace audit')
    for mode in ('pristine-0','cap-0'):read_verified(a.job_root/mode/'environment.json')
    b,bh=read_steps(a.job_root/'pristine-0/trace');c,ch=read_steps(a.job_root/'cap-0/trace');report=compare(b,c)
    report['trace_files_sha256_at_collection']={'pristine':bh,'cap':ch}
    a.out.mkdir(parents=True);(a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cross_arm_logit_differences','within_context_variation','cross_arm_variant_sets','trace_files_sha256_at_collection')},indent=2))
    print('different',len(report['cross_arm_logit_differences']),'ambiguous',len(report['within_context_variation']))
