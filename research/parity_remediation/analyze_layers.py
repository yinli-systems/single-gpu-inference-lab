"""Compare actual saved operator/KV inputs; preserve unmatched/variable histories."""
from __future__ import annotations
import argparse,collections,hashlib,json,re
from pathlib import Path

ORDER=('input_layernorm','self_attn.qkv_proj','self_attn.attn','self_attn.o_proj',
       'post_attention_layernorm','mlp.gate_up_proj','mlp.down_proj')

def position(name):
    m=re.search(r'\.layers\.(\d+)\.(.+)\|(input|output)\|',name)
    if not m:raise ValueError('unrecognized observed operator '+name)
    op=m.group(2)
    if op not in ORDER:raise ValueError('unknown observed operator')
    return (int(m.group(1)),ORDER.index(op),0 if m.group(3)=='input' else 1,name)

def signature(meta):
    return (meta['history_sha256'],meta['batch_signature'],meta['graph_used'],meta['graph_rows'],meta['row'])

def pair(a,b):
    if signature(a)!=signature(b):raise ValueError('unmatched physical execution coordinate')
    if set(a['shadow_sha256'])!=set(b['shadow_sha256']) or set(a['cache_sha256'])!=set(b['cache_sha256']):raise ValueError('operator coverage mismatch')
    changed=[k for k in sorted(a['shadow_sha256'],key=position) if a['shadow_sha256'][k]!=b['shadow_sha256'][k]]
    cache=[k for k in a['cache_sha256'] if a['cache_sha256'][k]!=b['cache_sha256'][k]]
    first=changed[0] if changed else None
    return dict(logits_equal=a['logits_sha256']==b['logits_sha256'],first_observed_shadow_difference=first,
                changed_shadow_count=len(changed),changed_shadow_keys=changed,
                changed_KV_count=len(cache),changed_KV_keys=cache,
                all_observed_inputs_equal=not changed and not cache,
                hidden_state_and_KV_compared=True,old_uninstrumented_failure_attributed=False)

def analyze(root,out):
    import torch
    from layer_observer import tensor_hash
    from evidence_contract import read_verified,file_hash
    if out.exists():raise FileExistsError('preserve results')
    arms={};files={};targets=set();counts={}
    for mode in ('pristine-0','cap-0'):
        read_verified(root/mode/'environment.json')
        rows=[]
        for path in sorted((root/mode/'trace/layers').glob('step-*.json')):
            meta=json.loads(path.read_text());data=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=True)
            if data['metadata']!=meta:raise ValueError('metadata/tensor file mismatch')
            if {k:tensor_hash(v) for k,v in data['shadows'].items()}!=meta['shadow_sha256']:raise ValueError('shadow corruption')
            if {k:tensor_hash(v) for k,v in data['cache'].items()}!=meta['cache_sha256']:raise ValueError('KV corruption')
            if tensor_hash(data['logits'])!=meta['logits_sha256']:raise ValueError('logit corruption')
            ids={position(k)[0] for k in meta['shadow_sha256']}
            if ids!=set(range(36)) or len(meta['cache_sha256'])!=72:raise ValueError('missing layer evidence')
            rel=str(path.relative_to(root));files[rel]=file_hash(path)
            files[str(path.with_suffix('.pt').relative_to(root))]=file_hash(path.with_suffix('.pt'))
            targets.add(meta['target']['request_id']);rows.append((path.name,meta));del data
        if not rows:raise ValueError('no historical-prefix decode snapshots for '+mode)
        arms[mode]=rows;counts[mode]=len(rows)
    groups={mode:collections.defaultdict(list) for mode in arms}
    for mode,rows in arms.items():
        for name,meta in rows:groups[mode][signature(meta)].append((name,meta))
    keys=set(groups['pristine-0'])&set(groups['cap-0']);comparisons=[]
    for key in sorted(keys):
        for an,a in groups['pristine-0'][key]:
            for bn,b in groups['cap-0'][key]:
                comparisons.append(dict(pristine=an,cap=bn,target=a['target'],**pair(a,b)))
    report=dict(snapshot_counts=counts,targets_observed=sorted(targets),matched_execution_coordinates=len(keys),
        comparisons=comparisons,pair_count=len(comparisons),file_hashes=files,
        unmatched_coordinates={m:len(set(g)-keys) for m,g in groups.items()},
        instrumentation_changes_execution=True,performance_claim=False,old_failure_resolved=False)
    out.mkdir(parents=True);(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('file_hashes','comparisons')},indent=2))
    for c in comparisons:print(json.dumps({k:v for k,v in c.items() if k not in ('changed_shadow_keys','changed_KV_keys')},sort_keys=True))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();analyze(a.job_root,a.out)
