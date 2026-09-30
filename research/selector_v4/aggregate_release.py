"""Recompute a whole-GPU v4 verdict from every shard's raw evaluation evidence."""
from __future__ import annotations
from pathlib import Path
import argparse,collections,json
from analyze_evaluation import collect_shard,summarize
from manifest import load


def run(a):
    all_cells=[];all_draws=[];all_cap=[];provenance=None;complete=[];decision_hashes=[];exact=True
    for shard in range(a.shards):
        decisions=Path(a.root)/'decisions'/a.gpu/f's{shard}.json'
        data=collect_shard(root=a.root,stage=a.stage,gpu=a.gpu,shard=shard,shards=a.shards,decisions=decisions)
        env=data['environment'];current={k:env[k] for k in ('gpu_name','gpu_uuid','driver','num_sms','case_hash','release_hash','source_archive_sha256','overlay_sha256','measurement_revision')}
        if provenance is None:provenance=current
        elif provenance!=current:raise RuntimeError('cross-shard provenance drift')
        all_cells.extend(data['cells']);all_draws.extend(data['chosen_draws']);all_cap.extend(data['cap_draws']);complete.extend(data['run_complete_sha256']);decision_hashes.append(data['decisions_sha256']);exact&=data['all_exact']
    manifest=load();expected_cases=[c['id'] for c in manifest['cases'] if c['family']==a.stage];expected={(case,dt,layout,split,ex) for case in expected_cases for dt in manifest['dtypes'] for layout in manifest['layouts'] for split in manifest['splits'] for ex in manifest['execution_modes']}
    observed={(c['key'][0],c['key'][1],c['key'][2],c['key'][3],c['execution_mode']) for c in all_cells}
    if observed!=expected or len(observed)!=len(all_cells):raise RuntimeError('global cell coverage')
    metrics=summarize(all_cells,all_draws,all_cap,require_cap=True);metrics['requirements']['numerical_exact']=exact;metrics['pass']=metrics['pass'] and exact
    case_map={c['id']:c for c in manifest['cases']};coverage=collections.Counter();regime=collections.defaultdict(list)
    for cell in all_cells:
        coverage[(cell['tactic'],cell['execution_mode'])]+=1;regime[(case_map[cell['key'][0]]['regime'],cell['tactic'])].append(cell['chosen_ratio'])
    result={'schema':1,'stage':a.stage,'gpu':a.gpu,'shards':a.shards,'provenance':provenance,'decision_sha256':decision_hashes,'run_complete_sha256':complete,'cell_count':len(all_cells),'coverage':{'|'.join(k):v for k,v in sorted(coverage.items())},'regime_point_worst':{'|'.join(k):min(v) for k,v in sorted(regime.items())},'cells':all_cells,**metrics,'default_promotion':False,'serving_promotion':False,'historical_token_divergence_resolved':False}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('pass','cell_count','coverage','requirements','chosen_worst','chosen_joint_lcb','max_regret','p95_regret','cap_selected','cap_geomean','cap_ci95')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['canary','release'],required=True);p.add_argument('--gpu',required=True);p.add_argument('--shards',type=int,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
