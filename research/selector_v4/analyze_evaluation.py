"""Evaluate frozen v4 choices on processes not used for calibration."""
from __future__ import annotations
from pathlib import Path
import argparse,json,math
from autotune import hierarchical_bootstrap,quantile,geometric_mean,canonical_hash,CONTROL_TOLERANCE
from evidence import load_run,sha


def gm(xs):return math.exp(sum(math.log(x) for x in xs)/len(xs))

def block_ratio(rows,group,rep,block,execution):
    a=[r['wall_us'] for r in rows if r['comparison_group']==group and r['rep']==rep and r['block']==block and r['execution_mode']==execution and r['role']=='A']
    b=[r['wall_us'] for r in rows if r['comparison_group']==group and r['rep']==rep and r['block']==block and r['execution_mode']==execution and r['role']=='B']
    if len(a)!=2 or len(b)!=2:raise RuntimeError('ABBA matrix')
    return gm(a)/gm(b)


def collect_shard(*,root:Path,stage:str,gpu:str,shard:int,shards:int,decisions:Path):
    root=Path(root);decisions=Path(decisions);decision=json.loads(decisions.read_text())
    if decision.get('schema')!=1 or decision.get('default')!='native':raise RuntimeError('decision schema')
    runs=[]
    for rep in range(3):
        paths=list((root/'runs').glob(f'{stage}-{gpu}-s{shard}-r{rep}-evaluation-*'))
        if len(paths)!=1:raise RuntimeError('run discovery')
        env,rows,quals,complete=load_run(root,paths[0],mode='evaluation',stage=stage,gpu=gpu,rep=rep,shard=shard,shards=shards,decisions=decisions)
        runs.append((env,rows,quals,complete,paths[0]))
    env0=runs[0][0];prov=decision.get('provenance',{})
    for key in ('case_hash','release_hash','source_archive_sha256','overlay_sha256'):
        if prov.get(key)!=env0[key]:raise RuntimeError('decision provenance '+key)
    if (prov.get('stage'),prov.get('gpu'),prov.get('shard'),prov.get('shards'))!=(stage,gpu,shard,shards):raise RuntimeError('decision scope')
    for env,_,_,_,_ in runs:
        for key in ('gpu_uuid','driver','source_archive_sha256','overlay_sha256','case_hash','release_hash'):
            if env[key]!=env0[key]:raise RuntimeError('environment drift '+key)
    qmaps=[{(q['case'],q['dtype'],q['layout'],q['split']):q for q in qs} for _,_,qs,_,_ in runs]
    cells=[];chosen_draws=[];cap_draws=[]
    for key in qmaps[0]:
        for ex in ('eager_full_call','graph1_replay','graph16_replay'):
            identity=qmaps[0][key]['identities'][ex];h=identity['sha256'];record=decision['records'].get(h);tactic='native' if record is None else record['tactic']
            if h!=canonical_hash(identity['payload']):raise RuntimeError('identity')
            if any(m[key]['chosen'][ex]!=tactic for m in qmaps):raise RuntimeError('choice drift')
            if tactic=='cap' and not all(m[key]['cap_supported'] for m in qmaps):raise RuntimeError('unsupported cap choice')
            chosen=[];oracle=[];null=[]
            for rep,(env,rows,_,_,_) in enumerate(runs):
                rr=[r for r in rows if (r['case'],r['dtype'],r['layout'],r['split'])==key]
                chosen.append([block_ratio(rr,'chosen',rep,b,ex) for b in range(8)])
                oracle.append([block_ratio(rr,'oracle',rep,b,ex) for b in range(8)])
                null.append([block_ratio(rr,'null',rep,b,ex) for b in range(8)])
            cd=hierarchical_bootstrap(chosen,seed=int(h[:8],16));nd=hierarchical_bootstrap(null,seed=int(h[8:16],16));od=hierarchical_bootstrap(oracle,seed=int(h[16:24],16))
            chosen_ratio=geometric_mean([x for row in chosen for x in row]);oracle_ratio=geometric_mean([x for row in oracle for x in row]);null_ci=(quantile(nd,.05),quantile(nd,.95));control=null_ci[0]>=1/(1+CONTROL_TOLERANCE) and null_ci[1]<=1+CONTROL_TOLERANCE
            regret=max(1.0,1/oracle_ratio) if tactic=='cap' else max(1.0,oracle_ratio)
            cell={'key':key,'execution_mode':ex,'identity_sha256':h,'tactic':tactic,'candidate_pool':qmaps[0][key]['candidate_pool']['eligible'],'chosen_ratio':chosen_ratio,'chosen_ci95':[quantile(cd,.025),quantile(cd,.975)],'oracle_ratio':oracle_ratio,'oracle_ci95':[quantile(od,.025),quantile(od,.975)],'null_ci90':null_ci,'control_resolves':control,'regret':regret}
            cells.append(cell);chosen_draws.append(cd)
            if tactic=='cap':cap_draws.append(cd)
    return {'cells':cells,'chosen_draws':chosen_draws,'cap_draws':cap_draws,'environment':env0,'decisions_sha256':sha(decisions),'run_complete_sha256':[sha(path/'complete.json') for *_,path in runs],'all_exact':all(q['exact'] for m in qmaps for q in m.values())}


def summarize(cells,chosen_draws,cap_draws,*,require_cap=True):
    if not cells or len(cells)!=len(chosen_draws):raise ValueError('cells/draws')
    point_worst=min(c['chosen_ratio'] for c in cells);joint=[min(row[i] for row in chosen_draws) for i in range(len(chosen_draws[0]))];joint_lcb=quantile(joint,.025)
    regrets=sorted(c['regret'] for c in cells);p95=regrets[int(.95*(len(regrets)-1))];max_regret=max(regrets)
    cap_ci=None;cap_ratio=None
    if cap_draws:
        aggregate=[math.exp(sum(math.log(row[i]) for row in cap_draws)/len(cap_draws)) for i in range(len(cap_draws[0]))]
        cap_ci=[quantile(aggregate,.025),quantile(aggregate,.975)];cap_ratio=geometric_mean([c['chosen_ratio'] for c in cells if c['tactic']=='cap'])
    requirements={'controls_resolve':all(c['control_resolves'] for c in cells),'chosen_worst_at_least_0_99':point_worst>=.99,'chosen_joint_lcb_at_least_0_99':joint_lcb>=.99,'max_regret_at_most_1_02':max_regret<=1.02,'p95_regret_at_most_1_01':p95<=1.01,'cap_selected_nonempty':bool(cap_draws) or not require_cap,'cap_aggregate_lcb_above_1':(bool(cap_ci) and cap_ci[0]>1.0) or not require_cap}
    return {'pass':all(requirements.values()),'requirements':requirements,'chosen_worst':point_worst,'chosen_joint_lcb':joint_lcb,'max_regret':max_regret,'p95_regret':p95,'cap_selected':len(cap_draws),'cap_geomean':cap_ratio,'cap_ci95':cap_ci}


def run(a):
    collected=collect_shard(root=a.root,stage=a.stage,gpu=a.gpu,shard=a.shard,shards=a.shards,decisions=a.decisions)
    metrics=summarize(collected['cells'],collected['chosen_draws'],collected['cap_draws'],require_cap=(a.stage=='canary'))
    env=collected['environment'];requirements=dict(metrics['requirements'],numerical_exact=collected['all_exact']);metrics['requirements']=requirements;metrics['pass']=metrics['pass'] and collected['all_exact']
    result={'schema':1,'stage':a.stage,'gpu':a.gpu,'shard':a.shard,'shards':a.shards,'case_hash':env['case_hash'],'release_hash':env['release_hash'],'source_archive_sha256':env['source_archive_sha256'],'overlay_sha256':env['overlay_sha256'],'decisions_sha256':collected['decisions_sha256'],'run_complete_sha256':collected['run_complete_sha256'],'cells':collected['cells'],**metrics,'default_promotion':False,'serving_promotion':False}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('pass','requirements','chosen_worst','chosen_joint_lcb','max_regret','p95_regret','cap_selected','cap_geomean','cap_ci95')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['canary','release'],required=True);p.add_argument('--gpu',required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,required=True);p.add_argument('--decisions',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
