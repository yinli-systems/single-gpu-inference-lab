"""Freeze deployment-mode tactic choices from calibration processes only."""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json,math,csv
from autotune import decide,canonical_hash,SCHEMA
from evidence import load_run,sha

def gm(xs):return math.exp(sum(math.log(x) for x in xs)/len(xs))
def clock_stable(path):
 with Path(path).open() as stream:
  rows=list(csv.DictReader(stream))
 values=[]
 for row in rows:
  raw=next((v for k,v in row.items() if k and 'clocks.current.sm' in k),None)
  util=next((v for k,v in row.items() if k and 'utilization.gpu' in k),None)
  try:
   utilization=float(util.split()[0]) if util else 0.0
   clock=float(raw.split()[0]) if raw else 0.0
  except (ValueError,AttributeError):
   continue
  if utilization>=90 and clock>0:values.append(clock)
 if len(values)<10:return False,{'active_samples':len(values),'reason':'insufficient-active-telemetry'}
 ordered=sorted(values);lo=ordered[max(0,int(.05*len(ordered))-1)];hi=ordered[min(len(ordered)-1,int(.95*len(ordered)))]
 stable=hi/lo<=1.05
 return stable,{'active_samples':len(values),'p05':lo,'p95':hi,'ratio':hi/lo}
def block_ratio(rows,group,rep,block,execution):
 a=[r['wall_us'] for r in rows if r['comparison_group']==group and r['rep']==rep and r['block']==block and r['execution_mode']==execution and r['role']=='A']
 b=[r['wall_us'] for r in rows if r['comparison_group']==group and r['rep']==rep and r['block']==block and r['execution_mode']==execution and r['role']=='B']
 if len(a)!=2 or len(b)!=2:raise RuntimeError('ABBA matrix')
 return gm(a)/gm(b)
def run(a):
 root=Path(a.root);runs=[]
 for rep in range(3):
  paths=list((root/'runs').glob(f'{a.stage}-{a.gpu}-s{a.shard}-r{rep}-calibration-*'))
  if len(paths)!=1:raise RuntimeError('run discovery')
  env,rows,quals,complete=load_run(root,paths[0],mode='calibration',stage=a.stage,gpu=a.gpu,rep=rep,shard=a.shard,shards=a.shards)
  runs.append((env,rows,quals,complete,paths[0]))
 env0=runs[0][0]
 for env,_,_,_,_ in runs:
  for key in ('gpu_uuid','driver','source_archive_sha256','overlay_sha256','case_hash','release_hash'):
   if env[key]!=env0[key]:raise RuntimeError('environment drift '+key)
 stable,clock=clock_stable(a.telemetry)
 qmaps=[{(q['case'],q['dtype'],q['layout'],q['split']):q for q in qs} for _,_,qs,_,_ in runs]
 records={};summary=[]
 for key in qmaps[0]:
  for ex in ('eager_full_call','graph1_replay','graph16_replay'):
   ratios=[];controls=[]
   for rep,(env,rows,_,_,_) in enumerate(runs):
    ratios.append([block_ratio([r for r in rows if (r['case'],r['dtype'],r['layout'],r['split'])==key],'candidate',rep,b,ex) for b in range(8)])
    controls.append([block_ratio([r for r in rows if (r['case'],r['dtype'],r['layout'],r['split'])==key],'null',rep,b,ex) for b in range(8)])
   q=qmaps[0][key];identity=q['identities'][ex]['payload'];identity_hash=q['identities'][ex]['sha256']
   if identity_hash!=canonical_hash(identity):raise RuntimeError('identity')
   pool=bool(q['candidate_pool']['eligible'])
   evidence=decide(native_over_cap=ratios,null_controls=controls,exact_outputs=all(m[key]['exact'] for m in qmaps),clock_stable=stable,candidate_pool=pool,seed=int(identity_hash[:8],16))
   record={'identity':identity,'tactic':evidence.tactic,'evidence':evidence.__dict__}
   records[identity_hash]=record;summary.append({'key':key,'execution_mode':ex,'candidate_pool':pool,'tactic':evidence.tactic,'ratio':evidence.ratio,'ci95':evidence.ci95,'control_ci90':evidence.control_ci90,'reason':evidence.reason})
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 provenance={'stage':a.stage,'gpu':a.gpu,'shard':a.shard,'shards':a.shards,'case_hash':env0['case_hash'],'release_hash':env0['release_hash'],'source_archive_sha256':env0['source_archive_sha256'],'overlay_sha256':env0['overlay_sha256'],'telemetry_sha256':sha(a.telemetry),'run_complete_sha256':[sha(path/'complete.json') for *_,path in runs]}
 out.write_text(json.dumps({'schema':SCHEMA,'records':records,'calibration':summary,'clock':clock,'clock_stable':stable,'provenance':provenance,'default':'native','release_evidence':False},indent=2)+'\n')
 print(json.dumps({'records':len(records),'cap':sum(x['tactic']=='cap' for x in summary),'native':sum(x['tactic']=='native' for x in summary),'clock_stable':stable},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['canary','release'],required=True);p.add_argument('--gpu',required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,required=True);p.add_argument('--telemetry',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
