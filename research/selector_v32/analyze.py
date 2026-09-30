from __future__ import annotations
import argparse,hashlib,itertools,json,math
from pathlib import Path
import numpy as np
from manifest import load,digest
from release_gate import evaluate

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(x,msg):
 if not x:raise RuntimeError(msg)
def weights(shard,n=10000):
 rng=np.random.default_rng(332211+shard*1009);w=np.zeros((n,24))
 for d in range(n):
  for p in rng.integers(0,3,3):
   for b in rng.integers(0,8,8):w[d,p*8+b]+=1
 return w/24

def qci(draw,q=(.025,.975)):return [float(x) for x in np.exp(np.quantile(draw,q))]
def load_run(path,mode,rep,stage,cases,blocks):
 e=json.loads((path/'environment.json').read_text());c=json.loads((path/'complete.json').read_text());rows=json.loads((path/'measurements.json').read_text());quals=json.loads((path/'qualification.json').read_text())
 require(c['complete'] and e['mode']==mode and e['rep']==rep and e['stage']==stage,'run identity')
 basic=set(itertools.product([x['id'] for x in cases],['float16','bfloat16'],['ragged','paged'],['auto','unsplit']))
 expected_rows=len(basic)*blocks*3*(4 if mode=='pristine' else 8);require(len(rows)==expected_rows==c['rows'],'row count')
 idx={}
 for r in rows:
  k=((r['case'],r['dtype'],r['layout'],r['split']),r['block'],r['execution_mode'],r['comparison_group'],r['position'])
  require(k not in idx and r['mode']==mode and r['rep']==rep and math.isfinite(r['wall_us']) and r['wall_us']>0,'bad timing row');idx[k]=r
 qmap={}
 for q in quals:
  k=(q['case'],q['dtype'],q['layout'],q['split']);require(k in basic and k not in qmap,'bad qualification');require(q['eager_graph_exact'] is True,'graph parity');qmap[k]=q
 require(set(qmap)==basic,'qualification incomplete')
 return dict(path=path,env=e,complete=c,index=idx,qual=qmap)

def process_grid(run,k,execution,arm,group,blocks):
 vals=np.empty((blocks,),float)
 for b in range(blocks):
  xs=[r['wall_us'] for (kk,bb,ex,g,p),r in run['index'].items() if kk==k and bb==b and ex==execution and g==group and r['arm']==arm]
  require(len(xs)==(4 if arm=='pristine' else 2),f'observation multiplicity {arm} {group}');vals[b]=math.exp(sum(math.log(x) for x in xs)/len(xs))
 return vals

def duplicate_grid(run,k,execution,arm,group,blocks):
 out=[]
 for b in range(blocks):
  xs=[r['wall_us'] for (kk,bb,ex,g,p),r in sorted(run['index'].items(),key=lambda z:z[0][-1]) if kk==k and bb==b and ex==execution and g==group and r['arm']==arm]
  require(len(xs)==2 if arm!='pristine' else len(xs)==4,'duplicate multiplicity')
  if arm=='pristine':out.append(math.log(math.sqrt(xs[0]*xs[-1]))-math.log(math.sqrt(xs[1]*xs[2])))
  else:out.append(math.log(xs[0])-math.log(xs[1]))
 return np.array(out)

def run(a):
 m=load();cases=[c for c in m['cases'] if c['family']==a.stage];blocks=2 if a.stage=='canary' else 8;shards=a.shards
 lookup={}
 for shard in range(shards):
  subset=[c for i,c in enumerate(cases) if i%shards==shard]
  for rep in range(1 if a.stage=='canary' else 3):
   for mode in ('pristine','paired'):
    paths=sorted((a.root/'runs').glob(f'{a.stage}-{a.gpu}-s{shard}-r{rep}-{mode}-*'));require(len(paths)==1,f'run discovery {shard} {rep} {mode} {paths}')
    lookup[(shard,rep,mode)]=load_run(paths[0],mode,rep,a.stage,subset,blocks)
 # Environment contract: each shard is one physical GPU and both modes share driver/runtime/source archive.
 hardware={};source_archives=set();official=set()
 for shard in range(shards):
  rs=[lookup[(shard,rep,mode)] for rep in range(1 if a.stage=='canary' else 3) for mode in ('pristine','paired')]
  uuids={r['env']['hardware']['out'].strip().splitlines()[1].split(',')[1].strip() for r in rs};drivers={r['env']['hardware']['out'].strip().splitlines()[1].split(',')[2].strip() for r in rs};require(len(uuids)==len(drivers)==1,'environment drift within shard');hardware[str(shard)]=dict(uuid=next(iter(uuids)),driver=next(iter(drivers)))
  source_archives|={r['env']['source_archive_sha256'] for r in rs};official|={r['env']['official_overlay_sha256'] for r in rs}
 require(len(source_archives)==len(official)==1,'campaign provenance drift')
 numerics={arm:dict(qualifications=0,exact_full_outputs=0,max_abs_vs_pristine=0.0) for arm in ('pristine','off','cap','guarded')};cells=[];drawcache={};canary_points=[]
 for shard in range(shards):
  subset=[c for i,c in enumerate(cases) if i%shards==shard];reps=range(1 if a.stage=='canary' else 3)
  base=lookup[(shard,0,'pristine')]
  for k in base['qual']:
   qs=[]
   for rep in reps:
    pq=lookup[(shard,rep,'pristine')]['qual'][k];cq=lookup[(shard,rep,'paired')]['qual'][k];require(cq['candidate_plan_core_equal'] is True,'candidate plan core');require(cq['inputs_sha256']==pq['inputs_sha256'],'input mismatch');require(cq['guard_expected']==cq['arms']['guarded']['plan_info'][-1],'guard plan mismatch')
    qs.append((pq,cq))
    numerics['pristine']['qualifications']+=1;numerics['pristine']['exact_full_outputs']+=int(pq['arms']['pristine']['pristine_exact'])
    for arm in ('off','cap','guarded'):numerics[arm]['qualifications']+=1;numerics[arm]['exact_full_outputs']+=int(cq['arms'][arm]['pristine_exact'])
   selected=bool(qs[0][1]['guard_expected']);require(all(bool(cq['guard_expected'])==selected for _,cq in qs),'selector changed')
   for execution in ('eager_full_call','graph1_replay','graph16_replay'):
    if a.stage=='canary':
     p=process_grid(lookup[(shard,0,'pristine')],k,execution,'pristine','position',blocks);off=process_grid(lookup[(shard,0,'paired')],k,execution,'off','guarded',blocks);g=process_grid(lookup[(shard,0,'paired')],k,execution,'guarded','guarded',blocks)
     canary_points.append(dict(case=k[0],dtype=k[1],layout=k[2],split=k[3],execution_mode=execution,selected=selected,off_ratio=float(np.exp(np.mean(np.log(p/off)))),guarded_ratio=float(np.exp(np.mean(np.log(p/g)))),paired_ratio=float(np.exp(np.mean(np.log(off/g))))));continue
    P=[];O=[];G=[];C=[];position={'pristine':[],'off':[],'guarded':[]}
    for rep in reps:
     pr=lookup[(shard,rep,'pristine')];pa=lookup[(shard,rep,'paired')]
     P.append(process_grid(pr,k,execution,'pristine','position',blocks));O.append(process_grid(pa,k,execution,'off','guarded',blocks));G.append(process_grid(pa,k,execution,'guarded','guarded',blocks));C.append(process_grid(pa,k,execution,'cap','cap',blocks))
     position['pristine'].append(duplicate_grid(pr,k,execution,'pristine','position',blocks));position['off'].append(duplicate_grid(pa,k,execution,'off','guarded',blocks));position['guarded'].append(duplicate_grid(pa,k,execution,'guarded','guarded',blocks))
    P=np.array(P);O=np.array(O);G=np.array(G);C=np.array(C);W=weights(shard)
    logs={'guarded':(np.log(P)-np.log(G)).reshape(24),'off':(np.log(P)-np.log(O)).reshape(24),'cap':(np.log(P)-np.log(C)).reshape(24),'paired_guarded':(np.log(O)-np.log(G)).reshape(24)}
    comparisons={}
    pos_ok=True;position_receipt={}
    for arm,grid in position.items():
     d=W@np.array(grid).reshape(24);ci=qci(d,(.05,.95));ok=ci[0]>=1/1.005 and ci[1]<=1.005;pos_ok&=ok;position_receipt[arm]=dict(CI90=ci,resolves_half_percent=ok)
    overlay_draw=W@logs['off'];overlay90=qci(overlay_draw,(.05,.95));overlay_ok=overlay90[0]>=1/1.01 and overlay90[1]<=1.01
    for mode,vec in logs.items():
     d=W@vec;comparisons[mode]=dict(ratio=float(math.exp(vec.mean())),CI95=qci(d),controls_resolve=bool(pos_ok and overlay_ok),position_controls=position_receipt,disabled_overlay_CI90=overlay90,disabled_overlay_resolves_one_percent=overlay_ok)
     drawcache[(len(cells),mode)]=d
    cells.append(dict(case=k[0],dtype=k[1],layout=k[2],split=k[3],execution_mode=execution,shard=shard,selected=selected,comparisons=comparisons))
 if a.stage=='canary':
  req=dict(numerical_exact=all(v['qualifications']==v['exact_full_outputs'] for v in numerics.values()),selected_nonempty=any(x['selected'] for x in canary_points),selected_point_worst_at_least_0_98=min(x['guarded_ratio'] for x in canary_points if x['selected'])>=.98,policy_point_worst_at_least_0_98=min(x['guarded_ratio'] for x in canary_points)>=.98,disabled_overlay_point_worst_at_least_0_98=min(x['off_ratio'] for x in canary_points)>=.98)
  result=dict(stage=a.stage,gpu=a.gpu,canary_gate={'pass':all(req.values()),'requirements':req},canary_points=canary_points,numerics=numerics,hardware=hardware,case_hash=m['case_hash'],source_archive_sha256=next(iter(source_archives)),official_overlay_sha256=next(iter(official)),single_gpu_release_gate_pass=False)
 else:
  gate=evaluate(cells,drawcache,numerics);result=dict(stage=a.stage,gpu=a.gpu,release_gate=gate,cells=cells,numerics=numerics,hardware=hardware,case_hash=m['case_hash'],source_archive_sha256=next(iter(source_archives)),official_overlay_sha256=next(iter(official)),single_gpu_release_gate_pass=gate['pass'])
 a.out.mkdir(parents=True);(a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 if a.stage=='canary':
  md=['# Selector-v3.2 paired canary','',f'**Canary gate: {"PASS" if result["canary_gate"]["pass"] else "HOLD"}**','',f'- Requirements: `{json.dumps(result["canary_gate"]["requirements"],sort_keys=True)}`']
 else:
  g=result['release_gate'];md=['# Selector-v3.2 paired release','',f'**Single-GPU release gate: {"PASS" if g["pass"] else "HOLD"}**','',f'- Selected cells: {g["selected"]["count"]}; geomean {g["selected"]["ratio"]:.6f}; point worst {g["selected"]["worst_point_ratio"]:.6f}; joint-min LCB {g["selected"]["simultaneous_worst_CI95"][0]:.6f}.',f'- Graph16 selected geomean {g["graph16_selected"]["ratio"]:.6f}; 95% CI [{g["graph16_selected"]["CI95"][0]:.6f}, {g["graph16_selected"]["CI95"][1]:.6f}].',f'- Requirements: `{json.dumps(g["requirements"],sort_keys=True)}`']
 (a.out/'RESULTS.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['canary','release'],required=True);p.add_argument('--gpu',choices=['gpu_4090','gpu_5090'],required=True);p.add_argument('--shards',type=int,required=True);run(p.parse_args())
