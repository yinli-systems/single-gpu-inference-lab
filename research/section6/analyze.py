"""Fail-closed section-six analysis; never substitutes profiler timings."""
from __future__ import annotations
import argparse,collections,hashlib,itertools,json,math,statistics
from pathlib import Path
import numpy as np
from measure import CASES,POLICIES,REGIMES,BLOCKS

def require(v,msg):
 if not v:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_run(p):
 require((p/'launcher-complete.txt').exists() and not (p/'failure.json').exists(),'incomplete run '+str(p))
 c=json.loads((p/'complete.json').read_text());e=json.loads((p/'environment.json').read_text())
 require(c['complete'] is True,'bad completion')
 for n,h in c['files'].items():
  require(Path(n).name==n and sha(p/n)==h,'evidence hash mismatch '+n)
 require(e['cases']==[[n,q,k] for n,q,k in CASES] and tuple(e['policies'])==POLICIES and tuple(e['regimes'])==REGIMES and e['blocks']==BLOCKS,'changed protocol')
 require(e['rep'] in (0,1,2) and e['version'] in ('0.6.18','0.7.0'),'wrong repeat/version')
 checks=json.loads((p/'qualification.json').read_text());plans=json.loads((p/'plans.json').read_text())
 expected_plans=set(itertools.product([c[0] for c in CASES],('float16','bfloat16'),('auto','unsplit')))
 require(len(plans)==24 and {(x['case'],x['dtype'],x['split']) for x in plans}==expected_plans,'plan coverage')
 checkmap={}
 for x in checks:
  k=(x['case'],x['dtype'],x['split'],x['policy'])
  require(k not in checkmap and x['exact'] is True,'qualification duplicate/failure');checkmap[k]=x
 require(set(checkmap)=={c+(pol,) for c in expected_plans for pol in ('identity',)+POLICIES},'qualification coverage')
 for cell in expected_plans:
  h=checkmap[cell+('identity',)]['descriptor_hash']
  for pol in ('identity_label','identity_copy'):
   require(checkmap[cell+(pol,)]['descriptor_hash']==h and checkmap[cell+(pol,)]['changed'] is False,'no-op control changed work')
 rows=json.loads((p/'measurements.json').read_text());idx={}
 expected=set(itertools.product(expected_plans,range(BLOCKS),POLICIES,REGIMES,range(4)))
 for x in rows:
  cell=(x['case'],x['dtype'],x['split']);k=(cell,x['block'],x['policy'],x['regime'],x['position'])
  require(k in expected and k not in idx,'duplicate/unexpected timing')
  require(x['rep']==e['rep'] and x['version']==e['version'],'wrong row identity')
  seq=('identity',x['policy'],x['policy'],'identity') if x['block']%2==0 else (x['policy'],'identity','identity',x['policy'])
  require(x['arm']==seq[x['position']],'unbalanced arm sequence')
  require(x['descriptor_hash']==checkmap[cell+(x['arm'],)]['descriptor_hash'],'row/qualification mismatch')
  require(x['calls']==(32 if x['regime']=='graph32_steady' else 1),'wrong measured call count')
  for f in ('device_us','wall_us','setup_us'):require(math.isfinite(x[f]) and x[f]>0,'bad timing')
  idx[k]=x
 require(set(idx)==expected and len(rows)==c['rows']==c['expected']==23040,'incomplete matrix')
 hardware=e['hardware']['out'].strip().splitlines()[-1].split(',')
 require(len(hardware)>=5,'hardware identity missing')
 return dict(path=p,env=e,rows=rows,index=idx,checks=checkmap,plans=plans,uuid=hardware[1].strip(),driver=hardware[2].strip(),receipt=sha(p/'complete.json'))

def weights():
 rng=np.random.default_rng(391832);w=np.zeros((10000,3*BLOCKS))
 for row in w:
  for rep in rng.integers(0,3,3):
   for b in rng.integers(0,BLOCKS,BLOCKS):row[rep*BLOCKS+b]+=1
 return w/(3*BLOCKS)

def estimate(a,w,confidence=.95):
 a=np.asarray(a);require(a.shape==(3,BLOCKS) and np.isfinite(a).all(),'invalid paired grid')
 means=w@a.reshape(-1);tail=(1-confidence)/2
 return dict(ratio=math.exp(float(a.mean())),CI=[math.exp(float(v)) for v in np.quantile(means,[tail,1-tail])],confidence=confidence)

def paired(runs,cell,pol,regime,metric='device_us'):
 out=[]
 for run in runs:
  vals=[]
  for b in range(BLOCKS):
   xs=[run['index'][(cell,b,pol,regime,pos)] for pos in range(4)]
   base=[math.log(x[metric]) for x in xs if x['arm']=='identity'];cand=[math.log(x[metric]) for x in xs if x['arm']!='identity']
   require(len(base)==len(cand)==2,'incomplete pair')
   vals.append(statistics.mean(base)-statistics.mean(cand))
  out.append(vals)
 return out

def analyze(root,out):
 require(not out.exists(),'preserve prior result')
 paths=sorted((root/'runs').glob('gpu_*-0.*-r*-*'))
 complete=[p for p in paths if (p/'complete.json').exists() and (p/'launcher-complete.txt').exists()]
 require(len(complete)==12,'twelve completed runs required; have '+str(len(complete)))
 groups=collections.defaultdict(list)
 for p in complete:
  x=validate_run(p);groups[(x['env']['gpu'],x['env']['version'])].append(x)
 require(len(groups)==4,'missing GPU/version group')
 w=weights();summary=dict(complete=True,measurement_rows=0,qualified_records=0,groups=[],
  default_promotion=False,serving_promotion=False,source_sha256=sha(__file__),
  scope='exposed diagnostic cases; run-only; cellwise conditional intervals; no unseen-workload or hardware-population claim')
 md=['# Section-six diagnostic results','', 'All diagnostic cells retained. This is not a serving speedup. Graph32 is a distinct steady-reuse regime.','',
 '|GPU|Version|Regime|No-op controls within ±0.5% (both labels)|Cells|','|---|---|---|---:|---:|']
 detail=[]
 for (gpu,ver),rs in sorted(groups.items()):
  rs.sort(key=lambda x:x['env']['rep'])
  require([r['env']['rep'] for r in rs]==[0,1,2],'repeat coverage')
  require(len({r['uuid'] for r in rs})==1 and len({r['driver'] for r in rs})==1,'not same-device same-driver repeats')
  require(len({json.dumps(r['env']['source'],sort_keys=True) for r in rs})==1,'mixed measurement source')
  summary['measurement_rows']+=sum(len(r['rows']) for r in rs);summary['qualified_records']+=sum(len(r['checks']) for r in rs)
  cells=[]
  for cell,regime in itertools.product(itertools.product([c[0] for c in CASES],('float16','bfloat16'),('auto','unsplit')),REGIMES):
   ps={pol:estimate(paired(rs,cell,pol,regime),w,.90 if pol.startswith('identity') else .95) for pol in POLICIES}
   clean=all(ps[pol]['CI'][0]>=1/1.005 and ps[pol]['CI'][1]<=1.005 for pol in ('identity_label','identity_copy'))
   entry=dict(case=cell[0],dtype=cell[1],split=cell[2],regime=regime,controls_resolve_one_percent=clean,policies=ps,
    locality_changed_repeats=sum(r['checks'][cell+('locality_packet8',)]['changed'] for r in rs),
    native_median_us=statistics.median(x['device_us'] for r in rs for x in r['rows'] if (x['case'],x['dtype'],x['split'],x['regime'],x['arm'])==cell+(regime,'identity')))
   cells.append(entry)
  counts=[]
  for regime in REGIMES:
   xs=[c for c in cells if c['regime']==regime];n=sum(c['controls_resolve_one_percent'] for c in xs)
   counts.append(dict(regime=regime,pass_count=n,total=len(xs)));md.append(f'|{gpu}|{ver}|{regime}|{n}|{len(xs)}|')
  summary['groups'].append(dict(gpu=gpu,version=ver,uuid=rs[0]['uuid'],driver=rs[0]['driver'],controls=counts,cells=cells,
   receipts={r['path'].name:r['receipt'] for r in rs}))
  for c in cells:
   if c['case']=='reversal2' and c['dtype']=='bfloat16' and c['split']=='unsplit':
    detail.append((gpu,ver,c))
 md+=['','## Exposed two-request reversal','', '|GPU|Version|Regime|Native/causal-heavy [95% CI]|Both no-op controls resolve 1%|','|---|---|---|---|---|']
 for gpu,ver,c in detail:
  z=c['policies']['causal_heavy'];md.append('|%s|%s|%s|%.6f [%.6f, %.6f]|%s|'%(gpu,ver,c['regime'],z['ratio'],*z['CI'],c['controls_resolve_one_percent']))
 summary['stage_D']='BLOCKED: no independently qualified net native plan+run remedy from these exposed diagnostics'
 summary['stage_E']='BLOCKED: default-OFF; no unqualified candidate inserted into live/full-model serving'
 out.mkdir(parents=True);(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');(out/'RESULTS.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();analyze(a.root,a.out)
