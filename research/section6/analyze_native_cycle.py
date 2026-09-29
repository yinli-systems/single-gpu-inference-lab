"""Pristine lifecycle audit. Separate process modes do NOT share physical blocks."""
import argparse,collections,itertools,json,math,statistics
from pathlib import Path
import numpy as np
from analyze import require,sha,weights,estimate
MODES=('pristine','disabled','candidate')

def cross_interval(base,alt):
 # Repeat IDs pair the declared input seed and launch rotation. Block samples
 # are INDEPENDENT between separately launched modes, not fake matched blocks.
 base=np.asarray(base);alt=np.asarray(alt);require(base.shape==alt.shape==(3,12),'invalid process grids')
 rng=np.random.default_rng(82731);draws=[]
 for _ in range(10000):
  samples=[]
  for r in rng.integers(0,3,3):
   samples.append(float(base[r,rng.integers(0,12,12)].mean()-alt[r,rng.integers(0,12,12)].mean()))
  draws.append(statistics.mean(samples))
 return dict(ratio=math.exp(float((base-alt).mean())),CI=[math.exp(float(v)) for v in np.quantile(draws,[.025,.975])])

def run(root,out):
 require(not out.exists(),'preserve existing output')
 paths=sorted((root/'native-runs').glob('gpu_*-r*-*'));require(len(paths)==18,'18 native-cycle runs required')
 groups=collections.defaultdict(list)
 binding=json.loads(json.loads((root/'receipts/native-preparation.json').read_text())['stdout'])
 for p in paths:
  require((p/'launcher-complete.txt').exists() and not (p/'failure.json').exists(),'incomplete/failed native run '+str(p))
  c=json.loads((p/'complete.json').read_text());e=json.loads((p/'environment.json').read_text())
  require(c['complete'] is True and c['rows']==96,'bad completion count')
  for f,h in c['files'].items():require(Path(f).name==f and sha(p/f)==h,'raw hash mismatch')
  require(e['mode'] in MODES and e['rep'] in (0,1,2) and e['flashinfer']=='0.7.0','unknown mode')
  expected_header='e66bce2652c0a3c5b54f88510aff2b4b48ff6a52c4ee23fa97f91fa06a935b87' if e['mode']=='pristine' else '1ed0b211270f46bad90b205dbb9bad4e86266f442e2be12e1d547bb6a718ad40'
  require(e['prefill_header_sha256']==expected_header,'unqualified native header')
  require(e['scheduler_header_sha256']==binding['scheduler_original_sha256' if e['mode']=='pristine' else 'scheduler_modified_sha256'],'unqualified planner')
  q=json.loads((p/'qualification.json').read_text());require(q['full_output_exact'] is True and q['lse_exact'] is True,'bad output qualification')
  rows=json.loads((p/'measurements.json').read_text());expected=set(itertools.product(range(12),(1,32),range(4)));idx={}
  for x in rows:
   k=(x['block'],x['calls'],x['position']);require(k in expected and k not in idx,'duplicate/unexpected cycle')
   require(x['rep']==e['rep'] and x['mode']==e['mode'],'cycle source mismatch')
   seq=('main','repeat','repeat','main') if x['block']%2==0 else ('repeat','main','main','repeat')
   require(x['label']==seq[x['position']] and math.isfinite(x['cycle_us']) and x['cycle_us']>0,'invalid cycle')
   idx[k]=x
  require(set(idx)==expected,'incomplete native cycle matrix')
  hw=e['hardware']['out'].strip().splitlines()[-1].split(',')
  groups[e['gpu']].append(dict(path=p,env=e,q=q,idx=idx,uuid=hw[1].strip(),driver=hw[2].strip(),receipt=sha(p/'complete.json')))
 require(len(groups)==2,'missing GPU family')
 w=weights();report=dict(complete=True,rows=1728,processes=18,qualification_records=18,GPUs={},default_promotion=False,serving_promotion=False,
  analysis_sha256=sha(__file__),scope='exposed BF16/unsplit exact witness; actual plan+sort+transfer+graph execution; startup/JIT excluded; no fresh/general serving claim')
 md=['# Native plan-plus-run result','', 'Pristine official0.7.0 versus disabled and enabled isolated native source. Single exposed witness. Three processes per mode; independent block resampling across modes.','',
 '|GPU|Actual calls per plan|Pristine/disabled [95% CI]|Pristine/candidate [95% CI]|All mode A/A controls resolve1%|','|---|---:|---|---|---|']
 for gpu,rs in sorted(groups.items()):
  require(len(rs)==9 and {(r['env']['mode'],r['env']['rep']) for r in rs}==set(itertools.product(MODES,range(3))),'mode/repeat coverage')
  require(len({r['uuid'] for r in rs})==len({r['driver'] for r in rs})==1,'physical context mismatch')
  require(len({r['env']['source_sha256'] for r in rs})==1,'mixed probe code')
  idx={(r['env']['mode'],r['env']['rep']):r for r in rs}
  for rep in range(3):
   require(len({json.dumps(idx[(m,rep)]['q']['hashes'],sort_keys=True) for m in MODES})==1,'cross-source input/output/LSE hash mismatch')
  launch=[]
  for mode in MODES:
   xs=json.loads((idx[(mode,0)]['path']/'launch-proof.json').read_text());require(len(xs)==1,'missing launch proof')
   require(xs[0]['args']['shared memory']==(65536 if mode=='candidate' else 49152),'wrong launch capacity')
   launch.append(dict(mode=mode,**xs[0]))
  values=[]
  for calls in (1,32):
   grids={};controls={};medians={}
   for mode in MODES:
    grid=[];aa=[];raw=[]
    for rep in range(3):
     run=idx[(mode,rep)];line=[];aline=[]
     for b in range(12):
      xs=[run['idx'][(b,calls,p)] for p in range(4)];ys=[math.log(x['cycle_us']) for x in xs];line.append(statistics.mean(ys));raw.extend(x['cycle_us'] for x in xs)
      aline.append(statistics.mean(math.log(x['cycle_us']) for x in xs if x['label']=='main')-statistics.mean(math.log(x['cycle_us']) for x in xs if x['label']=='repeat'))
     grid.append(line);aa.append(aline)
    grids[mode]=grid;controls[mode]=estimate(aa,w,.90);medians[mode]=statistics.median(raw)
   stats={m:cross_interval(grids['pristine'],grids[m]) for m in ('disabled','candidate')}
   clean=all(s['CI'][0]>=1/1.005 and s['CI'][1]<=1.005 for s in controls.values())
   values.append(dict(calls=calls,comparisons=stats,AA=controls,all_controls_resolve_one_percent=clean,cycle_medians_us=medians))
   def fmt(m):return '%.6f [%.6f, %.6f]'%(stats[m]['ratio'],*stats[m]['CI'])
   md.append('|%s|%d|%s|%s|%s|'%(gpu,calls,fmt('disabled'),fmt('candidate'),clean))
  report['GPUs'][gpu]=dict(uuid=rs[0]['uuid'],driver=rs[0]['driver'],values=values,launch_proof=launch,receipts={r['path'].name:r['receipt'] for r in rs})
 md+=['','No unconditional promotion. Exact-witness support is intentionally narrow. Paged cache, dynamic graph plans, threaded/multistream ownership, new shapes and real full-model serving are not qualified. A1% result is unresolved when its matching A/A controls fail; graph32 does not stand in for32 different transformer layers.']
 out.mkdir(parents=True);(out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(out/'RESULTS.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.root,a.out)
