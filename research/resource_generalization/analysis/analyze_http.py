"""HTTP integrity and paired-allocation statistics; no profile-time performance."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np

def require(c,m):
 if not c:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_mode(path,stage):
 require(not (path/'failure.json').exists(),'failed server '+str(path));c=json.loads((path/'complete.json').read_text())
 require(c['complete'] is True and c['stage']==stage and c['HTTP'] is True and c['full_model'] is True,'wrong completion')
 for f,h in c['files'].items():require(Path(f).name==f and sha(path/f)==h,'HTTP hash mismatch')
 data={}
 for f in sorted(path.glob('*-b*.json')):
  x=json.loads(f.read_text());require(not x['errors'],'request failures retained')
  expected=4 if stage=='smoke' else 16;require(len(x['requests'])==expected,'request coverage')
  require(len({r['id'] for r in x['requests']})==expected,'duplicate request ID')
  for r in x['requests']:
   require(r['tokens'] and len(r['tokens'])==len(r['token_times']),'missing token events')
   require(all(math.isfinite(r[k]) and r[k]>=0 for k in ['ttft','tpot','latency']),'invalid latency')
   require(all(a<=b for a,b in zip(r['token_times'],r['token_times'][1:])),'time reversal')
  require(x['output_tokens']==sum(len(r['tokens']) for r in x['requests']),'wrong throughput numerator')
  require(abs(x['output_tokens']/x['elapsed']-x['output_tokens_per_second'])<1e-8,'wrong throughput')
  data[f.name]=x
 expectedfiles={'smoke-b0.json'} if stage=='smoke' else {f'{w}-b{b}.json' for w in ['prefill','decode','mixed'] for b in range(3)}
 require(set(data)==expectedfiles,'workload matrix incomplete')
 return c,data

def run(a):
 require(not a.out.exists(),'preserve audit');jobs=[];allmismatch=[]
 for job in a.jobs:
  p=a.root/'serving-runs'/str(job);require((p/'launcher-complete.txt').exists() and (p/'exit.txt').read_text().strip()=='0','job not complete')
  modes={};model=json.loads((p/'model-binding.json').read_text());require(len(model)==5,'whole three-shard model not hashed')
  for mode in ['pristine','off','cap']:
   cp=json.loads((p/('lifecycle-'+mode)/'complete.json').read_text());require(cp['complete'] and cp['checks']==32,'missing lifecycle')
   for f,h in cp['files'].items():require(sha(p/('lifecycle-'+mode)/f)==h,'lifecycle hash mismatch')
   c,d=read_mode(p/mode,a.stage);require(c['mode']==mode,'mode mismatch');modes[mode]=d
  mismatches=[]
  for file,base in modes['pristine'].items():
   ids={r['id']:r['tokens'] for r in base['requests']}
   for mode in ['off','cap']:
    x=modes[mode][file];require(x['workload_sha256']==base['workload_sha256'],'changed workload')
    for r in x['requests']:
     y=ids[r['id']]
     if y!=r['tokens']:
      first=next((i for i,(u,v) in enumerate(zip(y,r['tokens'])) if u!=v),min(len(y),len(r['tokens'])))
      mismatches.append(dict(file=file,mode=mode,id=r['id'],first_difference=first,lengths=[len(y),len(r['tokens'])]))
  profile=p/'cap/profile-evidence.json';profiledata=json.loads(profile.read_text()) if profile.exists() else None
  allmismatch.extend([dict(job=job,**x) for x in mismatches])
  jobs.append(dict(job=job,hardware=(p/'hardware.csv').read_text(),model_binding=model,modes=modes,
   token_mismatches=mismatches,profile=profiledata))
 require(len({json.dumps(j['model_binding'],sort_keys=True) for j in jobs})==1,'model changed')
 report=dict(stage=a.stage,jobs=a.jobs,complete=True,full_model=True,HTTP=True,
  token_parity=not allmismatch,mismatches=allmismatch,
  total_successful_requests=sum(len(x['requests']) for j in jobs for ds in j['modes'].values() for x in ds.values()),
  cap_profile_launches_64KiB=sum((j['profile'] or {}).get('launch_hit_count_64KiB',0) for j in jobs),
  model_binding=jobs[0]['model_binding'],default_promotion=False,serving_promotion=False)
 if a.stage=='formal':
  require(len(jobs)==3,'three paired allocations required')
  rng=np.random.default_rng(939303);drawjobs=rng.integers(0,3,(10000,3));drawb=rng.integers(0,3,(10000,3,3));drawc=rng.integers(0,3,(10000,3,3))
  results=[]
  for work in ['prefill','decode','mixed']:
   for mode in ['off','cap']:
    for metric in ['output_tokens_per_second','strict_slo_goodput','TTFT-p50','TTFT-p99','TPOT-p50','TPOT-p99']:
     def get(x):
      if '-' in metric:
       kind,pct=metric.split('-');return x[kind]['0.5' if pct=='p50' else '0.99']
      return x[metric]
     b=np.array([[get(j['modes']['pristine'][f'{work}-b{k}.json']) for k in range(3)] for j in jobs]);c=np.array([[get(j['modes'][mode][f'{work}-b{k}.json']) for k in range(3)] for j in jobs])
     require(np.isfinite(b).all() and np.isfinite(c).all(),'bad serving metric')
     if (b<=0).any() or (c<=0).any():
      results.append(dict(workload=work,mode=mode,metric=metric,ratio=None,reason='zero observations: no log-ratio',pristine=b.tolist(),candidate=c.tolist()));continue
     # Higher throughput/goodput or lower latency is favorable.
     sign=-1 if metric in ['output_tokens_per_second','strict_slo_goodput'] else 1
     diff=sign*(np.log(b)[drawjobs[:,:,None],drawb]-np.log(c)[drawjobs[:,:,None],drawc]).mean(axis=(1,2))
     results.append(dict(workload=work,mode=mode,metric=metric,favorable_ratio=float(np.exp(sign*(np.log(b).mean()-np.log(c).mean()))),
       CI95=[float(x) for x in np.exp(np.quantile(diff,[.025,.975]))],pristine_geomean=float(np.exp(np.log(b).mean())),candidate_geomean=float(np.exp(np.log(c).mean()))))
  report['comparisons']=results;report['statistical_scope']='Paired physical allocation bootstrap, independent workload-block resampling within each arm; three allocations only. Tail estimates conditional on 16 requests per batch, not population p99 guarantees.'
 a.out.mkdir(parents=True);(a.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['smoke','formal'],required=True);p.add_argument('--jobs',type=int,nargs='+',required=True);run(p.parse_args())
