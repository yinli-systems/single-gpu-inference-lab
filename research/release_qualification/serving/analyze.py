"""Fail-closed multi-model serving performance and deterministic correctness analysis."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np

def require(c,m):
 if not c:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_mode(path,stage,blocks):
 require(not (path/'failure.json').exists(),'failed mode '+str(path))
 c=json.loads((path/'complete.json').read_text());require(c['complete'] and c['stage']==stage and c['full_model'] and c['HTTP'],'wrong completion')
 for f,h in c['files'].items():require(Path(f).name==f and sha(path/f)==h,'hash mismatch '+str(path/f))
 data={}
 for f in sorted(path.glob('*-b*.json')):
  x=json.loads(f.read_text());require(not x['errors'],'request errors '+str(f));require(len(x['requests']) in (6,8),'coverage')
  require(len({r['rid'] for r in x['requests']})==len(x['requests']),'RID reuse')
  for r in x['requests']:
   require(len(r['tokens'])==len(r['token_times']) and r['tokens'],'missing tokens')
   require(all(math.isfinite(r[k]) and r[k]>=0 for k in ('ttft','tpot','latency')),'bad timing')
  require(x['output_tokens']==sum(len(r['tokens']) for r in x['requests']),'throughput numerator')
  require(abs(x['output_tokens']/x['elapsed']-x['output_tokens_per_second'])<1e-8,'throughput formula')
  if x['workload'] in ('guarded_prefix','balanced_prefix'):require(x['cached_min']>=8192,'prefix miss')
  data[f.name]=x
 expected={f'{w}-b{b}.json' for w in ('guarded_prefix','balanced_prefix','short_prefill','decode','mixed') for b in range(blocks)}
 require(set(data)==expected,'incomplete workload matrix')
 return c,data

def numeric_logprobs(x):
 out=[]
 def rec(v):
  if isinstance(v,bool) or v is None or isinstance(v,str):return
  if isinstance(v,(int,float)):out.append(float(v));return
  if isinstance(v,list):
   for z in v:rec(z)
  elif isinstance(v,dict):
   for k in sorted(v):rec(v[k])
 for r in x['requests']:
  require(r['output_token_logprobs'] is not None,'missing output logprobs')
  rec(r['output_token_logprobs']);rec(r['output_top_logprobs'])
 return np.array(out,dtype=np.float64)
def ids(x):return {r['id']:r['tokens'] for r in x['requests']}
def get_metric(x,metric):
 if '-' in metric:
  kind,p=metric.split('-');return x[kind][{'p50':'0.5','p95':'0.95','p99':'0.99'}[p]]
 return x[metric]

def load_jobs(a):
 blocks={'smoke':1,'performance':4,'correctness':2}[a.stage];jobs=[]
 for job in a.jobs:
  p=a.root/'serving-runs'/str(job);require((p/'launcher-complete.txt').exists() and (p/'exit.txt').read_text().strip()=='0','job incomplete')
  binding=json.loads((p/'model-binding.json').read_text());modes={}
  for mode in ('pristine','off','cap','guarded'):
   c,d=read_mode(p/mode,a.stage,blocks);require(c['model_id']==a.model_id and c['mode']==mode,'identity');modes[mode]=d
  jobs.append(dict(job=job,root=p,binding=binding,modes=modes,env=json.loads((p/'pristine/environment.json').read_text())))
 require(len({json.dumps(j['binding'],sort_keys=True) for j in jobs})==1,'model changed')
 require(len({j['env']['model_id'] for j in jobs})==1,'model id changed')
 return jobs,blocks

def run(a):
 require(not a.out.exists(),'preserve analysis');jobs,blocks=load_jobs(a)
 report=dict(stage=a.stage,model_id=a.model_id,jobs=a.jobs,complete=True,full_model=True,HTTP=True,default_promotion=False,serving_promotion=False)
 # Workload hashes and request sets are exact across every arm/allocation.
 for w in ('guarded_prefix','balanced_prefix','short_prefill','decode','mixed'):
  for b in range(blocks):
   vals=[];idsets=[]
   for j in jobs:
    for mode in ('pristine','off','cap','guarded'):
     x=j['modes'][mode][f'{w}-b{b}.json'];vals.append(x['workload_sha256']);idsets.append(sorted(r['id'] for r in x['requests']))
   require(len(set(vals))==1,'workload changed')
   # RIDs deliberately include arm; compare logical IDs instead.
   require(len({tuple(r.split('-',4)[-1] for r in ids) for ids in idsets})==1,'logical requests changed')
 mismatches=[]
 for j in jobs:
  for f,base in j['modes']['pristine'].items():
   baseids=ids(base)
   for mode in ('off','cap','guarded'):
    for rid,tok in ids(j['modes'][mode][f]).items():
     if tok!=baseids[rid]:
      first=next((i for i,(u,v) in enumerate(zip(tok,baseids[rid])) if u!=v),min(len(tok),len(baseids[rid])))
      mismatches.append(dict(job=j['job'],file=f,mode=mode,id=rid,first_difference=first))
 report['cross_arm_token_mismatches']=mismatches
 if a.stage=='correctness':
  require(len(jobs)>=1,'correctness jobs')
  max_logprob=0.;comparisons=0
  for j in jobs:
   for f,base in j['modes']['pristine'].items():
    baseids=ids(base);bv=numeric_logprobs(base)
    for mode in ('off','cap','guarded'):
     x=j['modes'][mode][f];require(ids(x)==baseids,'deterministic token mismatch')
     cv=numeric_logprobs(x);require(cv.shape==bv.shape,'logprob shape');max_logprob=max(max_logprob,float(np.max(np.abs(cv-bv))) if len(bv) else 0.);comparisons+=1
  require(max_logprob==0,'deterministic logprobs not exact')
  report.update(token_parity=True,logprob_max_abs=max_logprob,comparison_files=comparisons,correctness_pass=True)
 else:
  require(a.stage=='performance' and len(jobs)==3,'three paired performance allocations')
  # Profile gate from the first physical allocation.
  prof={}
  first=jobs[0]
  for mode in ('pristine','off','cap','guarded'):
   x=json.loads((first['root']/mode/'profile-evidence.json').read_text());prof[mode]=x
  require(prof['guarded']['guarded_prefix']['launches_64KiB']>0,'guard did not activate')
  require(prof['guarded']['balanced_prefix']['launches_64KiB']==0,'guard activated on balanced control')
  require(prof['pristine']['guarded_prefix']['launches_64KiB']==0 and prof['off']['guarded_prefix']['launches_64KiB']==0,'control has cap launch')
  report['profile_gate']=prof
  rng=np.random.default_rng(2026093029);drawj=rng.integers(0,3,(20000,3));drawb=rng.integers(0,blocks,(20000,3,blocks));drawc=rng.integers(0,blocks,(20000,3,blocks))
  metrics=('output_tokens_per_second','strict_slo_goodput','TTFT-p50','TTFT-p95','TTFT-p99','TPOT-p50','TPOT-p95','TPOT-p99')
  results=[]
  for w in ('guarded_prefix','balanced_prefix','short_prefill','decode','mixed'):
   for mode in ('off','cap','guarded'):
    for metric in metrics:
     b=np.array([[get_metric(j['modes']['pristine'][f'{w}-b{k}.json'],metric) for k in range(blocks)] for j in jobs])
     c=np.array([[get_metric(j['modes'][mode][f'{w}-b{k}.json'],metric) for k in range(blocks)] for j in jobs])
     require(np.isfinite(b).all() and np.isfinite(c).all() and (b>0).all() and (c>0).all(),'bad metric')
     high=metric in ('output_tokens_per_second','strict_slo_goodput')
     point=float(np.exp((np.log(c).mean()-np.log(b).mean()) if high else (np.log(b).mean()-np.log(c).mean())))
     # Preserve physical-allocation pairing; resample blocks independently inside each arm.
     vals=[]
     for d in range(len(drawj)):
      bs=[];cs=[]
      for jj in drawj[d]:
       bs.extend(np.log(b[jj,drawb[d,jj]]));cs.extend(np.log(c[jj,drawc[d,jj]]))
      vals.append((np.mean(cs)-np.mean(bs)) if high else (np.mean(bs)-np.mean(cs)))
     ci=[float(x) for x in np.exp(np.quantile(vals,[.025,.975]))]
     results.append(dict(workload=w,mode=mode,metric=metric,favorable_ratio=point,CI95=ci,pristine_geomean=float(np.exp(np.log(b).mean())),candidate_geomean=float(np.exp(np.log(c).mean()))))
  report['comparisons']=results
  guarded_thr=[x for x in results if x['mode']=='guarded' and x['metric']=='output_tokens_per_second']
  report['guarded_throughput_summary']=dict(geomean=float(np.exp(np.mean([math.log(x['favorable_ratio']) for x in guarded_thr]))),
   worst=min(x['favorable_ratio'] for x in guarded_thr),below_0_99=[x['workload'] for x in guarded_thr if x['favorable_ratio']<.99],
   guarded_prefix=next(x for x in guarded_thr if x['workload']=='guarded_prefix'))
  report['performance_gate']=bool(not report['guarded_throughput_summary']['below_0_99'] and report['guarded_throughput_summary']['geomean']>1 and report['guarded_throughput_summary']['guarded_prefix']['CI95'][0]>1)
  report['statistical_scope']='Three paired physical allocations; allocation bootstrap and independent within-arm block resampling. Conditional on these models, synthetic inputs and RTX4090 GPUs.'
 a.out.mkdir(parents=True);(a.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('comparisons','profile_gate')},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['smoke','performance','correctness'],required=True);p.add_argument('--model-id',required=True);p.add_argument('--jobs',nargs='+',type=int,required=True);run(p.parse_args())
