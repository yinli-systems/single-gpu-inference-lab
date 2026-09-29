"""Strict complete-matrix analysis; no best-subset or canary inference."""
from __future__ import annotations
import argparse
from collections import defaultdict
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path
import random
import statistics as S
from campaign import ARMS, MODES, CONFIGS, states, geometry, schema_check, digest, dump

def quantile(xs,p):
 ys=sorted(xs);i=(len(ys)-1)*p;a=int(i);b=min(a+1,len(ys)-1)
 return ys[a]+(ys[b]-ys[a])*(i-a)

def ranks(xs):
 order=sorted(range(len(xs)),key=xs.__getitem__);out=[0.]*len(xs);i=0
 while i<len(order):
  j=i+1
  while j<len(order) and xs[order[j]]==xs[order[i]]:j+=1
  for index in order[i:j]:out[index]=(i+j-1)/2
  i=j
 return out

def spearman(x,y):
 a,b=ranks(x),ranks(y);ma,mb=S.mean(a),S.mean(b)
 den=math.sqrt(sum((v-ma)**2 for v in a)*sum((v-mb)**2 for v in b))
 return sum((v-ma)*(w-mb) for v,w in zip(a,b))/den if den else None

@lru_cache(maxsize=1)
def bootstrap_weights():
 # The exact registered Random(20260929) resampling order, cached once.
 # Each draw has three sampled processes and twelve sampled blocks per process.
 rng=random.Random(20260929);result=[]
 for _ in range(5000):
  processes=[rng.randrange(3) for _ in range(3)];counts=[0]*36
  for process in processes:
   for _ in range(12):counts[process*12+rng.randrange(12)]+=1
  result.append(tuple((i,n) for i,n in enumerate(counts) if n))
 return tuple(result)

def interval(process_blocks,level=.95):
 if len(process_blocks)!=3 or any(len(v)!=12 for v in process_blocks):
  raise ValueError('three complete process repeats, twelve matched blocks required')
 flat=[x for block in process_blocks for x in block]
 draws=[math.fsum(flat[i]*n for i,n in weights)/36 for weights in bootstrap_weights()]
 alpha=(1-level)/2
 return dict(mean=S.mean(S.mean(v) for v in process_blocks),
             CI=[quantile(draws,alpha),quantile(draws,1-alpha)],confidence=level,
             replicate_means=[S.mean(v) for v in process_blocks],
             interpretation='conditional process/block uncertainty; not GPU-population inference')

def validate_rows(rows):
 expected={(d,n,b,l,a,m) for d,n in itertools.product(('float16','bfloat16'),CONFIGS)
           for b in range(12) for l,p in states(n) for a,m in itertools.product(ARMS,MODES)}
 actual=[]
 for row in rows:
  schema_check(row)
  key=(row['dtype'],row['name'],row['block'],row['state'],row['arm'],row['mode'])
  actual.append(key)
  if type(row['rep']) is not int or row['rep'] not in (0,1,2):raise ValueError('invalid replicate')
  expected_id=f"{row['rep']}/{row['dtype']}/{row['name']}/{row['block']}/{row['state']}/{row['arm']}/{row['mode']}"
  if row['trial_id']!=expected_id:raise ValueError('trial identity does not match observation')
  label=row['state'];pair=next((p for l,p in states(row['name']) if l==label),None)
  if pair is None:raise ValueError('unknown state')
  g=geometry(row['name'],label,pair)
  if any(row[k]!=v for k,v in g.items()):raise ValueError('geometry does not match registered state')
  if row['timed_calls']!=(12 if row['mode']=='eager' else 48):raise ValueError('changed repetitions')
 if len(actual)!=len(set(actual)):raise ValueError('duplicate cell')
 if set(actual)!=expected:raise ValueError('missing/extra registered cells')
 if len({r['trial_id'] for r in rows})!=len(rows):raise ValueError('duplicate trial identity')

def analyze(root,out):
 if out.exists():raise FileExistsError('new output directory required')
 dirs=sorted(root.glob('test-*'))
 if len(dirs)!=6:raise ValueError(f'exactly six formal runs required, found {len(dirs)}')
 allrows=[];identities=set();indices=set();evidence=[];gpus=set();validations=[]
 for d in dirs:
  if (d/'failure.json').exists():raise ValueError('preserve failed job; do not promote partial data')
  c=json.loads((d/'complete.json').read_text());e=json.loads((d/'environment.json').read_text())
  if not c['complete'] or c['stage']!='test' or c['rows']!=3456 or c['expected']!=3456:
   raise ValueError('incomplete formal job')
  for file,key in [('measurements.json','measurements_sha256'),('validation.json','validation_sha256')]:
   if digest(d/file)!=c[key]:raise ValueError('changed raw checksum')
  if c['source_sha256']!=e['source_sha256'] or c['source_sha256']!=digest(Path(__file__).with_name('campaign.py')):raise ValueError('source identity mismatch')
  if c['checks']!=144 or c['graph_exact_checks']!=2016:raise ValueError('missing qualification')
  if e['backend']!='fa2' or e['flashinfer']!='0.6.18' or e['scratch_MiB_per_arm']!=512:
   raise ValueError('changed runtime contract')
  gpu=e['gpu'];rep=e['rep'];key=(gpu,rep)
  if key in indices:raise ValueError('duplicate process repeat')
  indices.add(key);gpus.add(gpu);identities.add((e['source_sha256'],e['flashinfer_source_sha256'],e['torch'],e['cuda']))
  rows=json.loads((d/'measurements.json').read_text());validate_rows(rows)
  raw=[json.loads(l) for l in (d/'raw.jsonl').read_text().splitlines()]
  if raw!=rows:raise ValueError('incremental raw log differs from completed matrix')
  for r in rows:
   if r['rep']!=rep:raise ValueError('mixed process identity')
   allrows.append(dict(r,gpu=gpu))
  checks=json.loads((d/'validation.json').read_text())
  if len(checks)!=144 or not all(v['finite'] and math.isfinite(v['fp32_max_abs']) for v in checks):raise ValueError('invalid qualification record')
  expected_checks={(d,n,l,a) for d,n in itertools.product(('float16','bfloat16'),CONFIGS) for l,p in states(n) for a in ARMS}
  if {(v['dtype'],v['config'],v['state'],v['arm']) for v in checks}!=expected_checks:raise ValueError('missing or duplicate qualification')
  if any(v['fp32_vectors']!=len(CONFIGS[v['config']][0])*3*32 for v in checks):raise ValueError('missing FP32 vectors')
  validations.extend(dict(v,gpu=gpu,rep=rep) for v in checks)
  evidence.append(dict(directory=d.name,complete_sha256=digest(d/'complete.json'),metadata=e,completion=c))
 if gpus!={'NVIDIA GeForce RTX 4090','NVIDIA GeForce RTX 5090'} or indices!={(g,r) for g in gpus for r in range(3)} or len(identities)!=1:
  raise ValueError('incomplete GPU/repetition/source matrix')
 grouped=defaultdict(dict)
 for r in allrows:
  key=(r['gpu'],r['dtype'],r['name'],r['arm'],r['mode'])
  grouped[key][(r['rep'],r['block'],r['state'])]=r
 summaries=[]
 for key,rows in sorted(grouped.items()):
  gpu,dtype,name,arm,mode=key
  def values(label,metric='device_us'):
   return [[rows[(rep,b,label)][metric] for b in range(12)] for rep in range(3)]
  def contrast(a,b,level):
   av,bv=values(a),values(b)
   return interval([[x-y for x,y in zip(xs,ys)] for xs,ys in zip(av,bv)],level)
  medians={l:S.median(v for seq in values(l) for v in seq) for l,p in states(name)}
  null=contrast('order-null','same',.90);eps=max(2.,.02*medians['same'])
  null.update(epsilon_us=eps,equivalent=(-eps<=null['CI'][0] and null['CI'][1]<=eps))
  eq=None
  if name!='n2':
   eq=contrast('eqC-b','eqC-a',.90);eps=max(2.,.02*medians['eqC-a'])
   eq.update(epsilon_us=eps,equivalent=(-eps<=eq['CI'][0] and eq['CI'][1]<=eps))
  distinct=[l for l,p in states(name) if l!='order-null']
  C={l:rows[(0,0,l)]['W'] for l in distinct}
  summaries.append(dict(gpu=gpu,dtype=dtype,config=name,arm=arm,mode=mode,
     primary=(arm=='auto' and mode=='graph'),same_minus_opposite_us=contrast('same','opposite',.95),
     state_median_us=medians,spearman_work_latency=spearman([C[l] for l in distinct],[medians[l] for l in distinct]),
     observed_median_half_range_us=(max(medians.values())-min(medians.values()))/2,
     order_null=null,equal_work=eq,plan_median_us=S.median(r['plan_us'] for r in rows.values()),
     synchronized_wall_median_us=S.median(r['wall_us'] for r in rows.values())))
 result=dict(complete=True,formal_runs=6,timing_rows=len(allrows),qualifications=len(validations),
             fp32_vectors=sum(v['fp32_vectors'] for v in validations),
             max_fp32_abs=max(v['fp32_max_abs'] for v in validations),
             graph_checks=sum(e['completion']['graph_exact_checks'] for e in evidence),
             full_model=False,serving_promotion=False,analysis_source_sha256=digest(__file__),
             evidence=evidence,cells=summaries)
 out.mkdir(parents=True);dump(out/'summary.json',result)
 text=['# Controlled permutation results','',
       'Kernel-only results; shape contrasts are not optimization speedups. All formal cells retained.',
       'Intervals are conditional on three process repeats and twelve blocks; no device-population or familywise claim.','',
       '|GPU|dtype|n|same-opposite, us [95% CI]|rho(W,time)|order-null equivalent|equal-W equivalent|',
       '|---|---|---|---|---|---|---|']
 for c in summaries:
  if not c['primary']:continue
  x=c['same_minus_opposite_us'];eq=c['equal_work']
  text.append(f"|{c['gpu']}|{c['dtype']}|{c['config']}|{x['mean']:.3f} [{x['CI'][0]:.3f}, {x['CI'][1]:.3f}]|{c['spearman_work_latency']:.3f}|{c['order_null']['equivalent']}|{eq['equivalent'] if eq else 'N/A'}|")
 (out/'RESULTS.md').write_text('\n'.join(text)+'\n')
 print(json.dumps({k:result[k] for k in ['formal_runs','timing_rows','qualifications','fp32_vectors','graph_checks','max_fp32_abs','full_model','serving_promotion']},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();analyze(a.root,a.out)
