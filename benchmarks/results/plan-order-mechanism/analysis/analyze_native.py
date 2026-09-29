"""Complete two-device native-integration pilot. No population significance claim."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from measure_order import cases
from plan_contract import canonical_hash,order_indices,validate_descriptors

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def selected_cases():
 cs=[c for c in cases('test') if c['case'] in ('discovery-n2','discovery-n4') and c['state'] in ('same','opposite')]
 return cs+[c for c in cases('test') if c['holdout'] and c['state']=='same']

def analyze(root,out):
 if out.exists():raise FileExistsError('preserve results')
 paths=sorted((root/'runs').glob('*'));paths=[p for p in paths if p.is_dir()]
 if len(paths)!=2:raise ValueError('two complete GPU pilot runs required')
 results={};sources=set();expected_cases=selected_cases();md=['# Native FA2 plan + run pilot','',
  'One process per GPU family. No inferential intervals or serving promotion.',
  'Times include actual native planning/sorting, metadata transfer and graph execution.',
  'Identity is the disabled path of the same experimental source, not a separate pristine build.','',
  '|GPU|set|actual calls per plan|identity/heavy-first cycle ratio|identity/A-A ratio|',
  '|---|---|---:|---:|---:|']
 for r in paths:
  if (r/'failure.json').exists() or not (r/'completed_utc.txt').exists():raise ValueError('incomplete/failed native run '+str(r))
  c=json.loads((r/'complete.json').read_text());e=json.loads((r/'environment.json').read_text())
  if c['complete'] is not True or e['cases']!=expected_cases:raise ValueError('changed native matrix')
  for f,h in c['files'].items():
   if Path(f).name!=f or sha(r/f)!=h:raise ValueError('bad native hash')
  sources.add(canonical_hash(e['source_binding']))
  rows=json.loads((r/'measurements.json').read_text());checks=json.loads((r/'qualification.json').read_text());plans=json.loads((r/'plans.json').read_text())
  case_map={(x['case'],x['state']):x for x in expected_cases}
  base_keys=set(itertools.product(case_map,('float16','bfloat16'),('auto','unsplit')))
  expected=set(itertools.product(base_keys,(1,36),range(6),('identity','identity_repeat','heavy_first')))
  index={}
  for row in rows:
   k=(((row['case'],row['state']),row['dtype'],row['split']),row['reuse'],row['block'],row['policy'])
   if k not in expected or k in index:raise ValueError('duplicate/unexpected native cell')
   if row['holdout'] is not case_map[k[0][0]]['holdout']:raise ValueError('changed heldout label')
   if row['actual_attention_calls']!=row['reuse']:raise ValueError('fake reuse count')
   if type(row['cycle_us']) not in (int,float) or not math.isfinite(row['cycle_us']) or row['cycle_us']<=0:raise ValueError('bad cycle time')
   index[k]=row
  if set(index)!=expected or c['rows']!=len(expected) or c['expected_rows']!=len(expected):raise ValueError('incomplete native rows')
  ph={};seen=set()
  for p in plans:
   key=((p['case'],p['state']),p['dtype'],p['split'])
   if key not in base_keys or key in seen:raise ValueError('bad plan coverage')
   seen.add(key);b=p['baseline'];qs=case_map[key[0]]['query'];ls=[q+k for q,k in zip(qs,case_map[key[0]]['cached'])]
   if b['query']!=qs or b['total_kv']!=ls:raise ValueError('changed plan shape')
   validate_descriptors(b['descriptors'],qs,ls,b['info'],b['chunk'],4,b['output_indptr'],b['merge_indptr'])
   order=order_indices(b['descriptors'],qs,ls,b['info']['cta_tile_q'],b['chunk'],bool(b['info']['split_kv']),4,'heavy_first')
   ph[(key,'heavy_first')]=canonical_hash([b['descriptors'][i] for i in order])
   ph[(key,'identity')]=ph[(key,'identity_repeat')]=canonical_hash(b['descriptors'])
   if p['heavy_descriptor_hash']!=ph[(key,'heavy_first')]:raise ValueError('policy divergence')
  if seen!=base_keys:raise ValueError('missing native plans')
  seen=set()
  for x in checks:
   key=(((x['case'],x['state']),x['dtype'],x['split']),x['policy'])
   if key not in ph or key in seen or x['descriptor_hash']!=ph[key]:raise ValueError('native source not qualified')
   if any(x[k] is not True for k in ('output_exact','lse_exact','graph_exact')):raise ValueError('native parity failure')
   seen.add(key)
  if seen!=set(ph) or c['qualifications']!=len(checks):raise ValueError('native qualification gap')
  compiled=json.loads((r/'compiled-libraries.json').read_text())
  if not compiled:raise ValueError('no source-compiled library evidence')
  summaries=[];cells=[]
  for key in sorted(base_keys):
   for reuse in (1,36):
    rr={p:math.exp(statistics.mean(math.log(index[(key,reuse,b,'identity')]['cycle_us']/index[(key,reuse,b,p)]['cycle_us']) for b in range(6))) for p in ('identity_repeat','heavy_first')}
    cells.append(dict(case=key[0][0],state=key[0][1],dtype=key[1],split=key[2],reuse=reuse,
      holdout=case_map[key[0]]['holdout'],ratios=rr,
      identity_median_us=statistics.median(index[(key,reuse,b,'identity')]['cycle_us'] for b in range(6)),
      heavy_median_us=statistics.median(index[(key,reuse,b,'heavy_first')]['cycle_us'] for b in range(6))))
  for name,hold in [('discovery',False),('holdout',True)]:
   for reuse in (1,36):
    group=[x for x in cells if x['holdout']==hold and x['reuse']==reuse]
    ratio=math.exp(statistics.mean(math.log(x['ratios']['heavy_first']) for x in group))
    aa=math.exp(statistics.mean(math.log(x['ratios']['identity_repeat']) for x in group))
    item=dict(set=name,reuse=reuse,ratio=ratio,AA_ratio=aa,cells=len(group),confidence_interval=None)
    summaries.append(item);md.append('|%s|%s|%s|%.6f|%.6f|'%(e['gpu'],name,reuse,ratio,aa))
  if e['gpu'] in results:raise ValueError('duplicate GPU pilot')
  results[e['gpu']]=dict(rows=len(rows),qualifications=len(checks),summaries=summaries,cells=cells,
    source_commit=(r/'source_commit.txt').read_text().strip(),environment=e,raw_hashes=c['files'],compiled_libraries=compiled)
 if len(sources)!=1 or not any('4090' in k for k in results) or not any('5090' in k for k in results):raise ValueError('native environment mismatch')
 report=dict(complete=True,GPUs=results,serving_promotion=False,full_model=False,statistical_superiority_claim=False,
   analysis_sha256=sha(__file__),scope='paired descriptive native planning+graph execution pilot; one process per GPU')
 out.mkdir(parents=True);(out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 (out/'RESULTS.md').write_text('\n'.join(md)+'\n');print('\n'.join(md),flush=True);return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();analyze(a.root,a.out)
