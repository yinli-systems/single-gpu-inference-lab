"""Fail-closed per-stage analysis. No model quality is inferred from kernel timings."""
from __future__ import annotations
import argparse,csv,hashlib,io,itertools,json,math,re,sys
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
import numpy as np
import release_gate as release_gate_module
from release_gate import evaluate_release_gate,subset_gate_inputs

def require(c,m):
 if not c:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def safe_rel(root,value):
 p=Path(value)
 require(not p.is_absolute() and '..' not in p.parts,'unsafe recovery path')
 return root/p

def json_digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def hardware_identity(env):
 out=env.get('hardware',{}).get('out','')
 rows=list(csv.DictReader(io.StringIO(out)))
 require(len(rows)==1,'single visible GPU required')
 row={str(k).strip():str(v).strip() for k,v in rows[0].items()}
 required={'name','uuid','driver_version','power.limit [W]'}
 require(required<=set(row) and all(row[k] for k in required),'incomplete hardware identity')
 require(row['name']==env.get('gpu'),'hardware/environment GPU mismatch')
 return dict(name=row['name'],uuid=row['uuid'],driver=row['driver_version'],power_limit_w=row['power.limit [W]'])

def deployment_environment_identity(env):
 required=('partition','case_hash','gpu','num_sm','cuda','torch','flashinfer','official_overlay_sha256',
  'source_archive_sha256','profiling_mode','profiled','nsight_compute_excluded','clocks_locked',
  'whole_node_exclusive','full_model','output_seed')
 require(all(k in env for k in required),'incomplete deployment environment identity')
 hw=hardware_identity(env)
 return dict(partition=env['partition'],case_hash=env['case_hash'],gpu=env['gpu'],num_sm=int(env['num_sm']),
  driver=hw['driver'],power_limit_w=hw['power_limit_w'],cuda=env['cuda'],torch=env['torch'],
  flashinfer=env['flashinfer'],official_overlay_sha256=env['official_overlay_sha256'],
  source_archive_sha256=env['source_archive_sha256'],profiling_mode=env['profiling_mode'],
  profiled=env['profiled'],nsight_compute_excluded=env['nsight_compute_excluded'],
  clocks_locked=env['clocks_locked'],whole_node_exclusive=env['whole_node_exclusive'],
  full_model=env['full_model'],output_seed=env['output_seed'])

def mode_build_identity(env):
 require(env.get('mode') in {'pristine','off','cap','guarded'},'unknown mode identity')
 require(isinstance(env.get('header_sha256'),str) and len(env['header_sha256'])==64,'missing header identity')
 require(isinstance(env.get('source'),dict) and env['source'],'missing measurement source identity')
 return dict(mode=env['mode'],header_sha256=env['header_sha256'],measurement_source_digest=json_digest(env['source']))

def freshness_sensitivity_amendment(root,measurement_commit,measured_cases):
 p=root/'receipts'/'freshness-sensitivity-amendment-v2.json'
 require(p.is_file(),'strict-freshness amendment missing')
 x=json.loads(p.read_text())
 require(x.get('schema')==2 and x.get('measurement_source_commit')==measurement_commit,'wrong freshness amendment identity')
 require(x.get('selector_rule_changed') is False and x.get('measurement_manifest_changed') is False,'freshness amendment changed frozen experiment')
 require(x.get('original_30_case_primary_gate_unchanged') is True,'primary gate was not preserved')
 require(x.get('registered_before_any_release_performance_ratio_analysis') is True and x.get('canary_performance_used') is False,'post-performance freshness amendment')
 excluded=set(x.get('secondary_strict_freshness_sensitivity_excludes',[]))
 require(excluded and excluded<set(measured_cases),'invalid strict-freshness exclusion')
 require(int(x.get('strictly_fresh_relative_to_canary_case_count',-1))==len(set(measured_cases)-excluded),'fresh case count mismatch')
 return dict(path=p.relative_to(root).as_posix(),sha256=sha(p),excluded_cases=sorted(excluded),receipt=x)

def verify_complete(path,expected_sha=None):
 require((path/'complete.json').exists() and (path/'launcher-complete.txt').exists(),'completed evidence missing')
 if expected_sha is not None:require(sha(path/'complete.json')==expected_sha,'complete receipt changed')
 c=json.loads((path/'complete.json').read_text());require(c.get('complete') is True,'completion flag missing')
 for f,h in c.get('files',{}).items():
  require(Path(f).name==f and (path/f).is_file() and sha(path/f)==h,'completed evidence hash mismatch')
 return c

def recovery_map(root,stage,gpu):
 final=root/'receipts'/'recovery-map-final.json';first=root/'receipts'/'recovery-map.json'
 p=final if final.exists() else first
 if not p.exists():return {},dict(map=None,retained_incomplete_runs=[],superseded_complete_runs=[],unscored_recovery_attempts=[]),set()
 x=json.loads(p.read_text());require(x.get('stage')==stage and x.get('gpu')==gpu,'wrong recovery map')
 require(x.get('measurement_source_commit'),'missing recovery source binding')
 schema=x.get('schema');require(schema in (1,2),'unsupported recovery schema')
 if schema==2:
  require(first.exists() and sha(first)==x['prior_map_sha256'],'prior recovery map changed')
 skipped={};replacement_paths=set();info=dict(map=p.relative_to(root).as_posix(),map_sha256=sha(p),
  retained_incomplete_runs=[],superseded_complete_runs=[],unscored_recovery_attempts=[])
 for e in x.get('entries',[]):
  old=safe_rel(root,e['superseded']);new=safe_rel(root,e['replacement'])
  require(old.is_dir() and new.is_dir(),'missing recovery directory')
  env=json.loads((old/'environment.json').read_text());key=(env['shard'],env['rep'],env['mode'])
  require(key==(e['key']['shard'],e['key']['rep'],e['key']['mode']),'recovery key mismatch')
  kind=e.get('superseded_kind','partial')
  if kind=='partial':
   require(not (old/'complete.json').exists() and not (old/'failure.json').exists(),'superseded partial changed state')
   require((old/'progress.json').exists(),'partial progress missing')
   require(sha(old/'environment.json')==e['old_environment_sha256'] and sha(old/'progress.json')==e['old_progress_sha256'],'partial evidence changed')
   info['retained_incomplete_runs'].append(dict(path=e['superseded'],progress=json.loads((old/'progress.json').read_text()),reason=e['reason'],old_job=e['old_job'],old_state=e['old_state'],replacement=e['replacement']))
  elif kind=='complete':
   verify_complete(old,e['old_complete_sha256'])
   info['superseded_complete_runs'].append(dict(path=e['superseded'],reason=e['reason'],old_job=e['old_job'],replacement=e['replacement']))
  else:raise ValueError('unknown superseded kind')
  verify_complete(new,e['replacement_complete_sha256'])
  newenv=json.loads((new/'environment.json').read_text());require((newenv['shard'],newenv['rep'],newenv['mode'])==key,'replacement identity mismatch')
  rel=old.relative_to(root).as_posix();require(rel not in skipped,'duplicate superseded path')
  skipped[rel]=e;replacement_paths.add(new.relative_to(root).as_posix())
 for e in x.get('unscored_attempts',[]):
  q=safe_rel(root,e['path']);verify_complete(q,e['complete_sha256'])
  rel=q.relative_to(root).as_posix();require(rel not in skipped,'duplicate unscored path');skipped[rel]=e
  info['unscored_recovery_attempts'].append(dict(path=rel,job=e['job'],reason=e['reason'],hardware=json.loads((q/'environment.json').read_text())['hardware']['out'].strip()))
 require(x.get('entries'),'empty recovery map')
 return skipped,info,replacement_paths

def load(root,stage,gpu):
 sys.path.insert(0,str(root/'source'));from manifest import load,digest
 m=load();runs=[];seen=set();source=set();physical=defaultdict(set);common_environment=set();mode_builds=defaultdict(set);device_uuids=defaultdict(set)
 skipped,recovery,replacement_paths=recovery_map(root,stage,gpu);loaded_paths=set()
 for path in sorted((root/'runs').glob(stage+'-'+gpu+'-*')):
  if not path.is_dir():continue
  rel=path.relative_to(root).as_posix()
  if rel in skipped:continue
  require(not (path/'failure.json').exists(),'failure retained '+str(path))
  require((path/'complete.json').exists() and (path/'launcher-complete.txt').exists(),'incomplete run '+str(path))
  c=json.loads((path/'complete.json').read_text());e=json.loads((path/'environment.json').read_text())
  require(c['complete'] is True and e['case_hash']==m['case_hash'],'wrong campaign')
  require(e.get('partition')==gpu,'partition/GPU-family evidence mismatch')
  expected_name={'gpu_4090':'NVIDIA GeForce RTX 4090','gpu_5090':'NVIDIA GeForce RTX 5090'}[gpu]
  require(e.get('gpu')==expected_name and e.get('profiled') is False and e.get('nsight_compute_excluded') is True,'wrong GPU or profiled timing evidence')
  require(bool(e.get('official_overlay_sha256')) and bool(e.get('source_archive_sha256')),'missing provenance identity')
  for f,h in c['files'].items():require(Path(f).name==f and sha(path/f)==h,'hash mismatch '+f)
  key=(e['shard'],e['rep'],e['mode']);require(key not in seen,'duplicate run');seen.add(key)
  source.add(digest(e['source']))
  hw=hardware_identity(e);physical[e['shard']].add(e['hardware']['out'].strip().splitlines()[1].strip())
  common_environment.add(json.dumps(deployment_environment_identity(e),sort_keys=True,separators=(',',':')))
  mode_builds[e['mode']].add(json.dumps(mode_build_identity(e),sort_keys=True,separators=(',',':')))
  device_uuids[e['shard']].add(hw['uuid'])
  wanted=('canary',) if stage=='canary' else ('release',)
  allcases=[x for x in m['cases'] if x['family'] in wanted]
  expectedcases=[x for i,x in enumerate(allcases) if i%e['shards']==e['shard']]
  require(e['cases']==expectedcases,'changed cases/shard mapping')
  blocks=2 if stage=='canary' else m['blocks']
  basic=set(itertools.product([x['id'] for x in expectedcases],m['dtypes'],m['layouts'],m['splits']))
  expected=set(itertools.product(basic,range(blocks),(0,1,16),range(4)));rows=json.loads((path/'measurements.json').read_text());index={}
  for row in rows:
   k=((row['case'],row['dtype'],row['layout'],row['split']),row['block'],row['calls'],row['position'])
   require(k in expected and k not in index,'unexpected/duplicate timing row')
   seq=('main','repeat','repeat','main') if row['block']%2==0 else ('repeat','main','main','repeat')
   require(row['label']==seq[row['position']] and row['mode']==e['mode'] and row['rep']==e['rep'],'timing identity mismatch')
   expected_execution={0:'eager',1:'graph1',16:'graph16'}[row['calls']]
   require(row.get('execution_mode')==expected_execution and isinstance(row.get('tactic_identity'),str) and len(row['tactic_identity'])==64,'execution/tactic identity mismatch')
   require(all(isinstance(row[z],(float,int)) and math.isfinite(row[z]) and row[z]>0 for z in ('run_device_us','run_wall_us','cycle_us')),'invalid timing')
   index[k]=row
  require(set(index)==expected and len(rows)==c['rows']==c['expected'],'matrix incomplete')
  quals=json.loads((path/'qualification.json').read_text());qmap={}
  case_by_id={case['id']:case for case in expectedcases}
  for q in quals:
   k=(q['case'],q['dtype'],q['layout'],q['split']);require(k in basic and k not in qmap,'qualification coverage')
   require(q['eager_graph_exact'] is True,'graph parity missing')
   require(q['pristine_exact'] is True and q['pristine_full_max_abs']==0,'nonexact resource-only output')
   require(q['FP32']['vectors']>0 and all(math.isfinite(q['FP32'][z]) for z in ['max_abs','rmse','lse_max_abs']),'bad FP32 receipt')
   require(set(q.get('tactic_identities',{}))=={'eager','graph1','graph16'} and len(set(q['tactic_identities'].values()))==3,'incomplete/non-distinct tactic identities')
   if e['mode']=='guarded':
    require(q['resource_cap_plan']==q['guard_expected'],'guard decision mismatch')
    if stage=='canary':
     require(q['guard_expected'] is bool(case_by_id[q['case']]['expected_selector'][gpu]),'canary selector boundary mismatch')
   if e['mode']=='off': require(q['resource_cap_plan'] is False,'off plan enabled cap')
   if e['mode']=='cap': require(q['resource_cap_plan'] is True,'cap plan disabled')
   qmap[k]=q
  require(set(qmap)==basic and len(quals)==c['qualifications'],'incomplete qualification')
  for (basic_key,block,calls,pos),row in index.items():
   require(row['tactic_identity']==qmap[basic_key]['tactic_identities'][{0:'eager',1:'graph1',16:'graph16'}[calls]],'timing/qualification tactic identity drift')
  runs.append(dict(path=path,env=e,index=index,qual=qmap));loaded_paths.add(rel)
 require(replacement_paths<=loaded_paths,'declared replacement not loaded')
 require(runs and len(source)==1,'no runs or mixed source')
 shards=runs[0]['env']['shards'];reps=1 if stage=='canary' else 3
 require(seen==set(itertools.product(range(shards),range(reps),m['modes'])),'incomplete declared mode/repeat/shard group')
 require(all(len(v)==1 for v in physical.values()),'within-shard hardware/driver changed')
 require(len(common_environment)==1,'deployment environment changed across shards')
 require(set(mode_builds)==set(m['modes']) and all(len(v)==1 for v in mode_builds.values()),'mode build identity changed across shards')
 require(all(len(v)==1 for v in device_uuids.values()),'within-shard GPU UUID changed')
 environment_receipt=dict(common=json.loads(next(iter(common_environment))),
  mode_builds={mode:json.loads(next(iter(values))) for mode,values in sorted(mode_builds.items())},
  gpu_uuids={str(shard):sorted(values) for shard,values in sorted(device_uuids.items())})
 by=defaultdict(dict)
 for r in runs:
  for k,q in r['qual'].items():by[k][(r['env']['rep'],r['env']['mode'])]=q
 for k,values in by.items():
  require(len({x['inputs_sha256'] for x in values.values()})==1,'cross-mode input mismatch')
  for rep in range(reps):
   base=values[(rep,'pristine')]
   for mode in ['off','cap','guarded']:
    q=values[(rep,mode)]
    require(q['out_sha256']==base['out_sha256'] and q['lse_sha256']==base['lse_sha256'],'cross-mode hash parity failure')
 return m,runs,physical,recovery,environment_receipt

@lru_cache(None)
def weights(shard,arm):
 seed=936612+shard*101+{'pristine':0,'off':7,'cap':13,'guarded':29}[arm]
 rng=np.random.default_rng(seed);w=np.zeros((10000,24),dtype=float)
 for d in range(10000):
  for p in rng.integers(0,3,size=3):
   for b in rng.integers(0,8,size=8):w[d,p*8+b]+=1
 return w/24

def run(a):
 require(not a.out.exists(),'preserve analysis output')
 require(re.fullmatch(r'[0-9a-f]{40}',a.analysis_commit or '') is not None,'invalid analysis commit')
 m,rs,physical,recovery,environment=load(a.root,a.stage,a.gpu)
 measurement_commit=(a.root/'source'/'SOURCE_COMMIT.txt').read_text().strip()
 require(re.fullmatch(r'[0-9a-f]{40}',measurement_commit) is not None,'invalid measurement commit')
 receipt=dict(stage=a.stage,gpu=a.gpu,complete=True,processes=len(rs),timing_rows=sum(len(r['index']) for r in rs),
  qualifications=sum(len(r['qual']) for r in rs),source_hashes=rs[0]['env']['source'],manifest_hash=m['case_hash'],
  measurement_source_commit=measurement_commit,analysis_source_commit=a.analysis_commit,
  analysis_source_sha256={'analyze.py':sha(Path(__file__)),'release_gate.py':sha(Path(release_gate_module.__file__))},
  deployment_environment=environment,hardware={str(k):sorted(v) for k,v in physical.items()},
  recovery_evidence=recovery,default_promotion=False,serving_promotion=False)
 receipt['numerics']={mode:dict(qualifications=sum(len(r['qual']) for r in rs if r['env']['mode']==mode),
  exact_full_outputs=sum(q['pristine_exact'] for r in rs if r['env']['mode']==mode for q in r['qual'].values()),
  max_abs_vs_pristine=max(q['pristine_full_max_abs'] for r in rs if r['env']['mode']==mode for q in r['qual'].values()),
  FP32_max_abs=max(q['FP32']['max_abs'] for r in rs if r['env']['mode']==mode for q in r['qual'].values()),
  FP32_vectors=sum(q['FP32']['vectors'] for r in rs if r['env']['mode']==mode for q in r['qual'].values())) for mode in m['modes']}
 md=['# Selector-v3 result','',a.stage+' / '+a.gpu,'']
 if a.stage!='canary':
  lookup={(r['env']['shard'],r['env']['rep'],r['env']['mode']):r for r in rs};cells=[];drawcache={}
  for shard in range(rs[0]['env']['shards']):
   r=lookup[(shard,0,'pristine')]
   for k in r['qual']:
    for calls,metric in itertools.product([0,1,16],['run_device_us','cycle_us']):
     grids={};aa={}
     for mode in m['modes']:
      grids[mode]=np.array([[np.mean([lookup[(shard,rep,mode)]['index'][(k,b,calls,pos)][metric] for pos in range(4) if lookup[(shard,rep,mode)]['index'][(k,b,calls,pos)]['label']=='main']) for b in range(8)] for rep in range(3)])
      repeated=np.array([[np.mean([lookup[(shard,rep,mode)]['index'][(k,b,calls,pos)][metric] for pos in range(4) if lookup[(shard,rep,mode)]['index'][(k,b,calls,pos)]['label']=='repeat']) for b in range(8)] for rep in range(3)])
      draws=weights(shard,mode)@np.log(grids[mode]/repeated).reshape(24);lo,hi=np.exp(np.quantile(draws,[.05,.95]))
      aa[mode]=dict(CI90=[float(lo),float(hi)],resolves1pct=bool(lo>=1/1.005 and hi<=1.005))
     guard_flags={lookup[(shard,rep,'guarded')]['qual'][k]['guard_expected'] for rep in range(3)}
     require(len(guard_flags)==1,'guard decision changed across repeats')
     row=dict(case=k[0],dtype=k[1],layout=k[2],split=k[3],calls=calls,metric=metric,shard=shard,
      selected=bool(next(iter(guard_flags))),AA=aa,comparisons={})
     for mode in ['off','cap','guarded']:
      bs=np.log(grids['pristine']).reshape(24);cs=np.log(grids[mode]).reshape(24)
      draws=weights(shard,'pristine')@bs-weights(shard,mode)@cs
      ratio=float(np.exp(bs.mean()-cs.mean()));ci=[float(x) for x in np.exp(np.quantile(draws,[.025,.975]))]
      row['comparisons'][mode]=dict(ratio=ratio,CI95=ci,controls_resolve=aa['pristine']['resolves1pct'] and aa[mode]['resolves1pct'])
      drawcache[(len(cells),mode)]=draws
     cells.append(row)
  summaries=[]
  for layout,calls,metric,mode in itertools.product(m['layouts'],[0,1,16],['run_device_us','cycle_us'],['off','cap','guarded']):
   indices=[i for i,c in enumerate(cells) if (c['layout'],c['calls'],c['metric'])==(layout,calls,metric)]
   selected=[cells[i] for i in indices];draws=np.mean([drawcache[(i,mode)] for i in indices],axis=0)
   point=float(np.exp(np.mean([math.log(c['comparisons'][mode]['ratio']) for c in selected])))
   summaries.append(dict(layout=layout,calls=calls,metric=metric,mode=mode,ratio=point,CI95=[float(x) for x in np.exp(np.quantile(draws,[.025,.975]))],
    cells=len(selected),controls_failed=sum(not c['comparisons'][mode]['controls_resolve'] for c in selected),
    worst_ratio=min(c['comparisons'][mode]['ratio'] for c in selected),point_regressions_gt1pct=sum(c['comparisons'][mode]['ratio']<1/1.01 for c in selected)))
  gate=evaluate_release_gate(cells,drawcache,receipt['numerics'])
  require(m.get('strict_freshness_extension') is True,'extension analyzer used with non-extension manifest')
  receipt.update(cells=cells,summaries=summaries,extension_candidate_gate=gate,
   extension_candidate_gate_pass=gate['pass'],single_gpu_release_gate_pass=False,
   statistical_scope='Separately preregistered two-case strict-freshness extension, conditional on measured devices. It cannot replace or promote the original 30-case primary and is not a hardware-population guarantee.')
  md+=['**Strict-freshness extension candidate gate: %s**'%('PASS' if gate['pass'] else 'HOLD'),'',
   '- This extension is non-promotional: `single_gpu_release_gate_pass=false` regardless of its standalone result.',
   '- Selected Graph16 cells: %s; geomean %.6f; 95%% CI [%.6f, %.6f]; point worst %.6f; joint-min LCB %.6f.'%(
    gate['selected']['count'],gate['selected']['ratio'],*gate['selected']['CI95'],gate['selected']['worst_point_ratio'],gate['selected']['simultaneous_worst_CI95'][0]),
   '- Requirements: `'+json.dumps(gate['requirements'],sort_keys=True)+'`','',
   '|layout|calls|metric|mode|pristine/candidate [95% CI]|worst|unresolved controls|','|---|---:|---|---|---|---:|---:|']
  for s in summaries:md.append('|%s|%s|%s|%s|%.6f [%.6f,%.6f]|%.6f|%s|'%(s['layout'],s['calls'],s['metric'],s['mode'],s['ratio'],*s['CI95'],s['worst_ratio'],s['controls_failed']))
 a.out.mkdir(parents=True);(a.out/'summary.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');(a.out/'RESULTS.md').write_text('\n'.join(md)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ['cells','source_hashes']},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['canary','release'],required=True);p.add_argument('--gpu',choices=['gpu_4090','gpu_5090'],required=True);p.add_argument('--analysis-commit',required=True);run(p.parse_args())
