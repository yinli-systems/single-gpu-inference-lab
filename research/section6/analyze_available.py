"""Audit every completed run; score only preregistered complete three-repeat groups.
Incomplete runs/groups are retained and explicitly unscored, never imputed.
"""
import argparse,collections,itertools,json
from pathlib import Path
from analyze import validate_run,require,weights,estimate,paired,sha
from measure import CASES,POLICIES,REGIMES

def run(root,out):
 require(not out.exists(),'preserve results')
 paths=sorted((root/'runs').glob('gpu_*-0.*-r*-*'))
 groups=collections.defaultdict(list);unfinished=[];complete=[]
 for p in paths:
  if not (p/'launcher-complete.txt').exists() or not (p/'complete.json').exists():
   unfinished.append(dict(run=p.name,progress=json.loads((p/'progress.json').read_text()) if (p/'progress.json').exists() else None,
    failure=json.loads((p/'failure.json').read_text()) if (p/'failure.json').exists() else None));continue
  r=validate_run(p);complete.append(r);groups[(r['env']['gpu'],r['env']['version'])].append(r)
 w=weights();report=dict(full_matrix_complete=len(complete)==12,completed_runs=len(complete),expected_runs=12,
  retained_timing_rows=sum(len(r['rows']) for r in complete),retained_qualifications=sum(len(r['checks']) for r in complete),
  retained_plans=sum(len(r['plans']) for r in complete),unfinished=unfinished,groups=[],
  analysis_sha256=sha(__file__),default_promotion=False,serving_promotion=False,
  scope='exposed diagnostics; complete three-repeat subgroups only; incomplete subgroups unscored, all completed raw data retained')
 md=['# Section-six available-evidence audit','',f'Completed runs: {len(complete)}/12. Incomplete subgroups are NOT given confirmatory intervals. All data and failures are retained.','',
 '|GPU|Version|Regime|Both no-op labels resolve 1%|Cells|','|---|---|---|---:|---:|']
 for (gpu,ver),rs in sorted(groups.items()):
  rs.sort(key=lambda x:x['env']['rep'])
  binding=[dict(run=r['path'].name,rep=r['env']['rep'],uuid=r['uuid'],driver=r['driver'],receipt=r['receipt']) for r in rs]
  if [r['env']['rep'] for r in rs]!=[0,1,2]:
   report['groups'].append(dict(gpu=gpu,version=ver,scored=False,reason='missing planned repeat',receipts=binding));continue
  require(len({r['uuid'] for r in rs})==len({r['driver'] for r in rs})==1,'different physical context')
  require(len({json.dumps(r['env']['source'],sort_keys=True) for r in rs})==1,'mixed measurement source')
  cells=[]
  for cell,reg in itertools.product(itertools.product([x[0] for x in CASES],('float16','bfloat16'),('auto','unsplit')),REGIMES):
   ps={pol:estimate(paired(rs,cell,pol,reg),w,.9 if pol.startswith('identity') else .95) for pol in POLICIES}
   clean=all(ps[pol]['CI'][0]>=1/1.005 and ps[pol]['CI'][1]<=1.005 for pol in ('identity_label','identity_copy'))
   cells.append(dict(case=cell[0],dtype=cell[1],split=cell[2],regime=reg,controls_resolve_one_percent=clean,policies=ps))
  counts=[]
  for reg in REGIMES:
   xs=[c for c in cells if c['regime']==reg];n=sum(x['controls_resolve_one_percent'] for x in xs)
   counts.append(dict(regime=reg,pass_count=n,total=len(xs)));md.append(f'|{gpu}|{ver}|{reg}|{n}|{len(xs)}|')
  report['groups'].append(dict(gpu=gpu,version=ver,scored=True,receipts=binding,controls=counts,cells=cells))
 md+=['','## Two-request reversal (BF16 / unsplit)','', '|GPU|Version|Regime|Native/causal-heavy [95% interval]|1% control resolution|','|---|---|---|---|---|']
 for g in report['groups']:
  if not g['scored']:continue
  for c in g['cells']:
   if (c['case'],c['dtype'],c['split'])!=('reversal2','bfloat16','unsplit'):continue
   s=c['policies']['causal_heavy'];md.append('|%s|%s|%s|%.6f [%.6f, %.6f]|%s|'%(g['gpu'],g['version'],c['regime'],s['ratio'],*s['CI'],c['controls_resolve_one_percent']))
 md+=['','Single-call and graph32 are different reuse/cache regimes. Intervals are conditional on this exposed suite and not adjusted for multiple comparisons. No current source, clock lock, or engine-wide performance guarantee follows from a passing cell.']
 out.mkdir(parents=True);(out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(out/'RESULTS.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.root,a.out)
