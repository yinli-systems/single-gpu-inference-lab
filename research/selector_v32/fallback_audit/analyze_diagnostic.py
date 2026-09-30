"""Audit every diagnostic sample. No trimming, replacement, or release promotion."""
from __future__ import annotations
import argparse,collections,hashlib,json,math,csv,io
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def geometric(values):return math.exp(sum(math.log(x) for x in values)/len(values))
def aggregate_groups(rows):
 groups=collections.defaultdict(list)
 for r in rows:
  if r['positive_control']:continue
  key=tuple(r[k] for k in ('case','dtype','layout','split','scenario','timer','execution'))
  groups[key].append(r)
 result=[];rng=np.random.default_rng(774991)
 for key,rr in groups.items():
  grid={}
  for r in rr:
   coord=(r['rep'],r['block'],r['position'])
   if coord in grid:raise ValueError('duplicate observation')
   expected=('off','guarded','guarded','off') if r['block']%2==0 else ('guarded','off','off','guarded')
   if r['arm']!=expected[r['position']]:raise ValueError('treatment order mismatch')
   if not math.isfinite(r['wall_us']) or r['wall_us']<=0:raise ValueError('bad timing')
   grid[coord]=r
  if set(grid)!={(p,b,z) for p in range(3) for b in range(8) for z in range(4)}:raise ValueError('incomplete diagnostic cell')
  ratios=np.empty((3,8));device_ratios=np.empty((3,8));devices=all(r['device_us'] is not None for r in rr)
  for p in range(3):
   for b in range(8):
    block=[grid[p,b,z] for z in range(4)];off=[r for r in block if r['arm']=='off'];guard=[r for r in block if r['arm']=='guarded']
    ratios[p,b]=math.log(geometric([r['wall_us'] for r in off])/geometric([r['wall_us'] for r in guard]))
    if devices:device_ratios[p,b]=math.log(geometric([r['device_us'] for r in off])/geometric([r['device_us'] for r in guard]))
  draws=[]
  for _ in range(4000):
   procs=rng.integers(0,3,3);draws.append(np.mean([ratios[p,rng.integers(0,8,8)].mean() for p in procs]))
  ci=np.exp(np.quantile(draws,[.025,.975])).tolist();gaps=[r['wall_minus_device_us'] for r in rr if r['wall_minus_device_us'] is not None]
  result.append(dict(zip(('case','dtype','layout','split','scenario','timer','execution'),key),paired_wall_ratio=float(np.exp(ratios.mean())),conditional_CI95=ci,paired_device_ratio=float(np.exp(device_ratios.mean())) if devices else None,host_gap_p99_us=float(np.quantile(gaps,.99)) if gaps else None,host_gap_max_us=max(gaps) if gaps else None,host_span_separation_count=sum(r['host_span_separation'] for r in rr),rows=len(rr),point_worst_block=float(np.exp(ratios.min())),not_release_evidence=True))
 return result

def run(root,out):
 root=Path(root);out=Path(out)
 if out.exists():raise FileExistsError('preserve earlier analysis')
 results={};inputs={}
 for gpu in ('gpu_4090','gpu_5090'):
  rows=[];hws=[];qual_count=0;graph_mismatches=0;positive=[]
  for rep in range(3):
   candidates=list((root/'runs').glob(f'{gpu}-r{rep}-*'))
   if len(candidates)!=1:raise ValueError('missing/duplicate process '+gpu+' '+str(rep))
   p=candidates[0];c=json.loads((p/'complete.json').read_text())
   for f,h in c['files'].items():
    if Path(f).name!=f or sha(p/f)!=h:raise ValueError('artifact hash '+f)
   e=json.loads((p/'environment.json').read_text());hws.append(e['hardware']['out']);rr=json.loads((p/'rows.json').read_text());qq=json.loads((p/'qualification.json').read_text())
   if len(rr)!=c['rows'] or len(qq)!=5 or c['noninjected_rows']!=2880:raise ValueError('matrix size')
   if e['rep']!=rep or e['partition']!=gpu or e['release_evidence'] is not False:raise ValueError('process identity')
   inputs[str(p)]=dict(complete_sha256=sha(p/'complete.json'),environment_sha256=sha(p/'environment.json'))
   rows.extend(rr);positive.extend([r for r in rr if r['positive_control']]);qual_count+=len(qq)
   for q in qq:
    for name,counts in q['graphs'].items():
     for count,arms in counts.items():
      if name=='same_storage' and arms['off']['pointers']!=arms['guarded']['pointers']:raise ValueError('storage identity')
      if arms['guarded']['plan'][-1]==0 and arms['off']['signature']!=arms['guarded']['signature']:graph_mismatches+=1
  if len(set(hws))!=1:raise ValueError('physical GPU/driver drift across repetitions')
  if len(positive)!=15 or not all(r['host_span_separation'] for r in positive):raise ValueError('positive control failure')
  results[gpu]=dict(hardware=hws[0],noninjected_rows=len(rows)-len(positive),positive_controls=len(positive),qualified_case_repeats=qual_count,fallback_graph_metadata_mismatches=graph_mismatches,groups=aggregate_groups(rows))
 out.mkdir(parents=True);summary=dict(results=results,inputs=inputs,release_gate_pass=False,historical_token_divergence_resolved=False,all_failed_canaries_retained=True)
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 md=['# Fallback identity and host-tail diagnosis','','Diagnostic only. Every non-injected sample is retained; old canary HOLD is unchanged.','', '| GPU | Scenario | Timer | Worst paired wall point across five exposed coordinates / two graph modes |','|---|---|---|---:|']
 for gpu,x in results.items():
  for scenario in ('independent','same_storage','same_graph_null'):
   for timer in ('legacy_events','pooled_events','wall_only'):
    a=[g for g in x['groups'] if g['scenario']==scenario and g['timer']==timer]
    md.append(f"| {gpu} | {scenario} | {timer} | {min(g['paired_wall_ratio'] for g in a):.6f}x |")
 md+=['','Same-graph null comparisons are not candidate wins. Conditional bootstrap intervals use three process repeats and eight blocks; they do not establish population-wide safety. No fresh release or full HTTP result is produced.']
 (out/'RESULTS.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.root,a.out)
