"""Complete source-verified residency diagnostic; not a native-serving A/B."""
import argparse,collections,hashlib,itertools,json,math,statistics
from pathlib import Path
from analyze import require,sha,weights,estimate

def analyze(root,out):
 require(not out.exists(),'preserve results')
 paths=sorted((root/'residency-runs').glob('gpu_*-r*-*'))
 require(len(paths)==6,'six completed diagnostic runs required')
 groups=collections.defaultdict(list)
 comps=('repeat48','heavy48','native64','heavy64')
 regimes=('graph_one_warm','graph32_steady','graph_one_pressure128MiB')
 for p in paths:
  require((p/'launcher-complete.txt').exists() and not (p/'failure.json').exists(),'incomplete run '+str(p))
  c=json.loads((p/'complete.json').read_text());e=json.loads((p/'environment.json').read_text())
  require(c['complete'] is True and c['rows']==576 and c['checks']==5,'bad completion')
  for n,h in c['files'].items():require(Path(n).name==n and sha(p/n)==h,'raw hash mismatch')
  qs=json.loads((p/'qualification.json').read_text());qm={q['arm']:q for q in qs}
  require(set(qm)=={'native48',*comps} and len(qs)==5 and all(q['exact'] is True for q in qs),'qualification coverage')
  require(qm['native48']['descriptor_hash']==qm['repeat48']['descriptor_hash']==qm['native64']['descriptor_hash'],'native order changed')
  require(qm['heavy48']['descriptor_hash']==qm['heavy64']['descriptor_hash'],'heavy order changed')
  launch=json.loads((p/'launch-proof.json').read_text());lm={l['arm']:l for l in launch}
  require(set(lm)=={'native48','heavy48','native64','heavy64'} and len(launch)==4,'launch proof coverage')
  require(len({(l['name'],l['args']['registers per thread']) for l in launch})==1,'device kernel changed')
  for l in launch:
   require(l['args']['grid']==[34,1,8] and l['args']['block']==[32,4,1],'launch geometry mismatch')
   require(l['args']['shared memory']==(65536 if l['arm'].endswith('64') else 49152),'launch smem mismatch')
  rows=json.loads((p/'measurements.json').read_text());idx={};expected=set(itertools.product(range(12),comps,regimes,range(4)))
  for x in rows:
   k=(x['block'],x['comparison'],x['regime'],x['position'])
   require(k in expected and k not in idx and x['rep']==e['rep'],'bad timing identity')
   seq=('native48',x['comparison'],x['comparison'],'native48') if x['block']%2==0 else (x['comparison'],'native48','native48',x['comparison'])
   require(x['arm']==seq[x['position']] and x['descriptor_hash']==qm[x['arm']]['descriptor_hash'],'timing arm/hash mismatch')
   require(x['calls']==(32 if x['regime']=='graph32_steady' else 1),'call count')
   for f in ('device_us','wall_us','setup_us'):require(math.isfinite(x[f]) and x[f]>0,'bad timing')
   idx[k]=x
  require(set(idx)==expected,'missing timing cells')
  hw=e['hardware']['out'].strip().splitlines()[-1].split(',')
  groups[e['gpu']].append(dict(path=p,env=e,index=idx,launch=launch,uuid=hw[1].strip(),driver=hw[2].strip(),receipt=sha(p/'complete.json')))
 require(len(groups)==2,'missing GPU family')
 w=weights();report=dict(complete=True,rows=3456,qualified_records=30,all_cases_exposed=True,full_model=False,
  default_promotion=False,serving_promotion=False,analysis_sha256=sha(__file__),GPUs={})
 md=['# Residency-pressure diagnostic','', 'Unprofiled paired CUDA-event measurements. One exposed BF16/unsplit/D128 shape. Native is the disabled path of the same experimental host source, not a pristine full-engine comparator.','',
 '|GPU|Regime|Native48/heavy48|Native48/native64|Native48/heavy64|Native64/heavy64|A/A resolves 1%|','|---|---|---|---|---|---|---|']
 for gpu,rs in sorted(groups.items()):
  rs.sort(key=lambda x:x['env']['rep']);require([x['env']['rep'] for x in rs]==[0,1,2],'repeat coverage')
  require(len({x['uuid'] for x in rs})==len({x['driver'] for x in rs})==1,'same-device context violated')
  require(len({x['env']['source_sha256'] for x in rs})==1,'mixed measurement code')
  summaries=[]
  for reg in regimes:
   grids={}
   for pol in comps:
    grid=[]
    for run in rs:
     vals=[]
     for b in range(12):
      xs=[run['index'][(b,pol,reg,k)] for k in range(4)]
      a=[math.log(x['device_us']) for x in xs if x['arm']=='native48'];z=[math.log(x['device_us']) for x in xs if x['arm']!= 'native48']
      require(len(a)==len(z)==2,'unbalanced block');vals.append(statistics.mean(a)-statistics.mean(z))
     grid.append(vals)
    grids[pol]=grid
   stats={k:estimate(v,w,.9 if k=='repeat48' else .95) for k,v in grids.items()}
   intr=[[grids['heavy64'][i][j]-grids['native64'][i][j] for j in range(12)] for i in range(3)]
   stats['order_ratio_at64']=estimate(intr,w)
   interaction=[[intr[i][j]-grids['heavy48'][i][j] for j in range(12)] for i in range(3)]
   stats['ratio_of_order_ratios']=estimate(interaction,w)
   aa=stats['repeat48'];clean=aa['CI'][0]>=1/1.005 and aa['CI'][1]<=1.005
   summaries.append(dict(regime=reg,stats=stats,AA_resolves_one_percent=clean))
   def fmt(k):
    t=stats[k];return '%.5f [%.5f, %.5f]'%(t['ratio'],*t['CI'])
   md.append('|%s|%s|%s|%s|%s|%s|%s|'%(gpu,reg,fmt('heavy48'),fmt('native64'),fmt('heavy64'),fmt('order_ratio_at64'),clean))
  report['GPUs'][gpu]=dict(uuid=rs[0]['uuid'],driver=rs[0]['driver'],summaries=summaries,launch_proofs=[r['launch'] for r in rs],receipts={r['path'].name:r['receipt'] for r in rs})
 md+=['','Shared-memory padding changes a resource-residency constraint and may also change cache/resource behavior. It does not directly measure CTA-to-SM issue order. Removal of a harmful reorder is not automatically a speedup over native48. D/E require a separate independently qualified remedy, pristine lifecycle A/B and supported callers.']
 out.mkdir(parents=True);(out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(out/'RESULTS.md').write_text('\n'.join(md)+'\n');print('\n'.join(md));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();analyze(a.root,a.out)
