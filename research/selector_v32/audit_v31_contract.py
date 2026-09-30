"""Recompute exposed v3 deployment-boundary point summaries from a v3 summary.json."""
import argparse,json,math
from pathlib import Path
DEP={(0,'cycle_us'):'eager_full_call',(1,'run_device_us'):'graph1_replay',(16,'run_device_us'):'graph16_replay'}
def run(p):
 x=json.loads(Path(p).read_text());cells=[c for c in x['cells'] if c['selected'] and (c['calls'],c['metric']) in DEP]
 out={}
 for arm in ('guarded','cap'):
  ratios=[c['comparisons'][arm]['ratio'] for c in cells];by={}
  for c in cells:by.setdefault(DEP[(c['calls'],c['metric'])],[]).append(c['comparisons'][arm]['ratio'])
  out[arm]={'cells':len(ratios),'geomean':math.exp(sum(map(math.log,ratios))/len(ratios)),'worst':min(ratios),'execution':{k:{'cells':len(v),'geomean':math.exp(sum(map(math.log,v))/len(v)),'worst':min(v)} for k,v in by.items()}}
 print(json.dumps(out,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('summary');a=p.parse_args();run(a.summary)
