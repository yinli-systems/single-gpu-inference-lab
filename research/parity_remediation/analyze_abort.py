"""Audit explicit old-task/new-state cancellation against source-bound cleanup fix."""
import argparse,json
from pathlib import Path
from evidence_contract import read_verified

def audit(root):
 out=dict(modes={},historical_token_divergence_fixed=False,production_promoted=False)
 for mode in ('original','patched'):
  r=root/mode;env=read_verified(r/'environment.json');files=sorted(r.glob('reused-*.json'))+sorted(r.glob('unique-*.json'))
  if len(files)!=8:raise ValueError('incomplete fixed request matrix')
  rows=[read_verified(f) for f in files]
  events=[json.loads(x) for x in (r/'cleanup.jsonl').read_text().splitlines()]
  registrations={(e['rid'],e['object_id']) for e in events if e['event']=='register'}
  bad=[]
  for e in events:
   if e['event']=='abort' and not e.get('abort_all') and e['cleanup_owner'] is not None and e['target_object'] is not None and e['cleanup_owner']!=e['target_object']:
    if (e['rid'],e['target_object']) not in registrations:raise ValueError('abort target not registered')
    bad.append(e)
  out['modes'][mode]=dict(requests=len(rows),complete_requests=sum(x['complete'] for x in rows),
      reused=[{k:x[k] for k in ('wire_id','received_tokens','finish_reason','complete')} for x in rows if x['condition']=='reused-rid'],
      unique_complete=sum(x['complete'] for x in rows if x['condition']=='unique-rid'),
      stale_cleanup_aborts=bad)
 o=out['modes']['original'];f=out['modes']['patched']
 out['bounded_reproduction_supported']=bool(o['stale_cleanup_aborts']) and o['complete_requests']<8 and o['unique_complete']==4
 out['bounded_fix_supported']=not f['stale_cleanup_aborts'] and f['complete_requests']==8
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('preserve evidence')
 d=audit(a.job_root);a.out.mkdir(parents=True);(a.out/'summary.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
