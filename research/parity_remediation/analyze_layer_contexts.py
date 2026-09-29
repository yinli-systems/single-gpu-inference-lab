"""Exposed cross-context diagnostics, deliberately NOT a matched-arm causal estimate."""
import argparse,hashlib,json
from functools import lru_cache
from pathlib import Path
from analyze_layers import position

@lru_cache(maxsize=2)
def load(path):
 import torch
 return torch.load(path,map_location='cpu',weights_only=True)

def stats(a,b):
 if a.shape!=b.shape:return dict(shape_match=False,shapes=[list(a.shape),list(b.shape)])
 d=(a.float()-b.float()).abs()
 return dict(shape_match=True,equal=bool((d==0).all()),max_abs=float(d.max()),different_elements=int((d!=0).sum()))

def run(root,out):
 if out.exists():raise FileExistsError('preserve context diagnosis')
 # Full tensor-file hashes must already have passed strict layer-audit.
 audit=json.loads((root.parents[1]/'layer-audit/summary.json').read_text())
 rows={}
 for mode in ('pristine-0','cap-0'):
  unique={}
  for p in sorted((root/mode/'trace/layers').glob('*.json')):
   m=json.loads(p.read_text());binary=p.with_suffix('.pt');relative=str(binary.relative_to(root))
   if hashlib.sha256(binary.read_bytes()).hexdigest()!=audit['file_hashes'][relative]:raise ValueError('snapshot changed after strict audit')
   key=json.dumps({k:m[k] for k in ('history_sha256','graph_rows','row','shadow_sha256','cache_sha256','logits_sha256')},sort_keys=True)
   unique.setdefault(key,(p,m))
  rows[mode]=list(unique.values())
 results=[]
 for ap,a in rows['pristine-0']:
  for bp,b in rows['cap-0']:
   if a['history_sha256']!=b['history_sha256']:continue
   ad=load(ap.with_suffix('.pt'));bd=load(bp.with_suffix('.pt'))
   changed=[k for k in sorted(a['shadow_sha256'],key=position) if a['shadow_sha256'][k]!=b['shadow_sha256'][k]]
   keys=[k for k in a['shadow_sha256'] if k.startswith('model.layers.0.self_attn.attn|input|')]
   cache={}
   for kind in ('k','v'):
    x,y=ad['cache']['0.'+kind],bd['cache']['0.'+kind]
    cache[kind]=dict(prefix=stats(x[:-1],y[:-1]),current=stats(x[-1:],y[-1:]))
   r=dict(target=a['target'],pristine_snapshot=ap.name,cap_snapshot=bp.name,
     same_complete_batch_signature=a['batch_signature']==b['batch_signature'],
     graph_rows=[a['graph_rows'],b['graph_rows']],rows=[a['row'],b['row']],
     first_observed_shadow_difference=changed[0] if changed else None,
     layer0_current_attention_inputs={k:stats(ad['shadows'][k],bd['shadows'][k]) for k in keys},
     layer0_KV=cache,logits=stats(ad['logits'],bd['logits']),
     first_difference_origin_not_attributed=True,diagnostic_only=True)
   if changed:r['first_shadow_delta']=stats(ad['shadows'][changed[0]],bd['shadows'][changed[0]])
   results.append(r)
 out.mkdir(parents=True);(out/'summary.json').write_text(json.dumps(dict(comparisons=results,matching_history_only=True,not_a_fair_performance_comparison=True,old_failure_resolved=False),indent=2)+'\n')
 for r in results:print(json.dumps(r,sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.job_root,a.out)
