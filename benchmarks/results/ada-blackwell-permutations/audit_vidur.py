"""Audit actual pinned Vidur feature-construction methods, not trained accuracy."""
from __future__ import annotations
import argparse,ast,hashlib,json,math
from pathlib import Path
from types import SimpleNamespace
from campaign import CONFIGS,states,geometry

PIN='abae7f63aa857300f5cdc6f5e0d27860cd24721b'
BLOB='a5a96466eb86d94503711afec6d45218bd38d93e'
METHODS=('_get_batch_prefill_attention_params','_get_attention_prefill_execution_time')

def audit(path):
 content=Path(path).read_bytes()
 blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
 if blob!=BLOB:raise ValueError('unreviewed upstream source; expected pinned Git blob')
 tree=ast.parse(content);selected=[];ranges={}
 for node in ast.walk(tree):
  if isinstance(node,ast.FunctionDef) and node.name in METHODS:
   ranges[node.name]=[node.lineno,node.end_lineno];node.returns=None
   for arg in node.args.args:arg.annotation=None
   selected.append(node)
 if len(selected)!=2:raise ValueError('missing upstream methods')
 # Only the two hash-verified methods execute. No module imports, initialization,
 # model training, arbitrary remote top-level code or simulator execution.
 ns={};module=ast.Module(body=selected,type_ignores=[])
 exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)
 class Capture:
  def __init__(self):self.keys=[]
  def __getitem__(self,key):self.keys.append(key);return 0.0
 class Probe:pass
 for name in METHODS:setattr(Probe,name,ns[name])
 results=[]
 for granularity in (64,128,256):
  for name in CONFIGS:
   captured=[]
   for label,pairing in states(name):
    g=geometry(name,label,pairing);probe=Probe();lookup=Capture()
    probe._config=SimpleNamespace(kv_cache_prediction_granularity=granularity)
    probe._attention_prefill_batching_overhead_fraction=0.
    probe._predictions={'attn_prefill':lookup}
    b=SimpleNamespace(requests=[SimpleNamespace(_is_prefill_complete=False,num_processed_tokens=k) for k in g['cached']],num_tokens=g['query'])
    probe._get_attention_prefill_execution_time(b)
    if len(lookup.keys)!=1:raise ValueError('unexpected number of prefill lookups')
    key=lookup.keys[0]
    expected=(sum(((k+granularity-1)//granularity)*granularity for k in g['cached']),round(math.sqrt(sum(q*q for q in g['query'])))**2)
    if key!=expected:raise AssertionError('reference key differs from real upstream code')
    captured.append(dict(state=label,key=list(key),W=g['W']))
   keys={tuple(x['key']) for x in captured};work={x['W'] for x in captured}
   results.append(dict(config=name,granularity=granularity,state_calls=len(captured),
     unique_lookup_keys=len(keys),distinct_work_values=len(work),all_states_alias=len(keys)==1,states=captured))
 return dict(scope='actual upstream methods with a key-capturing lookup; NOT a trained Vidur performance benchmark',
   repo='microsoft/vidur',commit=PIN,path='vidur/execution_time_predictor/sklearn_execution_time_predictor.py',
   git_blob=blob,source_sha256=hashlib.sha256(content).hexdigest(),method_lines=ranges,
   calls=sum(r['state_calls'] for r in results),all_reference_keys_match=True,rows=results,
   limitation='Conditional on this pinned prefill lookup and configured granularities. Not a claim about every Vidur fork/version, whole simulator error, or attained speedup.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('new output path required')
 result=audit(a.source);a.out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ['commit','git_blob','method_lines','calls','all_reference_keys_match']},indent=2))
