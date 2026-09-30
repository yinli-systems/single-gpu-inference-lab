from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path

def require(x,msg):
 if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor_delta(a,b):
 import torch
 if tuple(a.shape)!=tuple(b.shape) or a.dtype!=b.dtype:return dict(shape_equal=False,dtype_equal=a.dtype==b.dtype,equal=False,max_abs=None,different_elements=None)
 eq=torch.equal(a,b)
 if a.is_floating_point():
  delta=(a.to(torch.float32)-b.to(torch.float32)).abs();mx=float(delta.max()) if delta.numel() else 0.
 else:mx=0. if eq else None
 return dict(shape_equal=True,dtype_equal=True,equal=eq,max_abs=mx,different_elements=int((a!=b).sum()))
def layer_of(key):
 m=re.search(r'\.layers\.(\d+)',key);return int(m.group(1)) if m else 10**9
def kind_of(key):
 if re.search(r'\.layers\.\d+\|input\|',key):return 0
 if '.self_attn.qkv_proj|output|' in key:return 1
 if '.self_attn.attn|input|' in key:return 2
 if '.self_attn.attn|output|' in key:return 4
 if re.search(r'\.layers\.\d+\|output\|',key):return 5
 return 3
def ordered_shadow_keys(keys):return sorted(keys,key=lambda k:(layer_of(k),kind_of(k),k))
def read_run(path,cases_sha):
 require(not (path/'failure.json').exists(),'failed run '+str(path));c=json.loads((path/'complete.json').read_text())
 require(c.get('complete') and c.get('cases_sha256')==cases_sha,'wrong completion '+str(path))
 for f,h in c['files'].items():require((path/f).is_file() and sha(path/f)==h,'run hash '+str(path/f))
 for f,h in c['captures'].items():require((path/'trace'/f).is_file() and sha(path/'trace'/f)==h,'capture hash '+str(path/'trace'/f))
 env=json.loads((path/'environment.json').read_text());require(env['cases_sha256']==cases_sha and env['diagnostic_only'] is True,'wrong environment')
 return c,env
def request_result(path,target):
 x=json.loads((path/(target+'.json')).read_text());require(x['target']==target and not x.get('errors'),'wrong target result')
 rows={r['id']:r for r in x['requests']};require(set(rows)=={f'decode-{i:02d}' for i in range(16)},'request coverage')
 return rows[target]
def load_capture(path,target):
 import torch
 meta=json.loads((path/'trace'/(target+'-occ0.json')).read_text());payload=torch.load(path/'trace'/(target+'-occ0.pt'),map_location='cpu',weights_only=False)
 require(payload['metadata']==meta,'metadata payload mismatch')
 require(meta.get('history_sha256')==meta.get('target_history_sha256'),'target history metadata')
 require(set(meta['shadow_sha256'])==set(payload['shadows']) and set(meta['cache_sha256'])==set(payload['cache']),'tensor inventory')
 for k,t in payload['shadows'].items():require(sha_tensor(t)==meta['shadow_sha256'][k],'shadow hash '+k)
 for k,t in payload['cache'].items():require(sha_tensor(t)==meta['cache_sha256'][k],'cache hash '+k)
 require(sha_tensor(payload['raw_logits'])==meta['raw_logits_sha256'],'logit hash')
 return meta,payload
def sha_tensor(t):return hashlib.sha256(t.contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()
def compare_pair(pa,pb):
 ma,a=pa;mb,b=pb
 if ma['batch_signature']!=mb['batch_signature']:
  return dict(matched_execution=False,pristine_signature=ma['batch_signature'],cap_signature=mb['batch_signature'])
 require(ma['seq_lens']==mb['seq_lens'] and ma['input_ids']==mb['input_ids'] and ma['history_sha256_all']==mb['history_sha256_all'],'signature collision')
 require(set(a['shadows'])==set(b['shadows']) and set(a['cache'])==set(b['cache']),'tensor inventory mismatch')
 cache_diffs=[]
 for k in sorted(a['cache'],key=lambda x:(int(x.split('.')[0]),x)):
  q=tensor_delta(a['cache'][k],b['cache'][k])
  if not q['equal']:cache_diffs.append(dict(key=k,**q))
 shadow_diffs=[]
 for k in ordered_shadow_keys(a['shadows']):
  q=tensor_delta(a['shadows'][k],b['shadows'][k])
  if not q['equal']:shadow_diffs.append(dict(key=k,**q))
 logits=tensor_delta(a['raw_logits'],b['raw_logits'])
 first_state=None
 if cache_diffs:
  first_state=dict(kind='historical_kv',key=cache_diffs[0]['key'],delta=cache_diffs[0])
 elif shadow_diffs:
  first_state=dict(kind='current_operator_boundary',key=shadow_diffs[0]['key'],delta=shadow_diffs[0])
 elif not logits['equal']:first_state=dict(kind='post_model_logits',key='raw_logits',delta=logits)
 return dict(matched_execution=True,logical_histories_equal=True,physical_page_tables_equal=all(__import__('torch').equal(a['page_tables'][k],b['page_tables'][k]) for k in a['page_tables']),
  logical_kv_equal=not cache_diffs,shadow_tensors_equal=not shadow_diffs,raw_logits=logits,first_observed_difference=first_state,
  cache_difference_count=len(cache_diffs),shadow_difference_count=len(shadow_diffs))
def run(a):
 require(not a.out.exists(),'preserve analysis');cases=json.loads(a.cases.read_text());cases_sha=sha(a.cases)
 require(cases['schema']==1 and len(cases['targets'])==2,'case contract');root=a.root/'runs'/str(a.job)
 expected=['pristine-0','cap-0','cap-1','pristine-1'];require(root.is_dir(),'job root')
 runs={}
 for name in expected:
  path=root/name;c,e=read_run(path,cases_sha);runs[name]=dict(path=path,complete=c,environment=e)
 report=dict(complete=True,job=a.job,cases_sha256=cases_sha,diagnostic_only=True,performance_claim=False,targets=[])
 any_old=False;causal=False
 for target in cases['targets']:
  tid=target['id'];results={name:request_result(x['path'],tid) for name,x in runs.items()};captures={name:load_capture(x['path'],tid) for name,x in runs.items()}
  for meta,_ in captures.values():require(meta['history_sha256']==target['target_history_sha256'] and meta['seq_len']==target['seq_len'],'captured target mismatch')
  outputs={name:r['natural_token'] for name,r in results.items()};old=target['historical_tokens']
  historical_reproduced=any(outputs[p]==old[0] and outputs[c]==old[1] and outputs[p]!=outputs[c] for p in ('pristine-0','pristine-1') for c in ('cap-0','cap-1'))
  pairs=[]
  for p in ('pristine-0','pristine-1'):
   for c in ('cap-0','cap-1'):
    q=compare_pair(captures[p],captures[c]);q.update(pristine=p,cap=c,pristine_token=outputs[p],cap_token=outputs[c]);pairs.append(q)
  matched=[x for x in pairs if x['matched_execution']];attributed=any(x.get('first_observed_difference') is not None for x in matched)
  any_old|=historical_reproduced;causal|=attributed
  report['targets'].append(dict(id=tid,output_index_zero_based=target['output_index_zero_based'],historical_tokens=old,natural_tokens=outputs,
   historical_divergence_reproduced=historical_reproduced,matched_pair_count=len(matched),first_state_attributed=attributed,pairs=pairs))
 report.update(original_failure_reproduced=any_old,first_state_attributed=causal,serving_promoted=False,default_promoted=False)
 a.out.mkdir(parents=True);(a.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='targets'},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--job',type=int,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
