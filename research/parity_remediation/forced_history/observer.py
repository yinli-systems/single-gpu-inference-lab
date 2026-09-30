from __future__ import annotations
import hashlib,json,os,re
from pathlib import Path

def jhash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def thash(t):return hashlib.sha256(t.contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()
def flatten(x,prefix='0'):
 import torch
 if isinstance(x,torch.Tensor):yield prefix,x
 elif isinstance(x,(tuple,list)):
  for i,v in enumerate(x):yield from flatten(v,prefix+'.'+str(i))

def install_model(runner):
 if getattr(runner,'_sgi_forced_model_installed',False) or not hasattr(runner,'model'):return
 import torch
 runner._sgi_forced_model_installed=True;runner._sgi_forced_banks={};runner._sgi_forced_graph=False
 model=runner.model;selected={'layer':0,'qkv':0,'attn':0}
 def store(name,phase,values):
  for slot,t in flatten(values):
   if t.ndim<2 or not 1<=int(t.shape[0])<=16:continue
   key=('graph' if t.is_cuda and torch.cuda.is_current_stream_capturing() else 'eager',int(t.shape[0]))
   runner._sgi_forced_banks.setdefault(key,{})[name+'|'+phase+'|'+slot]=t.detach()
 for name,module in model.named_modules():
  if re.search(r'\.layers\.\d+$',name):
   selected['layer']+=1;module.register_forward_pre_hook(lambda m,a,n=name:store(n,'input',a));module.register_forward_hook(lambda m,a,o,n=name:store(n,'output',o))
  elif name.endswith('.self_attn.qkv_proj') and re.search(r'\.layers\.\d+\.',name):
   selected['qkv']+=1;module.register_forward_hook(lambda m,a,o,n=name:store(n,'output',o))
  elif name.endswith('.self_attn.attn') and re.search(r'\.layers\.\d+\.',name):
   selected['attn']+=1;module.register_forward_pre_hook(lambda m,a,n=name:store(n,'input',a));module.register_forward_hook(lambda m,a,o,n=name:store(n,'output',o))
 if selected!={'layer':36,'qkv':36,'attn':36}:raise RuntimeError('unreviewed 36-layer model structure '+repr(selected))
 runner._sgi_forced_selected=selected

def install():
 from sglang.srt.model_executor.model_runner import ModelRunner
 if getattr(ModelRunner,'_sgi_forced_class_installed',False):return
 ModelRunner._sgi_forced_class_installed=True
 load=ModelRunner.load_model
 def load_model(self,*a,**k):
  out=load(self,*a,**k);install_model(self);return out
 ModelRunner.load_model=load_model
 init=ModelRunner.init_cuda_graphs
 def init_graphs(self,*a,**k):install_model(self);return init(self,*a,**k)
 ModelRunner.init_cuda_graphs=init_graphs
 raw=ModelRunner._forward_raw
 def forward_raw(self,*a,**k):
  out=raw(self,*a,**k);self._sgi_forced_graph=bool(out.can_run_graph);return out
 ModelRunner._forward_raw=forward_raw
 old=ModelRunner.sample
 def sample(self,logits_output,forward_batch):
  import torch
  if not hasattr(self,'_sgi_forced_histories'):self._sgi_forced_histories={};self._sgi_forced_counts={}
  fb=forward_batch;bs=int(fb.batch_size);to_list=lambda x:x.detach().cpu().tolist() if hasattr(x,'detach') else list(x)
  inputs=to_list(fb.input_ids);seqs=to_list(fb.seq_lens)[:bs];pools=to_list(fb.req_pool_indices)[:bs]
  lens=[1]*bs if fb.forward_mode.is_decode() else list(fb.extend_seq_lens_cpu or [])
  histories=[];offset=0
  for pool,seq,n in zip(pools,seqs,lens):
   part=inputs[offset:offset+n];offset+=n;prev=self._sgi_forced_histories.get(pool)
   cur=list(part) if seq==n else (prev+part if prev is not None and len(prev)+n==seq else None)
   self._sgi_forced_histories[pool]=cur;histories.append(cur)
  cases=json.loads(Path(os.environ['SGI_FORCED_CASES']).read_text())
  target_by_hash={x['target_history_sha256']:x for x in cases['targets']}
  matches=[]
  for i,h in enumerate(histories):
   if h is not None and jhash(h) in target_by_hash:matches.append((i,target_by_hash[jhash(h)]))
  pending=[]
  for row,target in matches:
   count=self._sgi_forced_counts.get(target['id'],0)
   if count>=1:continue
   graph=bool(self._sgi_forced_graph);rows=int(self.decode_cuda_graph_runner.bs) if graph else bs
   bank=self._sgi_forced_banks.get(('graph' if graph else 'eager',rows))
   if bank is None:raise RuntimeError('graph/eager tensor bank missing')
   shadows={k:v[row:row+1].detach().cpu().contiguous() for k,v in bank.items()}
   page_tables={};logical_cache={};all_indices=[]
   for j,(pool,seq) in enumerate(zip(pools,seqs)):
    ind=self.req_to_token_pool.req_to_token[int(pool),:int(seq)].to(torch.long).detach().cpu().contiguous();page_tables[str(j)]=ind;all_indices.append(ind)
   target_indices=all_indices[row].to(self.device)
   for layer in range(36):
    kbuf,vbuf=self.token_to_kv_pool.get_kv_buffer(layer)
    logical_cache[f'{layer}.k']=kbuf.index_select(0,target_indices).detach().cpu().contiguous()
    logical_cache[f'{layer}.v']=vbuf.index_select(0,target_indices).detach().cpu().contiguous()
   raw_logits=logits_output.next_token_logits[row:row+1].detach().to('cpu',dtype=torch.float32).contiguous()
   meta=dict(target=target['id'],output_index_zero_based=target['output_index_zero_based'],history_sha256=target['target_history_sha256'],target_history_sha256=target['target_history_sha256'],
    seq_len=int(seqs[row]),row=row,batch_size=bs,graph_used=graph,graph_rows=rows,seq_lens=seqs,input_ids=inputs,
    history_sha256_all=[jhash(x) if x is not None else None for x in histories],batch_signature=jhash(dict(seq_lens=seqs,input_ids=inputs,histories=[jhash(x) if x is not None else None for x in histories],graph=graph,rows=rows)),
    shadow_order=sorted(shadows),shadow_sha256={k:thash(v) for k,v in shadows.items()},cache_sha256={k:thash(v) for k,v in logical_cache.items()},
    page_table_sha256={k:thash(v) for k,v in page_tables.items()},raw_logits_sha256=thash(raw_logits),diagnostic_only=True,no_graph_clone_inserted=True)
   pending.append((target,count,dict(metadata=meta,shadows=shadows,cache=logical_cache,page_tables=page_tables,raw_logits=raw_logits)))
  result=old(self,logits_output,forward_batch);sampled=to_list(result)[:bs]
  root=Path(os.environ['SGI_FORCED_TRACE']);root.mkdir(parents=True,exist_ok=True)
  for target,count,payload in pending:
   payload['metadata']['sampled_id']=int(sampled[payload['metadata']['row']]);stem=f"{target['id']}-occ{count}"
   torch.save(payload,root/(stem+'.pt'));(root/(stem+'.json')).write_text(json.dumps(payload['metadata'],indent=2)+'\n');self._sgi_forced_counts[target['id']]=count+1
  return result
 ModelRunner.sample=sample
