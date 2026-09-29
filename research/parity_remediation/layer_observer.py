"""Read-only shadow tensors for bounded, already-exposed decode witnesses.
Graph copies alter resource/timing behavior. Never consume these times as benchmarks.
"""
from __future__ import annotations
import hashlib,json,os,re
from pathlib import Path

def tensor_hash(t):
    return hashlib.sha256(t.contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()

def install_model(runner):
    if getattr(runner,'_sgi_layers_installed',False):return
    if not hasattr(runner,'model'):return
    import torch
    model=runner.model;runner._sgi_shadow_banks={};runner._sgi_shadow_mode=None
    runner._sgi_layer_snapshots=0;runner._sgi_layers_installed=True
    forward=model.forward
    def model_forward(*args,**kwargs):
        fb=kwargs.get('forward_batch',args[2] if len(args)>2 else None)
        if fb is None:raise RuntimeError('unrecognized model forward contract')
        runner._sgi_shadow_mode='decode' if fb.forward_mode.is_decode() else 'other'
        return forward(*args,**kwargs)
    model.forward=model_forward
    selected=[]
    suffixes=('input_layernorm','self_attn.qkv_proj','self_attn.attn','self_attn.o_proj',
              'post_attention_layernorm','mlp.gate_up_proj','mlp.down_proj')
    def flatten(obj,prefix='0'):
        if isinstance(obj,torch.Tensor):yield prefix,obj
        elif isinstance(obj,(tuple,list)):
            for i,x in enumerate(obj):yield from flatten(x,prefix+'.'+str(i))
    def store(name,phase,values):
        if runner._sgi_shadow_mode!='decode':return
        for slot,t in flatten(values):
            if t.ndim<2 or not 1<=t.shape[0]<=16:continue
            captured=t.is_cuda and torch.cuda.is_current_stream_capturing()
            key=('graph' if captured else 'eager',int(t.shape[0]))
            bank=runner._sgi_shadow_banks.setdefault(key,{})
            bank[name+'|'+phase+'|'+slot]=t.detach().clone()
    for name,module in model.named_modules():
        if re.search(r'\.layers\.\d+\.',name) and name.endswith(suffixes):
            selected.append(name)
            module.register_forward_pre_hook(lambda m,args,name=name:store(name,'input',args))
            module.register_forward_hook(lambda m,args,out,name=name:store(name,'output',out))
    runner._sgi_shadow_modules=selected
    layer_ids={int(re.search(r'\.layers\.(\d+)\.',n).group(1)) for n in selected}
    if len(layer_ids)!=36 or len(selected)!=36*len(suffixes):raise RuntimeError('this observer supports the pinned 36-layer Qwen operator structure only')

def snapshot(runner,fb,record,logits):
    if not fb.forward_mode.is_decode():return
    if not getattr(runner,'_sgi_layers_installed',False):raise RuntimeError('layer observer not installed')
    import torch
    targets=json.loads(Path(os.environ['SGI_LAYER_TARGETS']).read_text())
    matches=[i for i,h in enumerate(record['history_sha256']) if h in targets]
    if not matches:return
    if runner._sgi_layer_snapshots>=8:return
    if not record['all_histories_complete']:raise RuntimeError('incomplete history at layer witness')
    graph=bool(getattr(runner,'_sgi_last_forward_graph',False))
    rows=int(runner.decode_cuda_graph_runner.bs) if graph else int(fb.input_ids.shape[0])
    key=('graph' if graph else 'eager',rows);bank=runner._sgi_shadow_banks.get(key)
    if not bank:raise RuntimeError('matching graph shadow bank unavailable')
    base=Path(os.environ['SGI_TRACE_PATH'])/'layers';base.mkdir(exist_ok=True)
    for i in matches:
        if runner._sgi_layer_snapshots>=8:break
        h=record['history_sha256'][i];seq=int(record['seq_lens'][i]);pool=int(record['req_pool_indices'][i])
        if seq>256:raise RuntimeError('witness prefix out of bound')
        indices=runner.req_to_token_pool.req_to_token[pool,:seq].to(dtype=torch.long)
        tensors={name:t[i:i+1].detach().cpu().contiguous() for name,t in bank.items()}
        present={int(re.search(r'\.layers\.(\d+)\.',n).group(1)) for n in tensors}
        if len(present)!=36:raise RuntimeError('incomplete layer shadows')
        cache={}
        for layer in range(36):
            k,v=runner.token_to_kv_pool.get_kv_buffer(layer)
            if k.ndim!=3 or tuple(k.shape[1:])!=(8,128):raise RuntimeError('unreviewed KV buffer layout')
            if int(indices.min())<0 or int(indices.max())>=k.shape[0]:raise RuntimeError('invalid physical KV indices')
            cache[f'{layer}.k']=k.index_select(0,indices).detach().cpu().contiguous()
            cache[f'{layer}.v']=v.index_select(0,indices).detach().cpu().contiguous()
        meta=dict(history_sha256=h,target=targets[h],step=record['step'],batch_signature=record['batch_signature'],
                  batch_size=record['batch_size'],row=i,seq_len=seq,graph_used=graph,graph_rows=rows,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  shadow_sha256={k:tensor_hash(v) for k,v in tensors.items()},
                  cache_sha256={k:tensor_hash(v) for k,v in cache.items()},
                  logits_sha256=tensor_hash(logits[i:i+1]),capture_is_diagnostic=True)
        stem=f'step-{record["step"]:06d}-row-{i}-{h[:12]}'
        torch.save(dict(metadata=meta,shadows=tensors,cache=cache,physical_indices=indices.cpu(),logits=logits[i:i+1]),base/(stem+'.pt'))
        (base/(stem+'.json')).write_text(json.dumps(meta,indent=2)+'\n')
        runner._sgi_layer_snapshots+=1

def install_runner_hooks():
    from sglang.srt.model_executor.model_runner import ModelRunner
    if getattr(ModelRunner,'_sgi_layer_class_patched',False):return
    ModelRunner._sgi_layer_class_patched=True
    load=ModelRunner.load_model
    def load_model(self,*args,**kwargs):
        result=load(self,*args,**kwargs);install_model(self);return result
    ModelRunner.load_model=load_model
    raw=ModelRunner._forward_raw
    def forward_raw(self,*args,**kwargs):
        result=raw(self,*args,**kwargs);self._sgi_last_forward_graph=bool(result.can_run_graph);return result
    ModelRunner._forward_raw=forward_raw
    init=ModelRunner.init_cuda_graphs
    def init_graphs(self,*args,**kwargs):
        install_model(self);return init(self,*args,**kwargs)
    ModelRunner.init_cuda_graphs=init_graphs
