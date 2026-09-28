"""Qualification adapter, not a production integration or an optimized backend.

Only decisions are cached. Every public plan call rebuilds page-address metadata.
Uses a frozen pure C ABI so Python3.13 need not load a Python3.12 extension.
"""
from __future__ import annotations
import ctypes as ct
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
from contracts import POLICIES,PolicyCache,bounded_policy


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrozenCPolicy:
    def __init__(self,path):
        path=Path(path);meta=json.loads((path/'manifest.json').read_text())
        if meta['model_selection_sha256']!='d41c87f9e2f1c8eeac8ab411f63925e0eb2781f0229f8d7974ae6b6463a81f1c':raise ValueError('wrong frozen model')
        if digest(path/'libselector.so')!=meta['library_sha256']:raise ValueError('native binary changed')
        self.models=meta['models'];self.lib=ct.CDLL(str(path/'libselector.so'))
        ptr=ct.POINTER(ct.c_int64);out=ct.POINTER(ct.c_double)
        self.lib.select_policy.argtypes=[ct.c_int,ct.c_int,ptr,ptr,ct.c_int,ct.c_int,ct.c_int,out]
        self.lib.select_policy.restype=ct.c_int
    def choose(self,q,k,hq,hkv,sms):
        name='NVIDIA GeForce RTX 4090|median_cycle_ms|causal'
        arr=ct.c_int64*len(q);scores=(ct.c_double*6)()
        index=self.lib.select_policy(self.models[name],len(q),arr(*q),arr(*k),hq,hkv,sms,scores)
        if not 0<=index<6:raise ValueError('native metadata rejected')
        return POLICIES[index]


class PagedRouter:
    def __init__(self,native_path,fixed='none',budget=512*1024**2):
        self.native=FrozenCPolicy(native_path);self.fixed=fixed;self.budget=budget
        self.arm='auto';self.cache=PolicyCache(256);self.counts=Counter();self.records=[]
        self._original_plan=None
    def reset(self,arm):
        if arm not in ('auto','fixed_budget','native_cycle'):raise ValueError('unknown arm')
        self.arm=arm;self.cache=PolicyCache(256);self.counts=Counter();self.records=[]
    def install(self):
        import flashinfer
        if flashinfer.__version__!='0.6.18':raise ValueError('pinned0.6.18 required')
        klass=flashinfer.BatchPrefillWithPagedKVCacheWrapper
        if getattr(klass,'_sgi_reuse_qualified_adapter',False):raise RuntimeError('avoid nested patch')
        original_init=klass.__init__;original_plan=klass.plan
        self._original_plan=original_plan;signature=inspect.signature(original_plan);router=self
        def init(w,*args,**kwargs):
            if kwargs.get('backend','auto') not in ('auto','fa2'):raise ValueError('qualification requires explicitFA2')
            kwargs['backend']='fa2'
            original_init(w,*args,**kwargs)
        def plan(w,*args,**kwargs):
            import torch
            bound=signature.bind(w,*args,**kwargs);bound.apply_defaults();d=bound.arguments
            router.counts['public_plan_calls']+=1
            supported=(not w.is_cuda_graph_enabled and d.get('causal',False) and
                d.get('head_dim_qk')==128 and d.get('head_dim_vo') in (None,128) and
                d.get('num_qo_heads')==32 and d.get('num_kv_heads')==8 and
                d.get('window_left',-1)==-1 and d.get('pos_encoding_mode','NONE')=='NONE' and
                d.get('custom_mask') is None and d.get('packed_custom_mask') is None and
                d.get('logits_soft_cap') in (None,0,0.) and
                d.get('q_data_type') in (torch.float16,'float16') and
                d.get('kv_data_type') in (torch.float16,'float16',None))
            if not supported:
                router.counts['unsupported_stock']+=1
                return original_plan(w,*args,**kwargs)
            qp=d['qo_indptr'];ip=d['paged_kv_indptr'];lp=d['paged_kv_last_page_len'];page=int(d['page_size'])
            # CPU metadata is already provided by vLLM; synchronize if a caller uses device metadata.
            q=[int(x) for x in (qp[1:]-qp[:-1]).cpu().tolist()]
            total=[int(x) for x in ((ip[1:]-ip[:-1]-1)*page+lp).cpu().tolist()]
            k=[L-x for L,x in zip(total,q)]
            sms=torch.cuda.get_device_properties(qp.device if qp.is_cuda else torch.cuda.current_device()).multi_processor_count
            if sms!=128 or not q or len(q)>64 or min(q)<=0 or min(k)<0:
                router.counts['unsupported_geometry_stock']+=1
                return original_plan(w,*args,**kwargs)
            actual_budget=min(router.budget,w._float_workspace_buffer.numel()*w._float_workspace_buffer.element_size())
            key=(tuple(q),tuple(k),page,32,8,sms,actual_budget,str(w._kv_layout),str(d['q_data_type']),str(d.get('kv_data_type')),bool(d['causal']),float(d.get('sm_scale') or 0),router.arm)
            def choose():
                p='auto' if router.arm=='auto' else router.fixed if router.arm=='fixed_budget' else router.native.choose(q,k,32,8,sms)
                return bounded_policy(tuple(q),tuple(k),32,8,sms,p,actual_budget)
            mode=router.cache.resolve(key,choose)
            changed=dict(d);changed.pop('self',None)
            # fixed_split_size is in pages on the paged wrapper, tokens on ragged.
            if mode in ('auto','none'):changed['fixed_split_size']=None
            else:
                tokens=int(mode[1:])
                if tokens%page:raise ValueError('nonintegral split-page conversion')
                changed['fixed_split_size']=tokens//page
            changed['disable_split_kv']=(mode=='none')
            try:result=original_plan(w,**changed)
            except RuntimeError as exc:
                if 'Buffer overflow when allocating memory' not in str(exc):raise
                router.counts['allocator_fallback']+=1
                changed['fixed_split_size']=None;changed['disable_split_kv']=False
                result=original_plan(w,**changed);mode='auto'
            router.counts['qualified_plans']+=1;router.counts['policy_'+mode]+=1
            router.records.append(dict(q=q,k=k,page_size=page,mode=mode,arm=router.arm,
                scratch_bytes=actual_budget,layout=str(w._kv_layout),plan_info=[int(x) for x in w._plan_info]))
            return result
        klass.__init__=init;klass.plan=plan;klass._sgi_reuse_qualified_adapter=True
        return self
    def report(self):
        return dict(arm=self.arm,counts=dict(self.counts),cache_hits=self.cache.hits,
                    cache_misses=self.cache.misses,cache_evictions=self.cache.evictions,plans=self.records)
