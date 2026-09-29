"""Audited FA2 work-descriptor permutations. No Torch dependency in pure contracts.

This is an opt-in experimental benchmark adapter, not a live-engine cache.
The descriptor bijection preserves every original output destination.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import itertools
import json

FIELDS = ('padded_batch_size','total_num_rows','total_num_rows_offset','cta_tile_q',
          'request_indices_offset','qo_tile_indices_offset','kv_tile_indices_offset',
          'merge_indptr_offset','o_indptr_offset','kv_chunk_size_ptr_offset',
          'v_offset','s_offset','block_valid_mask_offset','enable_cuda_graph','split_kv')
POLICIES = ('identity','identity_repeat','request_reverse','heavy_first','interleave')
PREFILL_SHA = '743953a8d301d89c6943dc48a8f6b393c0d636b71b165ae1842c2c65c7defaab'

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def positive_int(x, name, allow_zero=False):
    if type(x) is not int or x < (0 if allow_zero else 1):
        raise ValueError('invalid '+name)
    return x

def decode_info(vector):
    if len(vector)!=len(FIELDS) or any(type(v) is not int for v in vector):
        raise ValueError('unsupported PrefillPlanInfo ABI')
    info=dict(zip(FIELDS,vector))
    if info['enable_cuda_graph']!=0:
        raise ValueError('dynamic/padded graph plans are outside this adapter contract')
    if info['split_kv'] not in (0,1): raise ValueError('invalid split flag')
    for k in FIELDS[:-2]: positive_int(info[k],k,allow_zero=True)
    for k in ('padded_batch_size','total_num_rows','cta_tile_q'): positive_int(info[k],k)
    return info

def expected_descriptors(q, lengths, group, tile, chunk, split):
    if not q or len(q)!=len(lengths): raise ValueError('mismatched request geometry')
    for n,L in zip(q,lengths):
        positive_int(n,'query');positive_int(L,'KV length')
        if L<n: raise ValueError('causal KV shorter than query')
    positive_int(group,'group');positive_int(tile,'tile')
    if split: positive_int(chunk,'chunk')
    out=[];o=[0];merge=[0]
    for r,(n,L) in enumerate(zip(q,lengths)):
        splits=(L+chunk-1)//chunk if split else 1
        out += [(r,t,s) for t in range((n*group+tile-1)//tile) for s in range(splits)]
        o.append(o[-1]+n*splits)
        start=merge[-1]
        merge.extend(start+(i+1)*splits for i in range(n))
    return out,o,merge

def validate_descriptors(desc,q,lengths,info,chunk,group,o,merge):
    expected,eo,em=expected_descriptors(q,lengths,group,info['cta_tile_q'],chunk,bool(info['split_kv']))
    if info['total_num_rows']!=sum(q): raise ValueError('row capacity mismatch')
    if info['padded_batch_size']!=len(expected): raise ValueError('padded or incomplete launch')
    if len(desc)!=len(expected) or Counter(map(tuple,desc))!=Counter(expected):
        raise ValueError('not a complete one-to-one work-descriptor map')
    if list(o)!=eo: raise ValueError('output destinations differ')
    if info['split_kv'] and list(merge)!=em: raise ValueError('merge destinations differ')
    return expected

def order_indices(desc,q,lengths,tile,chunk,split,group,policy):
    if policy not in POLICIES: raise ValueError('unknown policy')
    def work(d):
        r,t,s=d
        packed=min(tile,q[r]*group-t*tile)
        visible=min(lengths[r],lengths[r]-q[r]+((t+1)*tile+group-1)//group)
        start=s*chunk if split else 0
        end=min(visible,(s+1)*chunk) if split else visible
        return max(0,end-start)*packed
    indices=list(range(len(desc)))
    if policy=='request_reverse': indices.sort(key=lambda i:(-desc[i][0],desc[i][1],desc[i][2]))
    elif policy=='heavy_first': indices.sort(key=lambda i:(-work(desc[i]),desc[i][0],desc[i][1],desc[i][2]))
    elif policy=='interleave': indices.sort(key=lambda i:(desc[i][1],desc[i][2],-lengths[desc[i][0]],desc[i][0]))
    validate_permutation(indices,len(desc))
    return indices

def validate_permutation(order,n):
    if len(order)!=n or any(type(x) is not int for x in order) or set(order)!=set(range(n)):
        raise ValueError('order must be a complete bijection')

class PlanSnapshot:
    """Owns one isolated wrapper's exact metadata; never exports a reusable live plan."""
    def __init__(self,wrapper,q,lengths,group=4):
        import torch
        import flashinfer
        from pathlib import Path
        if flashinfer.__version__!='0.6.18': raise ValueError('unqualified FlashInfer version')
        if hashlib.sha256((Path(flashinfer.__file__).parent/'prefill.py').read_bytes()).hexdigest()!=PREFILL_SHA:
            raise ValueError('unqualified wrapper source')
        if torch.cuda.is_current_stream_capturing(): raise ValueError('cannot snapshot during capture')
        torch.cuda.synchronize()
        self.wrapper=wrapper; self.buffer=wrapper._int_workspace_buffer
        self.ptr=self.buffer.data_ptr();self.device=self.buffer.device
        self.vector=[int(x) for x in wrapper._plan_info]
        self.info=decode_info(self.vector);self.q=list(q);self.lengths=list(lengths);self.group=group
        def read(offset,count):
            if offset%4 or offset+4*count>self.buffer.numel()*self.buffer.element_size():
                raise ValueError('invalid workspace span')
            return self.buffer.view(torch.uint8).narrow(0,offset,count*4).view(torch.int32).cpu().tolist()
        self._read=read
        n=self.info['padded_batch_size']
        columns=[read(self.info[k],n) for k in ('request_indices_offset','qo_tile_indices_offset','kv_tile_indices_offset')]
        self.desc=list(zip(*columns));self.chunk=read(self.info['kv_chunk_size_ptr_offset'],1)[0]
        self.o=read(self.info['o_indptr_offset'],len(q)+1)
        self.merge=read(self.info['merge_indptr_offset'],sum(q)+1) if self.info['split_kv'] else []
        validate_descriptors(self.desc,q,lengths,self.info,self.chunk,group,self.o,self.merge)
        self.invariant=canonical_hash(dict(vector=self.vector,o=self.o,merge=self.merge,chunk=self.chunk))
        self.last_descriptor_hash=canonical_hash(self.desc)
        self.original_bytes=self.buffer.clone()
        self.report=dict(info=self.info,chunk=self.chunk,query=q,total_kv=lengths,descriptors=self.desc,
                         output_indptr=self.o,merge_indptr=self.merge,workspace_bytes=self.buffer.numel(),
                         descriptor_hash=canonical_hash(self.desc),invariant_hash=self.invariant)
    def apply(self,policy):
        import torch
        if torch.cuda.is_current_stream_capturing(): raise ValueError('cannot mutate during graph capture')
        torch.cuda.synchronize()
        if self.wrapper._int_workspace_buffer.data_ptr()!=self.ptr or self.buffer.device!=self.device:
            raise ValueError('workspace identity changed')
        if [int(x) for x in self.wrapper._plan_info]!=self.vector: raise ValueError('stale plan generation')
        current=list(zip(*(self._read(self.info[k],len(self.desc)) for k in
                          ('request_indices_offset','qo_tile_indices_offset','kv_tile_indices_offset'))))
        if canonical_hash(current)!=self.last_descriptor_hash:
            raise ValueError('descriptor generation changed outside this snapshot')
        protected=dict(vector=self.vector,o=self._read(self.info['o_indptr_offset'],len(self.q)+1),
                       merge=self._read(self.info['merge_indptr_offset'],sum(self.q)+1) if self.info['split_kv'] else [],
                       chunk=self._read(self.info['kv_chunk_size_ptr_offset'],1)[0])
        if canonical_hash(protected)!=self.invariant: raise ValueError('protected plan metadata changed')
        # The baseline snapshot is immutable, and this buffer is owned by this experiment.
        self.buffer.copy_(self.original_bytes)
        order=order_indices(self.desc,self.q,self.lengths,self.info['cta_tile_q'],self.chunk,
                            bool(self.info['split_kv']),self.group,policy)
        for col,key in enumerate(('request_indices_offset','qo_tile_indices_offset','kv_tile_indices_offset')):
            values=torch.tensor([self.desc[i][col] for i in order],dtype=torch.int32,device=self.device)
            offset=self.info[key]
            self.buffer.view(torch.uint8).narrow(0,offset,values.numel()*4).view(torch.int32).copy_(values)
        torch.cuda.synchronize()
        actual=list(zip(*(self._read(self.info[k],len(order)) for k in
                        ('request_indices_offset','qo_tile_indices_offset','kv_tile_indices_offset'))))
        validate_descriptors(actual,self.q,self.lengths,self.info,self.chunk,self.group,
                             self._read(self.info['o_indptr_offset'],len(self.q)+1),
                             self._read(self.info['merge_indptr_offset'],sum(self.q)+1) if self.info['split_kv'] else [])
        if actual!=[self.desc[i] for i in order]: raise ValueError('descriptor write/read mismatch')
        self.last_descriptor_hash=canonical_hash(actual)
        return self.last_descriptor_hash
