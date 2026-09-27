"""Geometry-only FA2 planner model, scoped to FP16 D=128 SM>=80, no CUDA graphs.

The split heuristic mirrors FlashInfer's documented implementation, not a new
algorithm. New candidate features describe causal work distribution of that plan.
No timings or labels are used by this module. Verify predictions against the
actual installed planner before interpreting any performance result.
"""
from __future__ import annotations
import hashlib
import json
import math
import statistics
from dataclasses import dataclass

POLICIES = ('auto', 'none', 's512', 's1024', 's2048', 's4096')


def ceildiv(x: int, y: int) -> int:
    if x < 0 or y <= 0:
        raise ValueError('ceildiv requires x>=0, y>0')
    return (x + y - 1) // y


@dataclass(frozen=True)
class Shape:
    name: str
    family: str
    split: str
    q: tuple[int, ...]
    k: tuple[int, ...]

    def __post_init__(self):
        if not self.q or len(self.q) != len(self.k):
            raise ValueError('q and cached k must be nonempty and aligned')
        if any(type(x) is not int for x in self.q + self.k):
            raise TypeError('all lengths must be integers, not booleans')
        if min(self.q) <= 0 or min(self.k) < 0:
            raise ValueError('queries positive, cached depths nonnegative')
        if self.split not in ('train', 'test', 'diagnostic'):
            raise ValueError('invalid split')

    def record(self):
        return dict(name=self.name, family=self.family, split=self.split,
                    q=list(self.q), k=list(self.k))

    @property
    def work(self):
        return sum(q*k+q*(q+1)//2 for q,k in zip(self.q,self.k))


def corpus() -> list[Shape]:
    result = []
    # Entire (query budget, cache scale) families are held out. No old discovery
    # cache scales are used in the primary matrix.
    for i,Q in enumerate((512,1024,2048)):
        for j,K in enumerate((1536,3072,6144,12288)):
            family=f'Q{Q}-K{K}'
            split='test' if (i+j)%2 else 'train'
            for cut in (Q//8,Q//4-1,Q//4+1,3*Q//8):
                depth=(K,K-(Q//2-cut))
                for state,qs in (('A',(cut,Q-cut)),('B',(Q//2,Q//2))):
                    result.append(Shape(f'{family}-eq{cut}-{state}',family,split,qs,depth))
            for n in (4,8):
                q=[Q//(2*n)]*(n-1);q.append(Q-sum(q))
                k=[(K*(r+1))//n for r in range(n)]
                for state,ks in (('A',k),('B',k[::-1])):
                    result.append(Shape(f'{family}-n{n}-{state}',family,split,tuple(q),tuple(ks)))
    # Known equal-work discontinuities are diagnosis-only, excluded from fitting,
    # model selection, and held-out performance summaries.
    for cut in (128,255,257,384):
        depth=(8192,8192-(512-cut))
        for state,qs in (('A',(cut,1024-cut)),('B',(512,512))):
            result.append(Shape(f'discovery-eq{cut}-{state}','discovery','diagnostic',qs,depth))
    keys=[(s.q,s.k) for s in result]
    if len(keys)!=len(set(keys)):
        raise AssertionError('duplicate exact geometry in corpus')
    return result


def plan(shape: Shape, hq: int, hkv: int, sms: int, policy: str) -> dict:
    if hq % hkv or min(hq,hkv,sms)<=0 or policy not in POLICIES:
        raise ValueError('unsupported heads, hardware or policy')
    g=hq//hkv
    packed=sum(shape.q)*g//len(shape.q)
    tile=128 if packed>64 else (64 if packed>16 else 16)
    lens=[q+k for q,k in zip(shape.q,shape.k)]
    if policy=='none':
        size=max(lens);split=False
    elif policy=='auto':
        low,high=128,max(lens);capacity=2*sms//hkv
        while low<high:
            mid=(low+high)//2
            count=sum(ceildiv(q*g,tile)*ceildiv(L,mid) for q,L in zip(shape.q,lens))
            if count>capacity:low=mid+1
            else:high=mid
        size=low;split=size<max(lens)
    else:
        size=int(policy[1:]);split=any(L>size for L in lens)
    tasks=[];rect=[];merge=0
    for q,k,L in zip(shape.q,shape.k,lens):
        chunks=ceildiv(L,size) if split else 1
        merge+=q*chunks
        for tile_index in range(ceildiv(q*g,tile)):
            q_end=min(q,ceildiv((tile_index+1)*tile,g))
            for c in range(chunks):
                start=c*size;end=min(L,start+size)
                rect.append(ceildiv(end-start,64))
                tasks.append(ceildiv(max(0,min(end,k+q_end)-start),64))
    return dict(tile=tile,kv_chunk=size,split=split,grid_x=len(tasks),
                active=sum(t>0 for t in tasks),task_iterations=tasks,
                rectangular_iterations=rect,merge_rows=merge if split else 0)


def features(shape: Shape, hq: int, hkv: int, sms: int, policy: str, kind: str) -> list[float]:
    q,k=shape.q,shape.k;n=len(q)
    triangle=sum(x*(x+1)/2 for x in q)
    marginal=sum(q)*sum(k)/n+triangle
    onehot=[float(policy==p) for p in POLICIES]
    base=[n,sum(q),sum(k),max(q),max(k),sum(x*x for x in q),
          statistics.pvariance(q),statistics.pvariance(k),hq,hkv,sms]
    if kind=='marginal': values=base+[marginal]
    else: values=base+[shape.work,marginal]
    if kind in ('plan','causal'):
        p=plan(shape,hq,hkv,sms,policy);r=p['rectangular_iterations']
        values += [p['tile'],p['kv_chunk'],p['grid_x']*hkv,float(p['split']),
                   sum(r)*hkv,max(r),sum(x*x for x in r)*hkv,
                   ceildiv(p['grid_x']*hkv,2*sms),p['merge_rows']*hq]
        if kind=='causal':
            t=p['task_iterations'];active=p['active']*hkv
            values += [active,(p['grid_x']-p['active'])*hkv,sum(t)*hkv,max(t),
                       sum(x*x for x in t)*hkv,ceildiv(active,2*sms),
                       max(max(t),sum(t)*hkv/(2*sms))]
    if kind not in ('marginal','joint','plan','causal'):
        raise ValueError('unknown feature representation')
    return [math.log1p(float(x)) for x in values]+onehot


def corpus_json() -> str:
    return json.dumps([x.record() for x in corpus()],sort_keys=True,indent=2)+'\n'


if __name__=='__main__':
    data=corpus_json()
    print(data,end='')
