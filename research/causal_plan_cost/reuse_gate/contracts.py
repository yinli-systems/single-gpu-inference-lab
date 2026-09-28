"""Metadata contracts only; neither timings nor speedups are synthesized."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math

POLICIES=('auto','none','s512','s1024','s2048','s4096')


def profitable_interval(setup_difference_ms: float, per_call_saving_ms: float):
    """Strict inequality setup-r*saving<0 for integer r>=1. None means none.

    Result [first,last]; last=None means unbounded. Inputs may be negative.
    This is a diagnostic linear model, not an integrated measurement.
    """
    if not all(math.isfinite(x) for x in (setup_difference_ms,per_call_saving_ms)):
        raise ValueError('finite inputs required')
    a,b=map(lambda x:Fraction(str(x)),(setup_difference_ms,per_call_saving_ms))
    if b==0:return [1,None] if a<0 else None
    ratio=a/b
    if b>0:return [max(1,ratio.numerator//ratio.denominator+1),None]
    ceiling=-((-ratio.numerator)//ratio.denominator)
    last=ceiling-1
    return [1,last] if last>=1 else None


def _validate(q,k,hq,hkv,sms):
    if not q or len(q)!=len(k) or len(q)>64:raise ValueError('aligned 1..64 requests required')
    if any(type(v) is not int for v in (*q,*k,hq,hkv,sms)):raise TypeError('integer metadata required')
    if min(q)<=0 or min(k)<0 or min(hq,hkv,sms)<=0 or hq%hkv:raise ValueError('invalid geometry')
    if sum(q)+sum(k)>=2**31:raise ValueError('int32 indptr limit')


def scratch_bytes(q,k,hq,hkv,sms,policy):
    """Installed0.6.18 FP16 D128 ragged FA2/no-graph scratch bound, O(n log K).

    No materialized per-CTA lists; compare this to the pinned reference formula.
    Library plan/run and stream serialization are still required.
    """
    _validate(q,k,hq,hkv,sms)
    if policy not in POLICIES:raise ValueError('unsupported policy')
    g=hq//hkv;packed=sum(q)*g//len(q)
    tile=128 if packed>64 else (64 if packed>16 else 16)
    lens=[a+b for a,b in zip(q,k)]
    cd=lambda a,b:(a+b-1)//b
    if policy=='none':return 16
    if policy=='auto':
        lo,hi=128,max(lens)
        while lo<hi:
            mid=(lo+hi)//2
            count=sum(cd(a*g,tile)*cd(b,mid) for a,b in zip(q,lens))
            if count>2*sms//hkv:lo=mid+1
            else:hi=mid
        chunk=lo
    else:chunk=int(policy[1:])
    if chunk>=max(lens):return 16
    grid=sum(cd(a*g,tile)*cd(b,chunk) for a,b in zip(q,lens))
    return hq*grid*tile*129*4+32


def bounded_policy(q,k,hq,hkv,sms,policy,budget):
    if type(budget) is not int or budget<16:raise ValueError('scratch ceiling too small')
    for candidate in dict.fromkeys((policy,'auto','none')):
        if scratch_bytes(q,k,hq,hkv,sms,candidate)<=budget:return candidate
    raise RuntimeError('no feasible safe policy')


@dataclass(frozen=True)
class PlanKey:
    q:tuple[int,...]
    k:tuple[int,...]
    hq:int
    hkv:int
    sms:int
    head_dim:int=128
    dtype:str='float16'
    layout:str='NHD'
    causal:bool=True
    window_left:int=-1
    position_mode:str='NONE'
    environment:str=''
    execution_mode:str='eager'
    scratch_ceiling:int=128*1024*1024
    def __post_init__(self):
        _validate(self.q,self.k,self.hq,self.hkv,self.sms)
        if self.head_dim!=128 or self.dtype!='float16' or self.layout!='NHD':raise ValueError('unsupported installed path')
        if not self.causal or self.window_left!=-1 or self.position_mode!='NONE':raise ValueError('unsupported attention semantics')
        if self.execution_mode!='eager':raise ValueError('graph path is not qualified')
        if not self.environment or self.scratch_ceiling<16:raise ValueError('incomplete environment/resource identity')


class PolicyCache:
    """Bounded decision LRU, not a cache of device plans or scratch buffers."""
    def __init__(self,capacity=2):
        if type(capacity) is not int or capacity<0:raise ValueError('invalid capacity')
        self.capacity=capacity;self.entries=OrderedDict();self.hits=0;self.misses=0;self.evictions=0
    def resolve(self,key,choose):
        if key in self.entries:
            self.hits+=1;value=self.entries.pop(key);self.entries[key]=value;return value
        self.misses+=1
        value=choose()
        if value not in POLICIES:raise ValueError('invalid cached policy')
        if self.capacity:
            self.entries[key]=value
            if len(self.entries)>self.capacity:self.entries.popitem(last=False);self.evictions+=1
        return value


def fresh_corpus():
    rows=[]
    for Q in (640,1280,2560):
        for K in (2048,4096,8192):
            family=f'reuse-Q{Q}-K{K}'
            for n in (2,8):
                # Four states permit key changes and A/B/C/A eviction tests.
                for state in range(3):
                    base=Q//(2*n);q=[base]*(n-1)+[Q-base*(n-1)]
                    q=q[state:]+q[:state]
                    k=[K*(i+2)//(n+1)+state*17 for i in range(n)]
                    rows.append(dict(name=f'{family}-n{n}-{state}',family=family,
                                     split='test',q=q,k=k,n=n,state=state))
    if len({(tuple(r['q']),tuple(r['k'])) for r in rows})!=len(rows):raise AssertionError('duplicate geometry')
    return rows


def corpus_hash():
    return hashlib.sha256(json.dumps(fresh_corpus(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
