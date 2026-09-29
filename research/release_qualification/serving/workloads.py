from __future__ import annotations
import random

def update(tokens,count,ids,n):
 if not isinstance(ids,list) or not all(type(x) is int for x in ids) or type(n) is not int or n<count:raise ValueError('invalid token stream')
 if len(ids)==n-count:new=ids
 elif len(ids)==n and ids[:count]==tokens:new=ids[count:]
 elif n==count and not ids:new=[]
 else:raise ValueError('token stream count/prefix mismatch')
 return tokens+new,n,len(new)
def prefix_tokens():
 rng=random.Random(2026093017);return [rng.randrange(200,16000) for _ in range(8448)]
def suffix(length,seed):
 rng=random.Random(seed);return [rng.randrange(200,16000) for _ in range(length)]
def work_specs(prefix,block):
 guarded=[16,32,64,128,256,512];balanced=[168]*6;short=[128,256,384,512,640,768]
 def cells(name,lengths,out,use_prefix):
  return [dict(id=f'{name}-{i:02d}',input_ids=(prefix if use_prefix else [])+suffix(n,2026093100+block*1000+i*17+len(name)),output_tokens=out,expect_cached=8192 if use_prefix else 0) for i,n in enumerate(lengths)]
 return {
  'guarded_prefix':dict(name='guarded_prefix',concurrency=6,cells=cells('guarded_prefix',guarded,32,True),seed=True),
  'balanced_prefix':dict(name='balanced_prefix',concurrency=6,cells=cells('balanced_prefix',balanced,32,True),seed=True),
  'short_prefill':dict(name='short_prefill',concurrency=6,cells=cells('short_prefill',short,32,False),seed=False),
  'decode':dict(name='decode',concurrency=8,cells=cells('decode',[128]*8,128,False),seed=False),
  'mixed':dict(name='mixed',concurrency=6,cells=cells('mixed-prefix',[16,192,512],64,True)+cells('mixed-short',[128,256,512],64,False),seed=True),
 }
