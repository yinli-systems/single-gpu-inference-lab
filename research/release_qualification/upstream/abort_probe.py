from __future__ import annotations
import argparse,asyncio,json,time
from pathlib import Path
import aiohttp

def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def update(tokens,count,ids,n):
 if len(ids)==n-count:new=ids
 elif len(ids)==n and ids[:count]==tokens:new=ids[count:]
 elif n==count and not ids:new=[]
 else:raise ValueError('stream mismatch')
 return tokens+new,n
async def one(session,url,rid,input_ids,nout):
 tokens=[];count=0;finish=None;frames=[];start=time.perf_counter()
 payload=dict(rid=rid,input_ids=input_ids,sampling_params=dict(temperature=0,max_new_tokens=nout,ignore_eos=True),stream=True)
 async with session.post(url+'/generate',json=payload) as r:
  if r.status!=200:raise RuntimeError(await r.text())
  async for line in r.content:
   t=line.decode().strip()
   if not t.startswith('data:'):continue
   b=t[5:].strip()
   if b=='[DONE]':break
   x=json.loads(b);frames.append(x)
   if 'error' in x:break
   m=x.get('meta_info',{});n=m.get('completion_tokens',count);tokens,count=update(tokens,count,x.get('output_ids',[]),n);finish=m.get('finish_reason',finish)
 return dict(rid=rid,tokens=tokens,received=len(tokens),expected=nout,finish_reason=finish,complete=len(tokens)==nout and isinstance(finish,dict) and finish.get('type')=='length',elapsed=time.perf_counter()-start,frames=frames)
async def health(url,timeout=720):
 start=time.monotonic()
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as s:
  while time.monotonic()-start<timeout:
   try:
    async with s.get(url+'/health') as r:
     if r.status==200:return
   except:pass
   await asyncio.sleep(.5)
 raise TimeoutError
async def run(a):
 a.out.mkdir(parents=True,exist_ok=False);await health(a.url);rng=__import__('random').Random(20260930);prompt=[rng.randrange(200,16000) for _ in range(128)];results=[]
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as s:
  for epoch in range(2):
   for call in range(2):
    x=await one(s,a.url,f'reuse-{epoch}',prompt,256);x.update(epoch=epoch,call=call,kind='reused');results.append(x);save(a.out/f'reused-{epoch}-{call}.json',x)
   await asyncio.sleep(2.2)
  for i in range(4):
   x=await one(s,a.url,f'unique-{i}',prompt,256);x.update(call=i,kind='unique');results.append(x);save(a.out/f'unique-{i}.json',x)
 await asyncio.sleep(2.2);save(a.out/'complete.json',dict(complete=True,requests=len(results),complete_requests=sum(x['complete'] for x in results),reused=[dict(rid=x['rid'],received=x['received'],complete=x['complete'],finish_reason=x['finish_reason']) for x in results if x['kind']=='reused']))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--out',type=Path,required=True);asyncio.run(run(p.parse_args()))
