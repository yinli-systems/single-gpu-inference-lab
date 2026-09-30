from __future__ import annotations
import argparse,asyncio,hashlib,json,os,signal,socket,subprocess,time
from pathlib import Path
import aiohttp
from force_processor import ForcedHistoryProcessor

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
async def health(url,proc):
 start=time.monotonic()
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as s:
  while time.monotonic()-start<600:
   if proc.poll() is not None:raise RuntimeError('server exited '+str(proc.returncode))
   try:
    async with s.get(url+'/health') as r:
     if r.status==200:return time.monotonic()-start
   except Exception:pass
   await asyncio.sleep(.5)
 raise TimeoutError('startup')
async def one(session,url,row,rid,processor,gate):
 await gate.wait();payload=dict(rid=rid,input_ids=row['input_ids'],sampling_params=dict(temperature=0,max_new_tokens=row['max_new_tokens'],ignore_eos=True,custom_params={'forced_tokens':row['forced_tokens']}),custom_logit_processor=processor,return_logprob=True,top_logprobs_num=20,stream=False)
 start=time.perf_counter()
 async with session.post(url+'/generate',json=payload) as r:
  text=await r.text()
  if r.status!=200:raise RuntimeError(f'HTTP {r.status}: {text[:800]}')
 x=json.loads(text);tokens=x.get('output_ids',[]);meta=x.get('meta_info',{})
 if tokens[:len(row['forced_tokens'])]!=row['forced_tokens']:raise RuntimeError('forced prefix mismatch '+row['id'])
 if len(tokens)!=row['max_new_tokens']:raise RuntimeError('output length mismatch '+row['id'])
 return dict(id=row['id'],rid=rid,tokens=tokens,natural_token=(tokens[-1] if row['natural'] else None),expected_pristine_next=row['expected_pristine_next'],historical_cap_next=row['historical_cap_next'],
  output_token_logprobs=meta.get('output_token_logprobs'),output_top_logprobs=meta.get('output_top_logprobs'),latency=time.perf_counter()-start,finish_reason=meta.get('finish_reason'))
async def run_batch(url,target,mode,rep):
 gate=asyncio.Event();processor=ForcedHistoryProcessor.to_str();rows=[];limit=asyncio.Semaphore(target['concurrency'])
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=240),connector=aiohttp.TCPConnector(limit=target['concurrency'])) as session:
  async def wrapped(i,row):
   async with limit:return await one(session,url,row,f'forced-{mode}-{rep}-{target["id"]}-{i:02d}',processor,gate)
  tasks=[asyncio.create_task(wrapped(i,row)) for i,row in enumerate(target['rows'])];await asyncio.sleep(.1);gate.set();rows=await asyncio.gather(*tasks)
 return dict(target=target['id'],mode=mode,rep=rep,requests=rows,case_hash=target['case_hash'],declared_concurrency=target['concurrency'])
async def main(a):
 cases=json.loads(a.cases.read_text());target_cases=[]
 for t in cases['targets']:
  q=dict(t);q['concurrency']=cases['concurrency'];q['case_hash']=cases['case_hash'];target_cases.append(q)
 if a.out.exists():raise FileExistsError(a.out)
 a.out.mkdir(parents=True);trace=a.out/'trace';trace.mkdir()
 import flashinfer,sglang
 if flashinfer.__version__!='0.7.0' or sglang.__version__!='0.5.20':raise RuntimeError('unqualified packages')
 pkg=Path(flashinfer.__file__).parent;header=sha(pkg/'data/include/flashinfer/attention/prefill.cuh')
 bind=pkg.parent/'RESOURCE_BINDING.json'
 if a.mode=='pristine':
  if header!='e66bce2652c0a3c5b54f88510aff2b4b48ff6a52c4ee23fa97f91fa06a935b87':raise RuntimeError('pristine header')
 else:
  b=json.loads(bind.read_text());
  if b['mode']!=a.mode or b['modified_hashes']['flashinfer/data/include/flashinfer/attention/prefill.cuh']!=header:raise RuntimeError('cap binding')
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 url=f'http://127.0.0.1:{port}'
 cmd=[os.sys.executable,str(a.source/'server_entry.py'),'--model-path',str(a.model),'--host','127.0.0.1','--port',str(port),'--attention-backend','flashinfer','--dtype','bfloat16','--mem-fraction-static','0.80','--context-length','8192','--chunked-prefill-size','1024','--max-running-requests','16','--cuda-graph-max-bs-decode','16','--page-size','16','--skip-tokenizer-init','--disable-radix-cache','--enable-custom-logit-processor','--random-seed','2026093074']
 env=dict(os.environ,SGI_FORCED_CASES=str(a.cases),SGI_FORCED_TRACE=str(trace),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 save(a.out/'environment.json',dict(mode=a.mode,rep=a.rep,command=cmd,cases_sha256=sha(a.cases),header_sha256=header,source={p.name:sha(p) for p in a.source.iterdir() if p.is_file()},diagnostic_only=True,performance_claim=False))
 log=(a.out/'server.log').open('w');proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 try:
  startup=await health(url,proc);results=[]
  for target in target_cases:
   x=await run_batch(url,target,a.mode,a.rep);save(a.out/(target['id']+'.json'),x);results.append(x)
  await asyncio.sleep(1)
  captures={p.name:sha(p) for p in sorted(trace.glob('*')) if p.is_file()}
  required={f'{t["id"]}-occ0.json' for t in target_cases}|{f'{t["id"]}-occ0.pt' for t in target_cases}
  if not required<=set(captures):raise RuntimeError('target captures incomplete')
  save(a.out/'complete.json',dict(complete=True,mode=a.mode,rep=a.rep,startup_seconds=startup,cases_sha256=sha(a.cases),files={p.name:sha(p) for p in a.out.glob('*.json') if p.name!='complete.json'},captures=captures,diagnostic_only=True,performance_claim=False))
 except BaseException as exc:
  save(a.out/'failure.json',dict(type=type(exc).__name__,message=str(exc),server_exit=proc.poll(),mode=a.mode,rep=a.rep,cases_sha256=sha(a.cases),diagnostic_only=True,performance_claim=False))
  raise
 finally:
  if proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
  log.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['pristine','cap'],required=True);p.add_argument('--rep',type=int,required=True);asyncio.run(main(p.parse_args()))
