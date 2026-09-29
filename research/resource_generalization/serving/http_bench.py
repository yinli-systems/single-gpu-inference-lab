"""Actual SGLang HTTP streaming; synthetic workload, complete real model weights.
Records end-client timing, response token IDs, failed requests and profile separately.
"""
from __future__ import annotations
import argparse,asyncio,gzip,hashlib,json,math,os,random,signal,socket,subprocess,sys,time
from pathlib import Path
import aiohttp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from measure import save,sha

def quantile(xs,p):
 xs=sorted(xs)
 if not xs or not all(math.isfinite(x) for x in xs):raise ValueError('nonfinite/empty observations')
 t=(len(xs)-1)*p;lo=int(t);return xs[lo]+(xs[min(lo+1,len(xs)-1)]-xs[lo])*(t-lo)
def update(tokens,count,ids,n):
 if not isinstance(ids,list) or not all(type(x) is int for x in ids) or type(n) is not int or n<count:raise ValueError('invalid token stream')
 if len(ids)==n-count:new=ids
 elif len(ids)==n and ids[:count]==tokens:new=ids[count:]
 elif n==count and not ids:new=[]
 else:raise ValueError('token stream count/prefix mismatch')
 return tokens+new,n,len(new)
def workloads():
 result=[]
 for kind,ls,nout,c in [('prefill',[256,512,1024,2048],32,4),('decode',[128],128,8),('mixed',[64,2048,256,4096],64,8)]:
  cells=[]
  for i in range(16):
   rng=random.Random(936644+i);length=ls[i%len(ls)]
   cells.append(dict(id=f'{kind}-{i:02d}',input_ids=[rng.randrange(200,16000) for _ in range(length)],output_tokens=nout))
  result.append(dict(name=kind,concurrency=c,cells=cells))
 return result

async def request(session,url,cell,origin):
 started=time.perf_counter();tokens=[];times=[];count=0;events=[];finish=None
 payload=dict(rid=cell['id'],input_ids=cell['input_ids'],sampling_params=dict(temperature=0,max_new_tokens=cell['output_tokens'],ignore_eos=True),stream=True)
 async with session.post(url+'/generate',json=payload) as resp:
  if resp.status!=200:raise RuntimeError(f'HTTP {resp.status}: '+(await resp.text())[:1000])
  async for line in resp.content:
   text=line.decode().strip()
   if not text.startswith('data:'):continue
   body=text[5:].strip()
   if body=='[DONE]':break
   obj=json.loads(body)
   if 'error' in obj:raise RuntimeError(str(obj['error']))
   meta=obj.get('meta_info',{});n=meta.get('completion_tokens')
   if n is None:raise RuntimeError('completion count absent')
   ids=obj.get('output_ids',[])
   tokens,count,added=update(tokens,count,ids,n);stamp=time.perf_counter()
   times.extend([stamp-started]*added);events.append(dict(received=stamp-started,count=count,new=added))
   finish=meta.get('finish_reason',finish)
 if len(tokens)!=cell['output_tokens'] or not times or finish is None:raise RuntimeError('incomplete streamed request '+cell['id'])
 end=time.perf_counter()
 return dict(id=cell['id'],input_tokens=len(cell['input_ids']),tokens=tokens,events=events,token_times=times,
  ttft=times[0],tpot=(times[-1]-times[0])/(len(times)-1) if len(times)>1 else 0.,latency=end-started,
  request_start=started-origin,completion=end-origin,finish_reason=finish)

async def batch(url,work):
 gate=asyncio.Semaphore(work['concurrency']);start=time.perf_counter();rows=[]
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180),connector=aiohttp.TCPConnector(limit=work['concurrency'])) as session:
  async def one(cell):
   async with gate:
    result=await request(session,url,cell,start);rows.append(result);return result
  returned=await asyncio.gather(*(one(c) for c in work['cells']),return_exceptions=True)
 elapsed=time.perf_counter()-start;fail=[repr(x) for x in returned if isinstance(x,BaseException)]
 result=dict(workload=work['name'],concurrency=work['concurrency'],elapsed=elapsed,requests=sorted(rows,key=lambda x:x['id']),errors=fail,
  workload_sha256=hashlib.sha256(json.dumps(work,sort_keys=True).encode()).hexdigest())
 if fail:return result
 if len(rows)!=len(work['cells']) or len({x['id'] for x in rows})!=len(rows):raise RuntimeError('request coverage mismatch')
 result.update(output_tokens=sum(len(x['tokens']) for x in rows),output_tokens_per_second=sum(len(x['tokens']) for x in rows)/elapsed,
  requests_per_second=len(rows)/elapsed,TTFT={str(p):quantile([x['ttft'] for x in rows],p) for p in [.5,.95,.99]},
  TPOT={str(p):quantile([x['tpot'] for x in rows],p) for p in [.5,.95,.99]},
  strict_slo_goodput=sum(x['ttft']<=2 and x['tpot']<=.05 for x in rows)/elapsed,
  timing_scope='HTTP request start to client SSE token arrival; closed-loop bounded-concurrency synthetic input-ID workload')
 return result

async def health(url,process,timeout):
 start=time.monotonic()
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as s:
  while time.monotonic()-start<timeout:
   if process.poll() is not None:raise RuntimeError('server exited '+str(process.returncode))
   try:
    async with s.get(url+'/health') as r:
     if r.status==200:return time.monotonic()-start
   except (aiohttp.ClientError,asyncio.TimeoutError):pass
   await asyncio.sleep(.5)
 raise TimeoutError('bounded server startup exceeded')

async def profile(url,out):
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as s:
  async with s.post(url+'/start_profile',json=dict(output_dir=str(out),activities=['CPU','GPU'],record_shapes=True)) as r:
   text=await r.text()
   if r.status!=200:raise RuntimeError('start profile failed '+text[:800])
  cell=dict(id='profile-only',input_ids=[500+i%100 for i in range(4096)],output_tokens=16)
  await request(s,url,cell,time.perf_counter())
  async with s.post(url+'/stop_profile') as r:
   text=await r.text()
   if r.status!=200:raise RuntimeError('stop profile failed '+text[:800])

async def main_async(a):
 if a.out.exists():raise FileExistsError('preserve serving evidence')
 a.out.mkdir(parents=True);model=a.model
 cfg=json.loads((model/'config.json').read_text())
 if cfg['num_hidden_layers']!=36 or cfg['num_attention_heads']!=32 or cfg['num_key_value_heads']!=8 or cfg['head_dim']!=128:raise RuntimeError('wrong complete model')
 import flashinfer,sglang
 pkg=Path(flashinfer.__file__).parent
 if flashinfer.__version__!='0.7.0':raise RuntimeError('unqualified FI version')
 header=sha(pkg/'data/include/flashinfer/attention/prefill.cuh')
 if a.mode=='pristine':
  from prepare import EXPECTED
  if header!=EXPECTED:raise RuntimeError('pristine mismatch')
 else:
  bind=json.loads((pkg.parent/'RESOURCE_BINDING.json').read_text())
  if bind['mode']!=a.mode or header!=bind['modified_sha256']:raise RuntimeError('source mismatch')
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 url=f'http://127.0.0.1:{port}'
 args=[sys.executable,'-m','sglang.launch_server','--model-path',str(model),'--host','127.0.0.1','--port',str(port),
  '--attention-backend','flashinfer','--dtype','bfloat16','--mem-fraction-static','0.80','--context-length','8192',
  '--chunked-prefill-size','1024','--max-running-requests','16','--cuda-graph-max-bs','16','--page-size','16',
  '--skip-tokenizer-init','--disable-radix-cache']
 env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SGLANG_TORCH_PROFILER_DIR=str(a.out/'profile'),SGLANG_ENABLE_JIT_DEEPGEMM='0')
 receipt=dict(mode=a.mode,rep=a.rep,stage=a.stage,command=args,flashinfer=flashinfer.__version__,sglang=sglang.__version__,prefill_sha256=header,
  config_sha256=sha(model/'config.json'),model_index_sha256=sha(model/'model.safetensors.index.json'),source_sha256=sha(__file__),
  all_model_layers=cfg['num_hidden_layers'],weights_path=str(model),weight_hashes_separate=True,
  full_model=True,HTTP=True,input_API='native token IDs, tokenizer initialization skipped',default_promotion=False,serving_promotion=False)
 save(a.out/'environment.json',receipt)
 log=(a.out/'server.log').open('w');proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 save(a.out/'owned_process.json',dict(pid=proc.pid,port=port))
 try:
  receipt['startup_seconds']=await health(url,proc,720);save(a.out/'environment.json',receipt)
  warm=dict(name='warmup',concurrency=2,cells=[dict(id=f'warm-{j}',input_ids=[500+j]*256,output_tokens=8) for j in range(2)])
  w=await batch(url,warm);save(a.out/'warmup.json',w)
  if w['errors']:raise RuntimeError('warmup requests failed')
  works=workloads()
  if a.stage=='smoke':works=[dict(name='smoke',concurrency=4,cells=[dict(c,id='smoke-'+c['id']) for c in workloads()[2]['cells'][:4]])]
  for b in range(1 if a.stage=='smoke' else 3):
   for work in works:
    result=await batch(url,work);save(a.out/f'{work["name"]}-b{b}.json',result)
    if result['errors']:raise RuntimeError('HTTP request failure preserved')
    print('SERVING',a.mode,a.rep,work['name'],b,result.get('output_tokens_per_second'),flush=True)
  if a.rep==0:
   try:
    await profile(url,a.out/'profile');kernels=[]
    for f in (a.out/'profile').rglob('*'):
     if f.is_file() and (f.name.endswith('.json') or f.name.endswith('.json.gz')):
      raw=gzip.open(f,'rt').read() if f.name.endswith('.gz') else f.read_text()
      data=json.loads(raw)
      for e in data.get('traceEvents',[]):
       if e.get('cat')=='kernel' and 'Prefill' in e.get('name',''):kernels.append(dict(name=e['name'],args=e.get('args',{})))
    save(a.out/'profile-evidence.json',dict(kernels=kernels,all_profile_timing_excluded=True,launch_hit_count_64KiB=sum(x['args'].get('shared memory')==65536 for x in kernels)))
   except Exception as e:save(a.out/'profile-failure.json',dict(type=type(e).__name__,message=str(e),path_qualification=False))
  save(a.out/'complete.json',dict(complete=True,full_model=True,HTTP=True,mode=a.mode,rep=a.rep,stage=a.stage,
   files={p.name:sha(p) for p in a.out.glob('*.json')},performance_promotion=False,serving_promotion=False))
 except BaseException as e:
  save(a.out/'failure.json',dict(type=type(e).__name__,message=str(e),server_exit=proc.poll()));raise
 finally:
  if proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
  log.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
 p.add_argument('--mode',choices=['pristine','off','cap','wide'],required=True);p.add_argument('--stage',choices=['smoke','formal'],required=True);p.add_argument('--rep',type=int,default=0)
 asyncio.run(main_async(p.parse_args()))
