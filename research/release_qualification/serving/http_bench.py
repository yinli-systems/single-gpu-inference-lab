"""Real multi-model SGLang HTTP release qualification with prefix-cache evidence."""
from __future__ import annotations
import argparse,asyncio,gzip,hashlib,json,math,os,random,signal,socket,subprocess,sys,time
from pathlib import Path
import aiohttp
from workloads import update,prefix_tokens,work_specs,suffix

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def quantile(xs,p):
 xs=sorted(xs)
 if not xs or not all(math.isfinite(x) for x in xs):raise ValueError('empty/nonfinite')
 t=(len(xs)-1)*p;lo=int(t);return xs[lo]+(xs[min(lo+1,len(xs)-1)]-xs[lo])*(t-lo)
async def health(url,proc,timeout=720):
 start=time.monotonic()
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as s:
  while time.monotonic()-start<timeout:
   if proc.poll() is not None:raise RuntimeError('server exited '+str(proc.returncode))
   try:
    async with s.get(url+'/health') as r:
     if r.status==200:return time.monotonic()-start
   except (aiohttp.ClientError,asyncio.TimeoutError):pass
   await asyncio.sleep(.5)
 raise TimeoutError('startup timeout')
async def post_text(url,path,payload=None):
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as s:
  async with s.post(url+path,json=payload) as r:
   text=await r.text()
   if r.status!=200:raise RuntimeError(f'{path} HTTP {r.status}: {text[:800]}')
   return text
async def request(session,url,cell,rid,origin,logprobs):
 started=time.perf_counter();tokens=[];times=[];count=0;finish=None;cached=0;metas=[]
 payload=dict(rid=rid,input_ids=cell['input_ids'],sampling_params=dict(temperature=0,max_new_tokens=cell['output_tokens'],ignore_eos=True),stream=True)
 if logprobs:payload.update(return_logprob=True,top_logprobs_num=5,logprob_start_len=-1)
 async with session.post(url+'/generate',json=payload) as resp:
  if resp.status!=200:raise RuntimeError(f'HTTP {resp.status}: '+(await resp.text())[:1000])
  async for line in resp.content:
   text=line.decode().strip()
   if not text.startswith('data:'):continue
   body=text[5:].strip()
   if body=='[DONE]':break
   obj=json.loads(body)
   if 'error' in obj:raise RuntimeError(str(obj['error']))
   meta=obj.get('meta_info',{});metas.append(meta);cached=max(cached,int(meta.get('cached_tokens',0) or 0))
   n=meta.get('completion_tokens')
   if n is None:raise RuntimeError('completion count absent')
   tokens,count,added=update(tokens,count,obj.get('output_ids',[]),n);stamp=time.perf_counter();times.extend([stamp-started]*added)
   finish=meta.get('finish_reason',finish)
 if len(tokens)!=cell['output_tokens'] or not times or finish is None:raise RuntimeError('incomplete '+rid)
 end=time.perf_counter();last=metas[-1] if metas else {}
 return dict(id=cell['id'],rid=rid,input_tokens=len(cell['input_ids']),tokens=tokens,cached_tokens=cached,
  token_times=times,ttft=times[0],tpot=(times[-1]-times[0])/(len(times)-1) if len(times)>1 else 0.,latency=end-started,
  request_start=started-origin,completion=end-origin,finish_reason=finish,
  output_token_logprobs=last.get('output_token_logprobs'),output_top_logprobs=last.get('output_top_logprobs'))
async def batch(url,work,tag,logprobs):
 gate=asyncio.Semaphore(work['concurrency']);start=time.perf_counter();rows=[]
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300),connector=aiohttp.TCPConnector(limit=work['concurrency'])) as session:
  async def one(i,cell):
   async with gate:
    rid=f'{tag}-{work["name"]}-{i:02d}'
    r=await request(session,url,cell,rid,start,logprobs);rows.append(r);return r
  returned=await asyncio.gather(*(one(i,c) for i,c in enumerate(work['cells'])),return_exceptions=True)
 elapsed=time.perf_counter()-start;errors=[repr(x) for x in returned if isinstance(x,BaseException)]
 out=dict(workload=work['name'],concurrency=work['concurrency'],elapsed=elapsed,requests=sorted(rows,key=lambda x:x['id']),errors=errors,
  workload_sha256=hashlib.sha256(json.dumps(work,sort_keys=True).encode()).hexdigest(),timing_scope='HTTP request start to client SSE token arrival')
 if errors:return out
 if len(rows)!=len(work['cells']) or len({x['rid'] for x in rows})!=len(rows):raise RuntimeError('coverage/unique RID failure')
 for cell,row in zip(sorted(work['cells'],key=lambda x:x['id']),out['requests']):
  if cell['expect_cached'] and row['cached_tokens']<cell['expect_cached']:raise RuntimeError(f'prefix cache miss {row["id"]}: {row["cached_tokens"]}')
 total=sum(len(x['tokens']) for x in rows)
 out.update(output_tokens=total,output_tokens_per_second=total/elapsed,requests_per_second=len(rows)/elapsed,
  TTFT={str(p):quantile([x['ttft'] for x in rows],p) for p in [.5,.95,.99]},TPOT={str(p):quantile([x['tpot'] for x in rows],p) for p in [.5,.95,.99]},
  strict_slo_goodput=sum(x['ttft']<=2 and x['tpot']<=.05 for x in rows)/elapsed,
  cached_min=min(x['cached_tokens'] for x in rows),cached_max=max(x['cached_tokens'] for x in rows))
 return out
async def flush_seed(url,prefix,tag,logprobs=False):
 await post_text(url,'/flush_cache',{})
 cell=dict(id='seed',input_ids=prefix,output_tokens=1,expect_cached=0)
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as s:
  r=await request(s,url,cell,tag+'-seed',time.perf_counter(),logprobs)
 return r
async def profile_work(url,prefix,work,out,tag):
 out.mkdir(parents=True,exist_ok=True)
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as s:
  async with s.post(url+'/start_profile',json=dict(output_dir=str(out),activities=['CPU','GPU'],record_shapes=True)) as r:
   t=await r.text()
   if r.status!=200:raise RuntimeError(t[:800])
 await flush_seed(url,prefix,tag+'-profile') if work['seed'] else None
 result=await batch(url,work,tag+'-profile',False)
 if result['errors']:raise RuntimeError('profile workload failed')
 await post_text(url,'/stop_profile',{})
 kernels=[]
 for f in out.rglob('*'):
  if f.is_file() and (f.name.endswith('.json') or f.name.endswith('.json.gz')):
   raw=gzip.open(f,'rt').read() if f.name.endswith('.gz') else f.read_text();data=json.loads(raw)
   for e in data.get('traceEvents',[]):
    if e.get('cat')=='kernel' and 'Prefill' in e.get('name',''):kernels.append(dict(name=e['name'],args=e.get('args',{})))
 return dict(workload=work['name'],kernel_count=len(kernels),launches_64KiB=sum(x['args'].get('shared memory')==65536 for x in kernels),
  launch_shared_memory=sorted({x['args'].get('shared memory') for x in kernels if x['args'].get('shared memory') is not None}),timing_excluded=True)
async def main_async(a):
 if a.out.exists():raise FileExistsError(a.out)
 a.out.mkdir(parents=True);model=a.model;cfg=json.loads((model/'config.json').read_text())
 heads=cfg['num_attention_heads'];head_dim=cfg.get('head_dim') or cfg['hidden_size']//heads
 if head_dim!=128:raise RuntimeError('unqualified head_dim')
 import flashinfer,sglang
 pkg=Path(flashinfer.__file__).parent
 if flashinfer.__version__!='0.7.0':raise RuntimeError('wrong FlashInfer')
 header=sha(pkg/'data/include/flashinfer/attention/prefill.cuh')
 if a.mode=='pristine':
  if header!='e66bce2652c0a3c5b54f88510aff2b4b48ff6a52c4ee23fa97f91fa06a935b87':raise RuntimeError('pristine mismatch')
 else:
  bind=json.loads((pkg.parent/'RESOURCE_BINDING.json').read_text())
  if bind['mode']!=a.mode or header!=bind['modified_hashes']['flashinfer/data/include/flashinfer/attention/prefill.cuh']:raise RuntimeError('overlay mismatch')
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 url=f'http://127.0.0.1:{port}';seed=20260931+a.rep
 args=[sys.executable,'-m','sglang.launch_server','--model-path',str(model),'--host','127.0.0.1','--port',str(port),'--attention-backend','flashinfer','--dtype','bfloat16',
  '--mem-fraction-static',str(a.mem_fraction),'--context-length','12288','--chunked-prefill-size','1024','--max-running-requests','16','--cuda-graph-max-bs-decode','8','--page-size','16','--skip-tokenizer-init','--random-seed',str(seed)]
 if a.stage=='correctness':args.append('--enable-deterministic-inference')
 env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SGLANG_TORCH_PROFILER_DIR=str(a.out/'profile'),SGLANG_ENABLE_JIT_DEEPGEMM='0')
 save(a.out/'environment.json',dict(mode=a.mode,model_id=a.model_id,rep=a.rep,stage=a.stage,command=args,flashinfer=flashinfer.__version__,sglang=sglang.__version__,header_sha256=header,
  config_sha256=sha(model/'config.json'),layers=cfg['num_hidden_layers'],heads=heads,kv_heads=cfg['num_key_value_heads'],head_dim=head_dim,full_model=True,HTTP=True,unique_rids=True,performance_promotion=False))
 log=(a.out/'server.log').open('w');proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);save(a.out/'owned_process.json',dict(pid=proc.pid,port=port))
 prefix=prefix_tokens()
 try:
  startup=await health(url,proc);envrec=json.loads((a.out/'environment.json').read_text());envrec['startup_seconds']=startup;save(a.out/'environment.json',envrec)
  warm=dict(name='warmup',concurrency=2,cells=[dict(id=f'warm-{i}',input_ids=suffix(256,8+i),output_tokens=8,expect_cached=0) for i in range(2)],seed=False)
  w=await batch(url,warm,f'{a.mode}-{a.stage}-r{a.rep}-warm',False);save(a.out/'warmup.json',w)
  if w['errors']:raise RuntimeError('warmup failure')
  blocks=1 if a.stage=='smoke' else (2 if a.stage=='correctness' else 4)
  base_order=['guarded_prefix','balanced_prefix','short_prefill','decode','mixed']
  for b in range(blocks):
   works=work_specs(prefix,b);order=base_order[b%len(base_order):]+base_order[:b%len(base_order)]
   for name in order:
    work=works[name]
    if work['seed']:save(a.out/f'seed-{name}-b{b}.json',await flush_seed(url,prefix,f'{a.mode}-{a.stage}-r{a.rep}-b{b}-{name}',a.stage=='correctness'))
    result=await batch(url,work,f'{a.mode}-{a.stage}-r{a.rep}-b{b}',a.stage=='correctness');save(a.out/f'{name}-b{b}.json',result)
    if result['errors']:raise RuntimeError('request failure')
    print('SERVING',a.model_id,a.mode,a.stage,a.rep,name,b,result['output_tokens_per_second'],flush=True)
  if a.rep==0 and a.stage in ('smoke','performance'):
   works=work_specs(prefix,99);prof={}
   for name in ('guarded_prefix','balanced_prefix'):
    prof[name]=await profile_work(url,prefix,works[name],a.out/'profile'/name,f'{a.mode}-r{a.rep}-{name}')
   save(a.out/'profile-evidence.json',prof)
  save(a.out/'complete.json',dict(complete=True,mode=a.mode,model_id=a.model_id,rep=a.rep,stage=a.stage,full_model=True,HTTP=True,
   files={p.name:sha(p) for p in a.out.glob('*.json')},performance_promotion=False,serving_promotion=False))
 except BaseException as e:
  save(a.out/'failure.json',dict(type=type(e).__name__,message=str(e),server_exit=proc.poll()));raise
 finally:
  if proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
  log.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--model-id',required=True);p.add_argument('--mem-fraction',type=float,required=True)
 p.add_argument('--mode',choices=['pristine','off','cap','guarded'],required=True);p.add_argument('--stage',choices=['smoke','performance','correctness'],required=True);p.add_argument('--rep',type=int,default=0)
 asyncio.run(main_async(p.parse_args()))
