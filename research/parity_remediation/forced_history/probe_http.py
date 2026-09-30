from __future__ import annotations
import argparse,asyncio,json,os,signal,socket,subprocess,time
from pathlib import Path
import aiohttp
from probe_processor import PositionProbeProcessor
async def health(url,p):
 start=time.monotonic()
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as s:
  while time.monotonic()-start<600:
   if p.poll() is not None:raise RuntimeError('server exit '+str(p.returncode))
   try:
    async with s.get(url+'/health') as r:
     if r.status==200:return
   except Exception:pass
   await asyncio.sleep(.5)
 raise TimeoutError('startup')
async def main(a):
 if a.out.exists():raise FileExistsError(a.out)
 a.out.mkdir(parents=True);poslog=a.out/'positions.jsonl'
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 url=f'http://127.0.0.1:{port}';cmd=[os.sys.executable,str(a.source/'probe_server_entry.py'),'--model-path',str(a.model),'--host','127.0.0.1','--port',str(port),'--attention-backend','flashinfer','--dtype','bfloat16','--mem-fraction-static','0.8','--context-length','8192','--chunked-prefill-size','1024','--max-running-requests','4','--cuda-graph-max-bs-decode','4','--page-size','16','--skip-tokenizer-init','--disable-radix-cache','--enable-custom-logit-processor','--random-seed','2026093074']
 env=dict(os.environ,SGI_FORCE_POSITION_LOG=str(poslog),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');log=(a.out/'server.log').open('w');p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 try:
  await health(url,p);forced=[101,202,303,404];payload=dict(rid='position-probe',input_ids=[500+i%17 for i in range(128)],sampling_params=dict(temperature=0,max_new_tokens=5,ignore_eos=True,custom_params={'forced_tokens':forced}),custom_logit_processor=PositionProbeProcessor.to_str(),stream=False)
  async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as s:
   async with s.post(url+'/generate',json=payload) as r:
    text=await r.text()
    if r.status!=200:raise RuntimeError(f'HTTP {r.status}: {text[:800]}')
  x=json.loads(text);tokens=x.get('output_ids',[]);rows=[json.loads(z) for z in poslog.read_text().splitlines()]
  result=dict(tokens=tokens,forced=forced,positions=rows,server_complete=len(tokens)==5,forced_prefix_matches=tokens[:len(forced)]==forced)
  (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
  if not result['forced_prefix_matches']:raise RuntimeError('probe forced prefix mismatch')
  (a.out/'complete.json').write_text(json.dumps(dict(complete=True,result=result),indent=2)+'\n')
 except BaseException as e:
  (a.out/'failure.json').write_text(json.dumps(dict(type=type(e).__name__,message=str(e),server_exit=p.poll()),indent=2)+'\n');raise
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
  log.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);asyncio.run(main(p.parse_args()))
