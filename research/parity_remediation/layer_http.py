"""Bounded full-model correctness diagnostics; no speedup claims from these calls."""
from __future__ import annotations
import argparse, asyncio, hashlib, json, os, signal, socket, subprocess, sys, time
from pathlib import Path
import aiohttp

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, data):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n'); tmp.replace(p)

def checked_json(path):
    path=Path(path); receipt=json.loads((path.parent/'complete.json').read_text())
    if receipt.get('files',{}).get(path.name)!=sha(path):
        raise ValueError('historical hash mismatch: '+str(path))
    return json.loads(path.read_text())

def first_difference(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)) if len(a)!=len(b) else None)

async def run(a):
    if a.out.exists(): raise FileExistsError('preserve evidence')
    a.out.mkdir(parents=True)
    sys.path.insert(0,str(a.historic_root/'source'))
    sys.path.insert(0,str(a.historic_root/'serving-source-v3'))
    from http_bench import batch,health,workloads
    import http_bench
    from stream_evidence import record_request
    original_request=http_bench.request
    async def evidence_request(session,url,cell,origin):
        index=evidence_request.count;evidence_request.count+=1
        rec=await record_request(session,url,cell,cell['id'])
        save(a.out/f'wire-{index:04d}.json',rec)
        if not rec['complete']:raise RuntimeError('incomplete stream; raw wire receipt '+str(index))
        # Actual client SSE arrival stamps; diagnostic workload, not a performance repeat.
        stamp=time.perf_counter()-origin
        return dict(id=cell['id'],input_tokens=len(cell['input_ids']),tokens=rec['tokens'],events=[],
            token_times=rec['token_arrival_seconds'],ttft=rec['token_arrival_seconds'][0],
            tpot=(rec['token_arrival_seconds'][-1]-rec['token_arrival_seconds'][0])/(len(rec['tokens'])-1) if len(rec['tokens'])>1 else 0.,
            latency=rec['elapsed_seconds'],request_start=stamp-rec['elapsed_seconds'],completion=stamp,
            finish_reason=rec['finish_reason'],diagnostic_timing_not_performance=True)
    evidence_request.count=0
    http_bench.request=evidence_request
    import flashinfer,sglang
    old=checked_json(a.historic_root/'serving-runs/1639805/pristine/environment.json')
    header=Path(flashinfer.__file__).parent/'data/include/flashinfer/attention/prefill.cuh'
    expected=checked_json(a.historic_root/f'serving-runs/1639805/{a.mode}/environment.json')
    if flashinfer.__version__!='0.7.0' or sglang.__version__!=old['sglang'] or sha(header)!=expected['prefill_sha256']:
        raise RuntimeError('execution context differs')
    cmd=list(old['command']);cmd[0]=sys.executable
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    cmd[cmd.index('--port')+1]=str(port)
    cmd=[sys.executable,str(Path(__file__).with_name('layer_server_entry.py'))]+cmd[3:]
    cmd.append('--disable-overlap-schedule')
    cmd.extend(['--random-seed','42'])
    if a.deterministic:cmd.append('--enable-deterministic-inference')
    url=f'http://127.0.0.1:{port}'
    model=Path(cmd[cmd.index('--model-path')+1])
    if sha(model/'config.json')!=old['config_sha256']:raise ValueError('model config changed')
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SGLANG_ENABLE_JIT_DEEPGEMM='0',SGI_TRACE_PATH=str(a.out/'trace'),SGI_LAYER_TARGETS=str(Path(__file__).with_name('layer-targets.json')))
    receipt=dict(mode=a.mode,rep=a.rep,deterministic=a.deterministic,command=cmd,
        source_sha256=sha(__file__),header_sha256=sha(header),sglang=sglang.__version__,
        diagnostic_only=True,performance_promoted=False,serving_promoted=False,
        historical_environment_sha256=sha(a.historic_root/'serving-runs/1639805/pristine/environment.json'))
    save(a.out/'environment.json',receipt)
    log=(a.out/'server.log').open('w')
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    save(a.out/'owned_process.json',dict(pid=proc.pid,port=port))
    try:
        receipt['startup_seconds']=await health(url,proc,720);save(a.out/'environment.json',receipt)
        warm=dict(name='warmup',concurrency=2,cells=[dict(id=f'warm-{j}',input_ids=[500+j]*256,output_tokens=8) for j in range(2)])
        result=await batch(url,warm);save(a.out/'warmup.json',result)
        if result['errors']:raise RuntimeError('warmup failed')
        for b in range(3):
            for work in workloads():
                historical=checked_json(a.historic_root/f'serving-runs/1639805/pristine/{work["name"]}-b{b}.json')
                result=await batch(url,work)
                if result['workload_sha256']!=historical['workload_sha256']:raise ValueError('workload drift')
                save(a.out/f'{work["name"]}-b{b}.json',result)
                if result['errors']:raise RuntimeError('request failure')
                print('DIAGNOSTIC',a.mode,a.rep,work['name'],b,len(result['requests']),flush=True)
        oldp=checked_json(a.historic_root/'serving-runs/1639805/pristine/decode-b1.json')
        oldc=checked_json(a.historic_root/'serving-runs/1639805/cap/decode-b1.json')
        sources={r['id']:r for r in workloads()[1]['cells']}
        pd={r['id']:r for r in oldp['requests']};cd={r['id']:r for r in oldc['requests']}
        for rid,index in [('decode-11',74),('decode-13',114)]:
            if first_difference(pd[rid]['tokens'],cd[rid]['tokens'])!=index:raise ValueError('wrong exposed witness')
            for trial in range(2):
                work=dict(name='isolated',concurrency=1,cells=[dict(sources[rid],id=f'isolated-{rid}-{trial}')])
                result=await batch(url,work);save(a.out/f'isolated-{rid}-{trial}.json',result)
                if result['errors']:raise RuntimeError('isolated request failure')
            # Same prefix only. Does not force the original incremental decode execution.
            prefix=sources[rid]['input_ids']+pd[rid]['tokens'][:index]
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
                for trial in range(2):
                    payload=dict(rid=f'prefix-{rid}-{trial}',input_ids=prefix,stream=False,
                        sampling_params=dict(temperature=0,max_new_tokens=1,ignore_eos=True),
                        return_logprob=True,top_logprobs_num=8,return_text_in_logprobs=False)
                    async with session.post(url+'/generate',json=payload) as resp:
                        text=await resp.text()
                        if resp.status!=200:raise RuntimeError(f'prefix HTTP {resp.status}: '+text[:1000])
                        result=json.loads(text)
                    save(a.out/f'prefix-{rid}-{trial}.json',dict(prefix_sha256=hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
                        historical_choices=[pd[rid]['tokens'][index],cd[rid]['tokens'][index]],response=result))
        save(a.out/'complete.json',dict(complete=True,diagnostic_only=True,mode=a.mode,rep=a.rep,
            deterministic=a.deterministic,files={p.name:sha(p) for p in a.out.glob('*.json')},serving_promoted=False))
    except BaseException as exc:
        save(a.out/'failure.json',dict(type=type(exc).__name__,message=str(exc),server_exit=proc.poll()));raise
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
        log.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--historic-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['pristine','cap'],required=True)
    p.add_argument('--rep',type=int,choices=[0,1],required=True);p.add_argument('--deterministic',action='store_true')
    asyncio.run(run(p.parse_args()))
