"""Bounded real-server stale cleanup reproduction, not an inference speed benchmark."""
from __future__ import annotations
import argparse,asyncio,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
import aiohttp
from evidence_contract import read_verified,file_hash
from stream_evidence import record_request

def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
async def run(a):
    if a.out.exists():raise FileExistsError('preserve failure')
    a.out.mkdir(parents=True)
    sys.path.insert(0,str(a.old/'source'));sys.path.insert(0,str(a.old/'serving-source-v3'))
    from http_bench import health,workloads
    import flashinfer
    old=read_verified(a.old/'serving-runs/1639805/pristine/environment.json')
    header=Path(flashinfer.__file__).parent/'data/include/flashinfer/attention/prefill.cuh'
    if file_hash(header)!=old['prefill_sha256']:raise RuntimeError('not pristine')
    cmd=[sys.executable,str(Path(__file__).with_name('abort_server_entry.py'))]+old['command'][3:]
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    cmd[cmd.index('--port')+1]=str(port);cmd+=['--random-seed','42','--enable-deterministic-inference']
    env=dict(os.environ,SGI_ABORT_MODE=a.mode,SGI_ABORT_LOG=str(a.out/'cleanup.jsonl'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    log=(a.out/'server.log').open('w');proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(mode=a.mode,command=cmd,source_sha256=file_hash(__file__),diagnostic_only=True,
        cleanup_patch_only=True,token_value_divergence_fixed=False,upstream_promoted=False)
    save(a.out/'environment.json',receipt);results=[]
    try:
        await health(f'http://127.0.0.1:{port}',proc,720)
        cell=workloads()[1]['cells'][11]
        # Each epoch sends two sequential requests with a shared RID. No adaptive repeats.
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
            for epoch in range(2):
                for call in range(2):
                    rec=await record_request(session,f'http://127.0.0.1:{port}',cell,f'reuse-{epoch}')
                    rec.update(epoch=epoch,call=call,condition='reused-rid');results.append(rec)
                    save(a.out/f'reused-{epoch}-{call}.json',rec)
                await asyncio.sleep(2.2)
            for call in range(4):
                rec=await record_request(session,f'http://127.0.0.1:{port}',cell,f'unique-{call}')
                rec.update(call=call,condition='unique-rid');results.append(rec);save(a.out/f'unique-{call}.json',rec)
        await asyncio.sleep(2.2)
        save(a.out/'complete.json',dict(complete=True,diagnostic_only=True,mode=a.mode,
            request_count=len(results),incomplete_requests=sum(not r['complete'] for r in results),
            files={f.name:file_hash(f) for f in a.out.iterdir() if f.is_file() and f.suffix in ('.json','.jsonl')},
            inference_correctness_promoted=False))
    except BaseException as exc:
        save(a.out/'failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
        log.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--old',type=Path,required=True);p.add_argument('--mode',choices=['original','patched'],required=True);a=p.parse_args();asyncio.run(run(a))
