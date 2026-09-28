"""Actual offline vLLMEngine screen; in-process, not HTTP serving.

Outputs are scored separately after all matched arms complete. No online tuning.
"""
from __future__ import annotations
import argparse,hashlib,json,os,random,statistics,time
from pathlib import Path


def percentiles(values):
    import numpy as np
    return {k:float(np.quantile(values,q)) for k,q in (('p50',.5),('p95',.95),('p99',.99))}


def workload(name):
    rng=random.Random(20261020+(name=='staggered'))
    lengths=[512,1024,2048,4096]*2 if name=='burst' else [256,1024,4096,8192]*3
    return [dict(id=f'{name}-{i}',prompt=[rng.randrange(100,10000) for _ in range(n)],
        max_tokens=96 if name=='burst' else (48 if i%2==0 else 96),
        arrival=0. if name=='burst' else i*.05) for i,n in enumerate(lengths)]


def run(engine,router,name,out,warmup=False):
    from vllm import SamplingParams
    cells=workload(name)
    if warmup:cells=[dict(id='warmup',prompt=[110]*256,max_tokens=8,arrival=0.)]
    start=time.perf_counter();pending=list(cells);records={};step=0
    while pending or engine.has_unfinished_requests():
        now=time.perf_counter()-start
        while pending and pending[0]['arrival']<=now:
            cell=pending.pop(0);request_start=time.perf_counter()
            engine.add_request(cell['id'],{'prompt_token_ids':cell['prompt']},
                SamplingParams(temperature=0.,max_tokens=cell['max_tokens'],ignore_eos=True,seed=20261020))
            records[cell['id']]=dict(id=cell['id'],scheduled_arrival=cell['arrival'],
                admission=request_start-start,prompt_tokens=len(cell['prompt']),expected_output_tokens=cell['max_tokens'],
                token_times=[],token_ids=[],finished=False)
        if not engine.has_unfinished_requests():
            time.sleep(min(.005,max(0,pending[0]['arrival']-(time.perf_counter()-start))))
            continue
        before=time.perf_counter();outputs=engine.step();observed=time.perf_counter();step+=1
        step_record=dict(step=step,started=before-start,completed=observed-start,outputs=[])
        for output in outputs:
            record=records[output.request_id];tokens=list(output.outputs[0].token_ids)
            if len(tokens)<len(record['token_ids']):raise RuntimeError('nonmonotonic engine output')
            added=len(tokens)-len(record['token_ids'])
            record['token_times'].extend([observed-start]*added);record['token_ids']=tokens
            record['finished']=bool(output.finished)
            step_record['outputs'].append(dict(id=output.request_id,added_tokens=added,finished=bool(output.finished)))
        if not warmup:
            with (out/(name+'-steps.jsonl')).open('a') as f:f.write(json.dumps(step_record)+'\n')
        if observed-start>240:raise TimeoutError('bounded engine screen:240s workload limit')
    elapsed=time.perf_counter()-start
    for record in records.values():
        ts=record['token_times']
        if not record['finished'] or len(ts)!=record['expected_output_tokens']:raise RuntimeError('incomplete request')
        record['ttft']=ts[0]-record['admission'];record['latency']=ts[-1]-record['admission']
        record['tpot']=(ts[-1]-ts[0])/(len(ts)-1) if len(ts)>1 else 0.
        record['offered_ttft']=ts[0]-record['scheduled_arrival']
        record['slo_2s_50ms']=record['offered_ttft']<=2. and record['tpot']<=.05
        record['slo_5s_100ms']=record['offered_ttft']<=5. and record['tpot']<=.1
    total=sum(len(r['token_ids']) for r in records.values())
    result=dict(workload=name,requests=len(records),complete=True,elapsed_seconds=elapsed,output_tokens=total,
        tokens_per_second=total/elapsed,requests_per_second=len(records)/elapsed,
        ttft_seconds=percentiles([r['ttft'] for r in records.values()]),
        offered_ttft_seconds=percentiles([r['offered_ttft'] for r in records.values()]),
        tpot_seconds=percentiles([r['tpot'] for r in records.values()]),
        request_latency_seconds=percentiles([r['latency'] for r in records.values()]),
        strict_slo_goodput=sum(r['slo_2s_50ms'] for r in records.values())/elapsed,
        lenient_slo_goodput=sum(r['slo_5s_100ms'] for r in records.values())/elapsed,
        requests_detail=list(records.values()),router=router.report(),
        workload_sha256=hashlib.sha256(json.dumps(cells,sort_keys=True).encode()).hexdigest())
    if not warmup:(out/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True);p.add_argument('--native',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--arm',choices=['auto','fixed_budget','native_cycle'],required=True)
    p.add_argument('--replicate',type=int,default=0);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    os.environ['VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE']=str(512*1024**2)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch,flashinfer
    from paged_router import PagedRouter
    router=PagedRouter(a.native,fixed='none').install();router.reset(a.arm)
    from vllm import EngineArgs,LLMEngine
    from vllm.config import AttentionConfig
    args=EngineArgs(model=str(a.model),dtype='half',tensor_parallel_size=1,
        enforce_eager=True,enable_prefix_caching=False,max_model_len=16384,
        max_num_batched_tokens=1024,max_num_seqs=32,block_size=16,num_gpu_blocks_override=2048,
        gpu_memory_utilization=.9,seed=20261020,disable_log_stats=True,
        attention_config=AttentionConfig(backend='FLASHINFER',use_trtllm_attention=False))
    tick=time.perf_counter();engine=LLMEngine.from_engine_args(args);load_seconds=time.perf_counter()-tick
    run(engine,router,'burst',a.out,warmup=True)
    results=[]
    for name in ('burst','staggered'):
        router.reset(a.arm);results.append(run(engine,router,name,a.out))
    if sum(r['router']['counts'].get('qualified_plans',0) for r in results)==0:
        raise RuntimeError('vLLM did not execute the qualified paged adapter; no claim possible')
    import vllm
    summary=dict(arm=a.arm,replicate=a.replicate,model=str(a.model),vllm=vllm.__version__,
        torch=torch.__version__,cuda=torch.version.cuda,flashinfer=flashinfer.__version__,
        dtype='float16',eager=True,KV_blocks=2048,page_size=16,scratch_MiB=512,
        GPU=torch.cuda.get_device_name(),load_seconds=load_seconds,complete=True,
        output_parity_not_yet_compared=True,HTTP_network=False,
        workloads=[{k:v for k,v in r.items() if k not in ('requests_detail','router')} for r in results],
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        source_sha256={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in ('engine_screen.py','paged_router.py','contracts.py')})
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('REAL_ENGINE_SCREEN',json.dumps(summary),flush=True)


if __name__=='__main__':main()
