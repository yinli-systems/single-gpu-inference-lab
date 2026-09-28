"""Real vLLMEngine measurement; no model substitutions or reward-based retries."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

ARMS=('auto','fixed_budget','native_cycle')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def save(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def quantiles(values):
    values=sorted(values)
    if not values or any(not math.isfinite(x) for x in values):raise ValueError('finite nonempty sample required')
    result={}
    for name,q in (('p50',.5),('p95',.95),('p99',.99)):
        x=q*(len(values)-1);lo=int(x);hi=min(lo+1,len(values)-1)
        result[name]=values[lo]+(values[hi]-values[lo])*(x-lo)
    return result


def observe(record,tokens,finished,stamp):
    previous=record['token_ids'];tokens=list(tokens)
    if stamp<record['admission'] or (record['token_times'] and stamp<record['token_times'][-1]):raise ValueError('nonmonotonic clock')
    if len(tokens)<len(previous) or tokens[:len(previous)]!=previous:raise ValueError('cumulative token prefix changed')
    added=len(tokens)-len(previous)
    record['token_times'].extend([stamp]*added);record['token_ids']=tokens;record['finished']=bool(finished)
    return added


def summarize(cells,records,elapsed):
    ids=[c['id'] for c in cells]
    if len(set(ids))!=len(ids) or set(ids)!=set(records):raise ValueError('planned request coverage mismatch')
    if elapsed<=0 or not math.isfinite(elapsed):raise ValueError('invalid elapsed time')
    expected={c['id']:c for c in cells}
    for rid,r in records.items():
        times=r['token_times'];cell=expected[rid]
        if not r['finished'] or len(times)!=cell['max_tokens'] or len(r['token_ids'])!=len(times):raise ValueError('incomplete request')
        if not times or times[-1]>elapsed+1e-6:raise ValueError('invalid completion timestamp')
        r['ttft']=times[0]-r['admission'];r['offered_ttft']=times[0]-cell['arrival']
        r['latency']=times[-1]-r['admission'];r['offered_latency']=times[-1]-cell['arrival']
        r['tpot']=(times[-1]-times[0])/(len(times)-1) if len(times)>1 else 0.
        r['strict_slo']=r['offered_ttft']<=2. and r['tpot']<=.05
        r['lenient_slo']=r['offered_ttft']<=5. and r['tpot']<=.1
    rr=list(records.values());tokens=sum(len(r['token_ids']) for r in rr)
    strict=sum(r['strict_slo'] for r in rr);lenient=sum(r['lenient_slo'] for r in rr)
    return dict(requests=len(rr),output_tokens=tokens,elapsed_seconds=elapsed,
        tokens_per_second=tokens/elapsed,requests_per_second=len(rr)/elapsed,
        ttft_seconds=quantiles([r['ttft'] for r in rr]),offered_ttft_seconds=quantiles([r['offered_ttft'] for r in rr]),
        tpot_seconds=quantiles([r['tpot'] for r in rr]),request_latency_seconds=quantiles([r['latency'] for r in rr]),
        offered_latency_seconds=quantiles([r['offered_latency'] for r in rr]),
        strict_slo_count=strict,lenient_slo_count=lenient,
        strict_slo_goodput=strict/elapsed,lenient_slo_goodput=lenient/elapsed)


def run(engine,router,name,out,phase):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from engine_screen import workload
    import torch
    cells=workload(name);pending=list(cells);records={};steps=[]
    torch.cuda.synchronize();start=time.perf_counter()
    while pending or engine.has_unfinished_requests():
        now=time.perf_counter()-start
        while pending and pending[0]['arrival']<=now:
            cell=pending.pop(0);admitted=time.perf_counter()-start
            engine.add_request(cell['id'],{'prompt_token_ids':cell['prompt']},
                SamplingParams(temperature=0.,max_tokens=cell['max_tokens'],ignore_eos=True,
                               seed=20261020,output_kind=RequestOutputKind.CUMULATIVE))
            records[cell['id']]=dict(id=cell['id'],scheduled_arrival=cell['arrival'],admission=admitted,
                prompt_tokens=len(cell['prompt']),expected_output_tokens=cell['max_tokens'],
                token_ids=[],token_times=[],finished=False)
        if not engine.has_unfinished_requests():
            if pending:time.sleep(min(.005,max(0.,pending[0]['arrival']-(time.perf_counter()-start))))
            continue
        before=time.perf_counter()-start;outputs=engine.step();stamp=time.perf_counter()-start
        events=[]
        for output in outputs:
            if output.request_id not in records:raise ValueError('unknown output request')
            if len(output.outputs)!=1:raise ValueError('one hypothesis required')
            added=observe(records[output.request_id],output.outputs[0].token_ids,output.finished,stamp)
            events.append(dict(id=output.request_id,added=added,finished=bool(output.finished)))
        steps.append(dict(started=before,completed=stamp,outputs=events))
        if stamp>240:raise TimeoutError('240-second workload bound')
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    metrics=summarize(cells,records,elapsed)
    result=dict(workload=name,phase=phase,complete=True,**metrics,requests_detail=list(records.values()),router=router.report(),
        workload_sha256=hashlib.sha256(json.dumps(cells,sort_keys=True).encode()).hexdigest(),
        timing='in-process engine observation; no filesystem writes inside interval',HTTP_network=False)
    save(out/f'{phase}-{name}.json',result)
    with (out/f'{phase}-{name}-steps.jsonl').open('w') as f:
        for s in steps:f.write(json.dumps(s)+'\n')
    return result


def main():
    p=argparse.ArgumentParser()
    for key in ('model','native','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--replicate',type=int,default=0)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    os.environ['VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE']=str(512*1024**2)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch,flashinfer,vllm
    from paged_router import PagedRouter
    router=PagedRouter(a.native,fixed='none').install();router.reset(a.arm)
    from vllm import EngineArgs,LLMEngine
    from vllm.config import AttentionConfig
    args=EngineArgs(model=str(a.model),dtype='half',tensor_parallel_size=1,enforce_eager=True,
        enable_prefix_caching=False,max_model_len=16384,max_num_batched_tokens=1024,
        max_num_seqs=32,block_size=16,num_gpu_blocks_override=2048,gpu_memory_utilization=.9,
        seed=20261020,disable_log_stats=True,
        attention_config=AttentionConfig(backend='FLASHINFER',use_trtllm_attention=False))
    started=time.perf_counter();engine=LLMEngine.from_engine_args(args);load_seconds=time.perf_counter()-started
    results=[]
    for name in ('burst','staggered'):
        router.reset(a.arm);run(engine,router,name,a.out,'warmup')
        router.reset(a.arm);results.append(run(engine,router,name,a.out,'measured'))
    if any(r['router']['counts'].get('qualified_plans',0)==0 for r in results):
        raise RuntimeError('each workload must actually execute the registered paged prefill adapter')
    site=Path(flashinfer.__file__).parent;here=Path(__file__).parent
    cache_cfg=getattr(getattr(engine,'vllm_config',None),'cache_config',None)
    report=dict(complete=True,arm=a.arm,replicate=a.replicate,model=str(a.model),
        model_config_sha256=sha(a.model/'config.json'),vllm=vllm.__version__,torch=torch.__version__,
        flashinfer=flashinfer.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(),
        dtype='float16',eager=True,requested_KV_blocks=2048,actual_config_KV_blocks=getattr(cache_cfg,'num_gpu_blocks',None),
        page_size=16,scratch_MiB=512,load_seconds=load_seconds,HTTP_network=False,
        full_model=True,greedy_parity='must be compared across complete arms',
        source_sha256={str(x):sha(x) for x in (here/'measure_engine.py',here.parent/'paged_router.py',here.parent/'contracts.py',here.parent/'engine_screen.py',site/'prefill.py',site/'sampling.py')},
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        workloads=[{k:v for k,v in r.items() if k not in ('requests_detail','router')} for r in results])
    save(a.out/'summary.json',report);print('REAL_ENGINE_COMPLETE',json.dumps(report),flush=True)
    if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':main()
