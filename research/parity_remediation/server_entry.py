"""Diagnostic-only sampler boundary snapshots inside an owned SGLang worker.
No tensor/weight mutation. CPU synchronization changes scheduling; timings invalid.
"""
import hashlib,json,os,runpy
from pathlib import Path

def install():
    path=os.environ.get('SGI_TRACE_PATH')
    if not path:return
    import torch
    from sglang.srt.model_executor.model_runner import ModelRunner
    old=ModelRunner.sample
    if getattr(old,'_sgi_trace',False):return
    histories={};counter=[0];dumps=[0]
    def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
    def tensor_list(x):
        if x is None:return None
        return x.detach().cpu().tolist() if hasattr(x,'detach') else list(x)
    def sample(self,logits_output,forward_batch):
        fb=forward_batch;bs=int(fb.batch_size)
        inputs=tensor_list(fb.input_ids);seqs=tensor_list(fb.seq_lens)[:bs]
        pools=tensor_list(fb.req_pool_indices)[:bs]
        decode=fb.forward_mode.is_decode()
        lens=[1]*bs if decode else tensor_list(fb.extend_seq_lens_cpu)
        row_histories=[];offset=0
        for i,(pool,seq) in enumerate(zip(pools,seqs)):
            n=lens[i] if lens is not None and len(lens)==bs else None
            current=None
            if n is not None and offset+n<=len(inputs):
                part=inputs[offset:offset+n];offset+=n;prev=histories.get(pool)
                if seq==n:current=list(part)
                elif prev is not None and len(prev)+n==seq:current=prev+part
            histories[pool]=current
            row_histories.append(digest(current) if current is not None else None)
        x=logits_output.next_token_logits
        record=dict(step=counter[0],batch_size=bs,forward_mode=str(fb.forward_mode),
            seq_lens=seqs,req_pool_indices=pools,input_ids=inputs,
            extend_seq_lens=lens,history_sha256=row_histories,
            all_histories_complete=all(v is not None for v in row_histories),
            diagnostic_only=True)
        cpu=x.detach()[:bs].to(device='cpu',dtype=torch.float32).contiguous()
        values,indices=torch.topk(cpu,8,dim=-1)
        record.update(top8_values=values.tolist(),top8_ids=indices.tolist(),
            logit_sha256=[hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in cpu])
        record['batch_signature']=digest({k:record[k] for k in ('forward_mode','seq_lens','input_ids','history_sha256')})
        root=Path(path);root.mkdir(parents=True,exist_ok=True)
        if decode and any(n in (201,202,203,241,242,243) for n in seqs) and dumps[0]<8:
            name=f'logits-{os.getpid()}-{counter[0]:06d}.pt'
            torch.save(dict(logits=cpu,metadata=record),root/name)
            record['full_logits_file']=name;dumps[0]+=1
        result=old(self,logits_output,forward_batch)
        record['sampled_ids']=tensor_list(result)
        with (root/f'steps-{os.getpid()}.jsonl').open('a') as f:
            f.write(json.dumps(record,allow_nan=False)+'\n')
        counter[0]+=1
        return result
    sample._sgi_trace=True;ModelRunner.sample=sample
install()
if __name__=='__main__':runpy.run_module('sglang.launch_server',run_name='__main__')
