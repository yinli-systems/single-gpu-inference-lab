"""Direct eager attention-stack timing, NOT full LLM serving.

All arms reuse plans equally. Frozen native weights; independent layer tensors.
Caller supplies pinned geometry.py/native.py and the hash-checked existing model.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
import traceback
from contracts import PlanKey,PolicyCache,bounded_policy,scratch_bytes,fresh_corpus,corpus_hash,POLICIES
from geometry import Shape,corpus
from native import Native,sha

ARMS=('auto','fixed_budget','native_cycle','native_run')
SEQUENCES={'unchanged':(0,0,0,0),'alternating':(0,1,0,1),'eviction':(0,1,2,0)}
EXPECTED_MODEL_SELECTION='d41c87f9e2f1c8eeac8ab411f63925e0eb2781f0229f8d7974ae6b6463a81f1c'


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def train_fixed(campaign,gpu,sms):
    """TRAIN-only budget/reuse fixed choices; old measurements form a cost proxy.

    proxy=cycle+(r-1)*run. This trains a fixed comparator, not a reuse result.
    Direct new timing scores this comparator and all other arms identically.
    """
    groups=defaultdict(list)
    for f in sorted((campaign/'runs').glob('full-*/measurements.jsonl')):
        m=json.loads(f.with_name('summary.json').read_text())
        if m['gpu']!=gpu:continue
        if sha(f)!=m['measurements_sha256']:raise ValueError('changed TRAIN evidence')
        for line in f.read_text().splitlines():
            r=json.loads(line)
            if r['shape']['split']=='train' and r['hq']==32:
                if r['status']!='complete':raise ValueError('missing TRAIN observation')
                groups[(r['shape']['name'],r['policy'])].append(r)
    by_shape=defaultdict(dict)
    for (name,p),rs in groups.items():
        if len(rs)!=3:raise ValueError('three TRAIN process replicates required')
        s=rs[0]['shape'];by_shape[name][p]=(s,statistics.median(r['median_ms'] for r in rs),statistics.median(r['median_cycle_ms'] for r in rs))
    if len(by_shape)!=72:raise ValueError('72 TRAIN shapes required')
    selected={}
    for budget in (128*1024**2,512*1024**2):
        for repeat in (1,2,4,8,16,36):
            scores={}
            for p in POLICIES:
                values=[]
                for policies in by_shape.values():
                    s=policies['auto'][0]
                    mode=bounded_policy(tuple(s['q']),tuple(s['k']),32,8,sms,p,budget)
                    _,run,cycle=policies[mode]
                    values.append(math.log(cycle+(repeat-1)*run))
                scores[p]=statistics.mean(values)
            choice=min(POLICIES,key=lambda p:(scores[p],POLICIES.index(p)))
            selected[f'{budget}:{repeat}']={'policy':choice,'TRAIN_proxy_scores':scores}
    return selected


def options(mode,torch):
    kw=dict(causal=True,q_data_type=torch.float16,kv_data_type=torch.float16)
    if mode=='none':kw['disable_split_kv']=True
    elif mode!='auto':kw['fixed_split_size']=int(mode[1:])
    return kw


def main():
    p=argparse.ArgumentParser()
    for name in ('out','native','analysis','campaign'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stage',choices=['canary','test'],default='canary')
    p.add_argument('--replicate',type=int,default=0);p.add_argument('--blocks',type=int,default=4)
    a=p.parse_args()
    if a.blocks<3:raise ValueError('at least three blocks')
    a.out.mkdir(parents=True,exist_ok=False)
    import torch,flashinfer
    if flashinfer.__version__!='0.6.18':raise ValueError('pinned backend required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    hw=torch.cuda.get_device_properties(0);gpu=hw.name;sms=hw.multi_processor_count
    if sms not in (128,170):raise ValueError('unregistered GPU')
    native=Native(a.native)
    if sha(a.analysis/'model_selection.json')!=EXPECTED_MODEL_SELECTION:raise ValueError('changed weights/selection')
    if native.info['model_selection_sha256']!=EXPECTED_MODEL_SELECTION:raise ValueError('changed native artifact')
    if not json.loads((a.native/'validation.json').read_text())['all_policies_identical']:raise ValueError('native parity not certified')
    frozen=train_fixed(a.campaign,gpu,sms);save(a.out/'TRAIN_fixed.json',frozen)
    seed=20261011+a.replicate;torch.manual_seed(seed)
    if a.stage=='canary':
        wanted={'Q512-K1536-n8-A','Q2048-K6144-n8-A','discovery-eq255-A','discovery-eq255-B'}
        bases=[s for s in corpus() if s.name in wanted]
        if len(bases)!=4:raise ValueError('missing canary fixture')
        repeat_counts=(1,36)
        sequences=('unchanged','eviction')
    else:
        bases=[Shape(r['name'],r['family'],r['split'],tuple(r['q']),tuple(r['k'])) for r in fresh_corpus() if r['state']==0]
        repeat_counts=(1,2,4,8,16,36);sequences=tuple(SEQUENCES)
    random.Random(seed).shuffle(bases)
    root=Path(flashinfer.__file__).parent
    source_hash=sha(root/'prefill.py')
    env=f'{gpu}|SM{sms}|torch{torch.__version__}|CUDA{torch.version.cuda}|FI{flashinfer.__version__}|{source_hash}|FA2|native{sha(a.native/"manifest.json")}'
    manifest=dict(kind='direct_matched_eager_attention_stack_reuse',stage=a.stage,seed=seed,replicate=a.replicate,
        gpu=gpu,sms=sms,torch=torch.__version__,cuda=torch.version.cuda,flashinfer=flashinfer.__version__,
        backend='fa2',dtype='float16',heads=[32,8],head_dim=128,max_layers=36,
        mode='eager',arms=ARMS,repeat_counts=repeat_counts,sequences=sequences,policy_cache_capacity=2,
        scratch_ceilings=[128*1024**2,512*1024**2],corpus_sha256=corpus_hash(),
        model_selection_sha256=EXPECTED_MODEL_SELECTION,native_manifest_sha256=sha(a.native/'manifest.json'),
        source_sha256={f:sha(Path(__file__).with_name(f)) for f in ('contracts.py','measure_reuse.py')},
        prefill_sha256=source_hash,blocks=a.blocks,full_model=False,layer_data_distinct=True,
        full_vllm=False,promotion_target='36 layers native_cycle vs fixed_budget',
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    save(a.out/'manifest.json',manifest)
    rows=[];started=time.perf_counter();allocated_start=torch.cuda.memory_allocated();failures=[]
    for base in bases:
        build_start=time.perf_counter()
        if a.stage=='test':
            states=[Shape(r['name'],r['family'],r['split'],tuple(r['q']),tuple(r['k'])) for r in fresh_corpus() if r['family']==base.family and r['n']==len(base.q)]
        else:
            states=[base]+[Shape(base.name+f'-change{i}',base.family,'diagnostic',base.q,tuple(x+i*17 for x in base.k)) for i in (1,2)]
        qsize=max(sum(s.q) for s in states);ksize=max(sum(s.q)+sum(s.k) for s in states)
        # Every layer has distinct storage and random values. Attention only; no model weights.
        qs=[torch.randn((qsize,32,128),device='cuda',dtype=torch.float16) for _ in range(36)]
        ks=[torch.randn((ksize,8,128),device='cuda',dtype=torch.float16) for _ in range(36)]
        vs=[torch.randn_like(k) for k in ks]
        outs=[torch.empty_like(q) for q in qs]
        layer_bytes=sum(t.numel()*t.element_size() for t in qs+ks+vs+outs)
        metadata=[]
        for s in states:
            qo=[0];ko=[0]
            for q,k in zip(s.q,s.k):qo.append(qo[-1]+q);ko.append(ko[-1]+q+k)
            metadata.append((torch.tensor(qo,device='cuda',dtype=torch.int32),torch.tensor(ko,device='cuda',dtype=torch.int32),qo,ko))
        torch.cuda.synchronize();build_seconds=time.perf_counter()-build_start
        for budget in manifest['scratch_ceilings']:
            torch.cuda.reset_peak_memory_stats()
            scratch=torch.empty(budget,device='cuda',dtype=torch.uint8)
            wrappers={arm:flashinfer.BatchPrefillWithRaggedKVCacheWrapper(scratch,kv_layout='NHD',backend='fa2') for arm in ARMS}
            def select(arm,s,repeat):
                if arm=='auto':candidate='auto'
                elif arm=='fixed_budget':candidate=frozen[f'{budget}:{repeat}']['policy']
                else:
                    target='median_cycle_ms' if arm=='native_cycle' else 'median_ms'
                    candidate=native.choose(gpu+'|'+target+'|causal',s,32,8,sms)
                return bounded_policy(s.q,s.k,32,8,sms,candidate,budget)
            def run_layer(w,s,si,layer):
                totalq=sum(s.q);totalk=sum(s.q)+sum(s.k)
                w.run(qs[layer][:totalq],ks[layer][:totalk],vs[layer][:totalk],out=outs[layer][:totalq])
            # Check each actual selected policy, all states and all 36 distinct layers.
            checks=0;maxdiff=0.;nref=0;maxref=0.
            modes_per_state=[]
            for si,s in enumerate(states):
                modes={'auto'}
                for repeat in repeat_counts:
                    modes.update(select(arm,s,repeat) for arm in ARMS)
                modes_per_state.append(sorted(modes))
                qptr,kptr,qo,ko=metadata[si]
                w=wrappers['auto'];w.plan(qptr,kptr,32,8,128,**options('auto',torch))
                references=[]
                for layer in range(36):
                    run_layer(w,s,si,layer);references.append(outs[layer][:sum(s.q)].clone())
                for layer in (0,35):
                    for ri,(ql,depth) in enumerate(zip(s.q,s.k)):
                        for qi in sorted({0,ql-1}):
                            for h in (0,31):
                                end=ko[ri]+depth+qi+1;kh=h//4
                                qq=qs[layer][qo[ri]+qi,h].float();kk=ks[layer][ko[ri]:end,kh].float();vv=vs[layer][ko[ri]:end,kh].float()
                                expected=torch.softmax(kk@qq/math.sqrt(128),0)@vv
                                actual=references[layer][qo[ri]+qi,h].float()
                                torch.testing.assert_close(actual,expected,atol=.005,rtol=.02)
                                maxref=max(maxref,float((actual-expected).abs().max()));nref+=1
                for mode in modes:
                    if scratch_bytes(s.q,s.k,32,8,sms,mode)>budget:raise AssertionError('ceiling violated')
                    w=wrappers['native_cycle'];w.plan(qptr,kptr,32,8,128,**options(mode,torch))
                    for layer in range(36):
                        run_layer(w,s,si,layer);actual=outs[layer][:sum(s.q)]
                        torch.testing.assert_close(actual,references[layer],atol=.005,rtol=.02)
                        maxdiff=max(maxdiff,float((actual-references[layer]).abs().max()));checks+=1
                del references
            def episode(arm,repeat,sequence):
                cache=PolicyCache(2);active=None;planned=0;calls=0;chosen=[]
                w=wrappers[arm]
                for si in SEQUENCES[sequence]:
                    s=states[si]
                    key=PlanKey(s.q,s.k,32,8,sms,environment=env,scratch_ceiling=budget)
                    mode=cache.resolve(key,lambda:select(arm,s,repeat))
                    identity=(key,mode)
                    if active!=identity:
                        active=None
                        qptr,kptr,_,_=metadata[si]
                        w.plan(qptr,kptr,32,8,128,**options(mode,torch));planned+=1
                        active=identity
                    for layer in range(repeat):run_layer(w,s,si,layer);calls+=1
                    chosen.append(mode)
                return dict(plans=planned,runs=calls,hits=cache.hits,misses=cache.misses,evictions=cache.evictions,policies=chosen)
            for repeat in repeat_counts:
                for sequence in sequences:
                    values={arm:[] for arm in ARMS};counters={}
                    for arm in ARMS:episode(arm,repeat,sequence)
                    torch.cuda.synchronize()
                    for block in range(a.blocks):
                        order=list(ARMS);random.Random(f'{seed}:{base.name}:{repeat}:{sequence}:{budget}:{block}').shuffle(order)
                        pair={arm:[] for arm in ARMS}
                        for arm in order+order[::-1]:
                            torch.cuda.synchronize();tick=time.perf_counter_ns()
                            count=episode(arm,repeat,sequence);torch.cuda.synchronize()
                            elapsed=(time.perf_counter_ns()-tick)/1e6
                            if arm in counters and count!=counters[arm]:raise AssertionError('stateful nondeterminism')
                            counters[arm]=count;pair[arm].append(elapsed)
                        for arm in ARMS:values[arm].append(statistics.mean(pair[arm]))
                    row=dict(base=base.record(),gpu=gpu,replicate=a.replicate,layers=repeat,sequence=sequence,
                        scratch_ceiling=budget,layer_tensor_bytes=layer_bytes,build_seconds=build_seconds,
                        peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved(),
                        arms_ms=values,median_ms={arm:statistics.median(v) for arm,v in values.items()},counters=counters,
                        selected_modes_checked=modes_per_state,full_tensor_checks=checks,max_abs=maxdiff,
                        fp32_vectors=nref,fp32_max_abs=maxref,correctness_pass=True)
                    with (a.out/'measurements.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    rows.append(row)
                    print('REUSE_CELL',base.name,repeat,sequence,budget,row['median_ms'],flush=True)
            del wrappers,scratch,w
            torch.cuda.empty_cache()
        del qs,ks,vs,outs,metadata
        torch.cuda.empty_cache()
    summary=dict(manifest,complete=True,rows=len(rows),failures=failures,elapsed_seconds=time.perf_counter()-started,
                 measurements_sha256=sha(a.out/'measurements.jsonl'))
    save(a.out/'summary.json',summary);print('REUSE_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
