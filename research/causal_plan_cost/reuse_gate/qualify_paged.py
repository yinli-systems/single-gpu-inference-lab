"""Numerical/interface qualification only, not a performance claim."""
import argparse,hashlib,json,math,time
from pathlib import Path
import torch,flashinfer
from paged_router import PagedRouter

p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.out.exists():raise FileExistsError('preserve previous qualification')
a.out.mkdir(parents=True)
torch.manual_seed(20261020);torch.backends.cuda.matmul.allow_tf32=False
torch.set_num_threads(2)
router=PagedRouter(a.native).install()
checks=fp32=0;worst=worst_ref=0.;reports=[];started=time.perf_counter()
qstates=[(255,385),(256,384),(128,512)]
kstates=[(2048,4096),(2065,4113),(2082,4130)]
for layout in ('NHD','HND'):
    scratch=torch.empty(512*1024**2,device='cuda',dtype=torch.uint8)
    w=flashinfer.BatchPrefillWithPagedKVCacheWrapper(scratch,layout,backend='fa2')
    page=16;maxpages=512
    qlayers=[torch.randn(640,32,128,device='cuda',dtype=torch.float16) for _ in range(36)]
    kval=[torch.randn(maxpages,page,8,128,device='cuda',dtype=torch.float16) for _ in range(36)]
    vval=[torch.randn_like(x) for x in kval]
    refs={}
    state_metadata=[]
    for si in (0,1,2,0):
        q=qstates[si];depth=kstates[si];total=[x+y for x,y in zip(q,depth)]
        npages=[(x+page-1)//page for x in total]
        indices=torch.randperm(maxpages,device='cuda',dtype=torch.int64)[:sum(npages)].to(torch.int32)
        state_metadata.append(dict(q=q,k=depth,qp=torch.tensor([0,q[0],sum(q)],dtype=torch.int32),
            ip=torch.tensor([0,npages[0],sum(npages)],dtype=torch.int32),idx=indices,
            last=torch.tensor([(x-1)%page+1 for x in total],dtype=torch.int32)))
    for arm in ('auto','fixed_budget','native_cycle'):
        router.reset(arm)
        for transition,s in enumerate(state_metadata):
            # The last transition has same logical lengths as the first but different physical pages.
            w.plan(s['qp'],s['ip'],s['idx'],s['last'],32,8,128,page,
                   causal=True,q_data_type=torch.float16,kv_data_type=torch.float16)
            for layer in range(36):
                K=kval[layer] if layout=='NHD' else kval[layer].transpose(1,2)
                V=vval[layer] if layout=='NHD' else vval[layer].transpose(1,2)
                out=w.run(qlayers[layer],(K,V))
                if arm=='auto':refs[transition,layer]=out.clone()
                else:
                    torch.testing.assert_close(out,refs[transition,layer],atol=.005,rtol=.02)
                    worst=max(worst,float((out-refs[transition,layer]).abs().max()));checks+=1
                if arm=='auto' and layer in (0,35):
                    for ri,(ql,depth) in enumerate(zip(s['q'],s['k'])):
                        ids=s['idx'][s['ip'][ri]:s['ip'][ri+1]].to(torch.int64)
                        logicalK=kval[layer][ids].reshape(-1,8,128)
                        logicalV=vval[layer][ids].reshape(-1,8,128)
                        qstart=int(s['qp'][ri])
                        for qi in (0,ql-1):
                            for h in (0,31):
                                upto=depth+qi+1
                                scores=logicalK[:upto,h//4].float()@qlayers[layer][qstart+qi,h].float()/math.sqrt(128)
                                expected=torch.softmax(scores,0)@logicalV[:upto,h//4].float()
                                actual=out[qstart+qi,h].float()
                                torch.testing.assert_close(actual,expected,atol=.005,rtol=.02)
                                worst_ref=max(worst_ref,float((actual-expected).abs().max()));fp32+=1
        state=router.report()
        if state['counts'].get('qualified_plans')!=4 or state['cache_hits']!=1 or state['cache_misses']!=3:raise AssertionError('page-aware plan lifetime/cache hit wrong')
        reports.append(dict(layout=layout,**state))
    del w,scratch,qlayers,kval,vval,refs,out,K,V
    torch.cuda.empty_cache()
result=dict(passed=True,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
    flashinfer=flashinfer.__version__,nonauto_full_layer_transition_checks=checks,selected_FP32_vectors=fp32,
    max_abs_vs_auto=worst,max_abs_FP32=worst_ref,atol=.005,rtol=.02,bitwise=False,
    physical_page_permutation_checked=True,same_geometry_new_data_checked=True,reports=reports,
    elapsed_seconds=time.perf_counter()-started,source_sha256={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in ('qualify_paged.py','paged_router.py','contracts.py')})
(a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print('PAGED_QUALIFIED',json.dumps(result),flush=True)
