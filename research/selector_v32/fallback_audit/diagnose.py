"""Frozen host-tail/graph-identity diagnosis; no release eligibility."""
from __future__ import annotations
import argparse,gc,hashlib,itertools,json,math,os,random,resource,sys,time
from pathlib import Path


def digest_file(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');q.replace(p)
def _cuda_checked(result, op):
 if not isinstance(result, tuple) or not result:raise RuntimeError(f'{op}: unexpected binding result')
 err,*values=result
 if int(err)!=0:raise RuntimeError(f'{op} failed with CUDA error {int(err)}')
 if len(values)==1:return values[0]
 return tuple(values)

def graph_data_compat(graph, runtime_module=None, driver_module=None):
 """Graph topology/config without cudaGraphNodeGetToolsId (CUDA<13.1 compatible)."""
 if runtime_module is None:
  import cuda.bindings.runtime as runtime_module
 if driver_module is None:
  import cuda.bindings.driver as driver_module
 rt=runtime_module;drv=driver_module;raw=graph.raw_cuda_graph()
 _unused,num=_cuda_checked(rt.cudaGraphGetNodes(raw,numNodes=0),'cudaGraphGetNodes(count)')
 nodes,num=_cuda_checked(rt.cudaGraphGetNodes(raw,numNodes=int(num)),'cudaGraphGetNodes(data)')
 type_names={
  rt.cudaGraphNodeType.cudaGraphNodeTypeKernel:'kernel',rt.cudaGraphNodeType.cudaGraphNodeTypeMemcpy:'memcpy',
  rt.cudaGraphNodeType.cudaGraphNodeTypeMemset:'memset',rt.cudaGraphNodeType.cudaGraphNodeTypeHost:'host',
  rt.cudaGraphNodeType.cudaGraphNodeTypeGraph:'child_graph',rt.cudaGraphNodeType.cudaGraphNodeTypeEmpty:'empty',
  rt.cudaGraphNodeType.cudaGraphNodeTypeWaitEvent:'wait_event',rt.cudaGraphNodeType.cudaGraphNodeTypeEventRecord:'event_record',
  rt.cudaGraphNodeType.cudaGraphNodeTypeMemAlloc:'mem_alloc',rt.cudaGraphNodeType.cudaGraphNodeTypeMemFree:'mem_free'}
 handle_to_idx={};infos=[]
 for i in range(int(num)):
  node=nodes[i];handle_to_idx[int(node)]=i;ntype=_cuda_checked(rt.cudaGraphNodeGetType(node),'cudaGraphNodeGetType')
  info=dict(index=i,node_type=type_names.get(ntype,str(ntype)),kernel_name=None,grid_dim=None,block_dim=None,shared_mem_bytes=None,dependencies=[],dependents=[])
  if ntype==rt.cudaGraphNodeType.cudaGraphNodeTypeKernel:
   cu_node=drv.CUgraphNode(init_value=int(node));err,params=drv.cuGraphKernelNodeGetParams(cu_node)
   if int(err)!=int(drv.CUresult.CUDA_SUCCESS):raise RuntimeError(f'cuGraphKernelNodeGetParams failed {int(err)}')
   info['grid_dim']=[int(params.gridDimX),int(params.gridDimY),int(params.gridDimZ)]
   info['block_dim']=[int(params.blockDimX),int(params.blockDimY),int(params.blockDimZ)]
   info['shared_mem_bytes']=int(params.sharedMemBytes)
   if int(params.func):
    func=drv.CUfunction(init_value=int(params.func));err,name=drv.cuFuncGetName(func)
    if int(err)==int(drv.CUresult.CUDA_SUCCESS):info['kernel_name']=name.decode() if isinstance(name,bytes) else str(name)
    else:info['kernel_name']=f'func_handle:{int(params.func)}'
  infos.append(info)
 _a,_b,_c,num_edges=_cuda_checked(rt.cudaGraphGetEdges(raw,numEdges=0),'cudaGraphGetEdges(count)')
 if int(num_edges):
  frm,to,_edge_data,num_edges=_cuda_checked(rt.cudaGraphGetEdges(raw,numEdges=int(num_edges)),'cudaGraphGetEdges(data)')
  for i in range(int(num_edges)):
   src=handle_to_idx.get(int(frm[i]));dst=handle_to_idx.get(int(to[i]))
   if src is not None and dst is not None:infos[src]['dependents'].append(dst);infos[dst]['dependencies'].append(src)
 for info in infos:info['dependencies'].sort();info['dependents'].sort()
 graph_id=None;graph_id_available=False
 if hasattr(rt,'cudaGraphGetId'):
  try:
   graph_id=_cuda_checked(rt.cudaGraphGetId(raw),'cudaGraphGetId');graph_id_available=True
  except RuntimeError:
   # Optional diagnostic identity only; topology/launch signature does not use it.
   graph_id=None
 return dict(graph_id=None if graph_id is None else int(graph_id),graph_id_available=graph_id_available,tools_id_available=False,nodes=infos)

def normalized_graph(meta):
 fields=('node_type','kernel_name','grid_dim','block_dim','shared_mem_bytes','dependencies','dependents')
 return [{'index':n.get('index'),**{k:n.get(k) for k in fields}} for n in meta['nodes']]
def signature(meta):return hashlib.sha256(json.dumps(normalized_graph(meta),sort_keys=True).encode()).hexdigest()
def anomaly(row):
 gap=None if row['device_us'] is None else row['wall_us']-row['device_us']
 return dict(wall_minus_device_us=gap,host_span_separation=(gap is not None and gap>max(20.,.01*row['device_us'])),cause_proven=False)


def run(args):
 conf=json.loads(Path(__file__).with_name('manifest.json').read_text());parent=Path(conf['parent_campaign'])
 if (parent/'source/SOURCE_COMMIT.txt').read_text().strip()!=conf['parent_source_commit']:raise RuntimeError('frozen source mismatch')
 sys.path.insert(0,str(parent/'source'))
 import measure as M
 import torch,flashinfer
 if args.out.exists():raise FileExistsError('preserve prior diagnostic')
 if flashinfer.__version__!='0.7.0':raise RuntimeError('wrong version')
 binding=json.loads((Path(flashinfer.__file__).parent.parent/'RESOURCE_BINDING.json').read_text())
 for rel,h in binding['modified_hashes'].items():
  if digest_file(Path(flashinfer.__file__).parent.parent/rel)!=h:raise RuntimeError('overlay drift')
 args.out.mkdir(parents=True);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[:min(2,len(allowed))]))
 hardware=M.command(['nvidia-smi','--query-gpu=name,uuid,driver_version,power.limit','--format=csv'])
 env=dict(parent=conf['parent_campaign'],parent_source=conf['parent_source_commit'],diagnostic_source_sha256=digest_file(__file__),manifest_sha256=digest_file(Path(__file__).with_name('manifest.json')),torch=torch.__version__,cuda=torch.version.cuda,flashinfer=flashinfer.__version__,hardware=hardware,affinity=sorted(os.sched_getaffinity(0)),rep=args.rep,slurm_job=os.environ.get('SLURM_JOB_ID'),partition=os.environ.get('SLURM_JOB_PARTITION'),graph_keep_template=True,profiled=False,diagnostic_instrumentation=True,release_evidence=False)
 save(args.out/'environment.json',env);allrows=[];allquals=[];memory=[];manifest=M.load();start=time.monotonic()
 def one_case(spec):
  case=next(x for x in manifest['cases'] if x['id']==spec['case']);key=dict(case=case['id'],family='canary',dtype=spec['dtype'],layout=spec['layout'],split=spec['split'])
  if spec['layout']!='ragged':raise RuntimeError('frozen diagnostic supports only ragged')
  cell=M.digest(key);torch.manual_seed(int(cell[:8],16));qs=case['q'];ls=[q+c for q,c in zip(qs,case['cached'])];dtype=getattr(torch,spec['dtype'])
  q=torch.randn((sum(qs),32,128),device='cuda',dtype=dtype);k=torch.randn((sum(ls),8,128),device='cuda',dtype=dtype);v=torch.randn_like(k)
  ks=list(k.split(ls));vs=list(v.split(ls));qi=torch.tensor([0]+list(itertools.accumulate(qs)),device='cuda',dtype=torch.int32)
  ref_path=parent/'refs'/env['partition']/'s0'/(cell+'.pt');ref=torch.load(ref_path,map_location='cpu',weights_only=True)
  if M.digest([M.thash(q),M.thash(k),M.thash(v)])!=ref['inputs']:raise RuntimeError('historical inputs differ')
  def runtime(arm):
   rt=M.Runtime(flashinfer,torch,'ragged',qi,ks,vs,k,v,ls,dtype,spec['split'],arm,True);rt.attach(q);return rt
  def capture(rt,count,arm):
   rt.arm=arm;rt.wrapper._sgi_resource_policy={'off':0,'guarded':2}[arm];info,_=rt.plan()
   side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(side):
    for _ in range(3):rt.call(q)
   torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
   g=torch.cuda.CUDAGraph(keep_graph=True)
   with torch.cuda.graph(g):
    for _ in range(count):rt.call(q)
   g.instantiate();rt.out.fill_(float('nan'));rt.lse.fill_(float('nan'));g.replay();torch.cuda.synchronize()
   if not torch.equal(rt.out.cpu(),ref['out']) or not torch.equal(rt.lse.cpu(),ref['lse']):raise RuntimeError('graph/reference parity')
   meta=graph_data_compat(g);return {'graph':g,'plan':info,'metadata':meta,'signature':signature(meta),'pointers':rt.pointers(q,k,v),'runtime':rt}
  independent={arm:runtime(arm) for arm in ('off','guarded')};shared=runtime('off')
  scenarios={name:{} for name in conf['scenarios']}
  for count in (1,16):
   scenarios['independent'][count]={arm:capture(rt,count,arm) for arm,rt in independent.items()}
   scenarios['same_storage'][count]={'off':capture(shared,count,'off'),'guarded':capture(shared,count,'guarded')}
   a=scenarios['same_storage'][count]['off'];scenarios['same_graph_null'][count]={'off':a,'guarded':a}
  qual=dict(key,reference_sha256=digest_file(ref_path),inputs_sha256=ref['inputs'],graphs={})
  for name,counts in scenarios.items():
   qual['graphs'][name]={}
   for count,arms in counts.items():
    aa=arms['off'];bb=arms['guarded']
    if aa['plan'][:-1]!=bb['plan'][:-1]:raise RuntimeError('plan core mismatch')
    selected=bool(bb['plan'][-1]) if name!='same_graph_null' else False
    if not selected and aa['signature']!=bb['signature']:raise RuntimeError('fallback graph topology/config differs')
    if name=='same_storage' and aa['pointers']!=bb['pointers']:raise RuntimeError('shared storage not equal')
    qual['graphs'][name][str(count)]={arm:{k:val for k,val in a.items() if k not in ('graph','runtime')} for arm,a in arms.items()}
  allquals.append(qual)
  pool=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
  pool[0].record();pool[1].record();pool[1].synchronize()
  def measure(entry,count,timer,inject=False):
   graph=entry['graph'];replays=16 if count==1 else 1;graph.replay();torch.cuda.synchronize()
   if timer=='legacy_events':events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
   elif timer=='pooled_events':events=pool
   elif timer=='wall_only':events=None
   else:raise ValueError(timer)
   stamp=time.time_ns();r0=resource.getrusage(resource.RUSAGE_THREAD);cpu0=time.thread_time_ns();restore=gc.isenabled();gc.disable()
   try:
    t0=time.perf_counter_ns()
    if events:events[0].record()
    t1=time.perf_counter_ns()
    for _ in range(replays):graph.replay()
    if events:events[1].record()
    t2=time.perf_counter_ns()
    if events:events[1].synchronize()
    else:torch.cuda.current_stream().synchronize()
    t3=time.perf_counter_ns()
    if inject:time.sleep(.002)
    t4=time.perf_counter_ns()
   finally:
    if restore:gc.enable()
   cpu1=time.thread_time_ns();r1=resource.getrusage(resource.RUSAGE_THREAD)
   device=None if events is None else events[0].elapsed_time(events[1])*1000/16
   result=dict(timestamp_unix_ns=stamp,wall_us=(t4-t0)/1000/16,device_us=device,record_us=(t1-t0)/1000,enqueue_us=(t2-t1)/1000,wait_us=(t3-t2)/1000,post_wait_us=(t4-t3)/1000,window_wall_us=(t4-t0)/1000,thread_cpu_us=(cpu1-cpu0)/1000,voluntary_switches=r1.ru_nvcsw-r0.ru_nvcsw,involuntary_switches=r1.ru_nivcsw-r0.ru_nivcsw,attention_invocations=16,positive_control=inject)
   result.update(anomaly(result));return result
  rng=random.Random(62173+args.rep*1009+int(cell[:8],16))
  for block in range(conf['blocks']):
   combinations=list(itertools.product(conf['scenarios'],conf['timers'],(1,16)));rng.shuffle(combinations)
   for scenario,timer,count in combinations:
    seq=('off','guarded','guarded','off') if block%2==0 else ('guarded','off','off','guarded')
    for pos,arm in enumerate(seq):
     entry=scenarios[scenario][count][arm]
     allrows.append(dict(key,rep=args.rep,block=block,position=pos,arm=arm,scenario=scenario,timer=timer,execution=f'graph{count}_replay',selected=bool(entry['plan'][-1]),graph_signature=entry['signature'],**measure(entry,count,timer)))
  injected=measure(scenarios['same_graph_null'][1]['off'],1,'pooled_events',inject=True)
  allrows.append(dict(key,rep=args.rep,block=-1,position=0,arm='injected_host_delay',scenario='positive_control',timer='pooled_events',execution='graph1_replay',selected=False,**injected))
  if not injected['host_span_separation']:raise RuntimeError('host-delay positive control not detected')
  print('CASE_DONE',args.rep,key,len(allrows),flush=True)
 try:
  for spec in conf['cases']:
   one_case(spec);gc.collect();torch.cuda.empty_cache();memory.append(dict(spec,allocated=int(torch.cuda.memory_allocated()),reserved=int(torch.cuda.memory_reserved())))
   save(args.out/'progress.json',dict(rows=len(allrows),cases=len(allquals),elapsed_s=time.monotonic()-start))
  expected=len(conf['cases'])*len(conf['scenarios'])*len(conf['timers'])*len(conf['executions'])*conf['blocks']*4
  if sum(not x['positive_control'] for x in allrows)!=expected:raise RuntimeError('incomplete diagnostic')
  save(args.out/'rows.json',allrows);save(args.out/'qualification.json',allquals);save(args.out/'memory.json',memory)
  save(args.out/'complete.json',dict(rows=len(allrows),noninjected_rows=expected,qualified_cases=len(allquals),files={p.name:digest_file(p) for p in args.out.glob('*.json')},release_evidence=False,default_promotion=False))
 except BaseException as e:
  save(args.out/'partial_rows.json',allrows);save(args.out/'partial_qualification.json',allquals);save(args.out/'failure.json',dict(error=type(e).__name__,message=str(e)));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--rep',type=int,choices=range(3),required=True);run(p.parse_args())
