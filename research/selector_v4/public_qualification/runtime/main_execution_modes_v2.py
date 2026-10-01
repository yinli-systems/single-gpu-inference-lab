"""Source-bound functional/calibration check of all prepared-runner envelopes.

Only the already-exposed representative geometry is used. This is not HTTP,
metadata-update or independent fresh-release qualification.
"""
import argparse,hashlib,json,os
from pathlib import Path
import torch
from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
from flashinfer import autotune_v2, MeasurementPolicy
from flashinfer.autotuner import AutoTuner
from flashinfer.autotune_cache import autotune_v2_reload
from flashinfer.jit.core import logger
logger.setLevel("DEBUG")
from flashinfer.prefill import make_prefill_resource_runner

def digest_tensor(t):return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def main(out):
 if out.exists():raise FileExistsError('preserve mode results')
 out.mkdir(parents=True);torch.manual_seed(42);torch.set_num_threads(1)
 qs=[3,39,107,175,243,311,411];ks=[q+c for q,c in zip(qs,[22528,16512,11840,8256,2624,672,80])]
 def ip(lengths):return torch.tensor([0,*torch.tensor(lengths).cumsum(0).tolist()],dtype=torch.int32,device='cuda')
 q=torch.randn(sum(qs),32,128,dtype=torch.bfloat16,device='cuda');k=torch.randn(sum(ks),8,128,dtype=q.dtype,device=q.device);v=torch.randn_like(k);inputs=[q,k,v]
 w=BatchPrefillWithRaggedKVCacheWrapper(torch.empty(128<<20,dtype=torch.uint8,device='cuda'),backend='fa2');w.plan(ip(qs),ip(ks),32,8,128,causal=True,q_data_type=q.dtype,disable_split_kv=True)
 native=w.run(q,k,v).clone();results={}
 for mode,count in [('eager_run',1),('graph_replay',1),('graph_replay',16)]:
  name=mode+'-'+str(count)
  runner=make_prefill_resource_runner(w,inputs,qo_lengths=qs,kv_lengths=ks,execution_mode=mode,graph_replays=count)
  receipt=runner.calibrate(inputs)
  assert runner._proxy is not None, getattr(runner,'_failure_reason','resource absent')
  assert runner._current(inputs),'prepared plan invalidated by ordinary execution'
  policy=MeasurementPolicy(execution_mode="eager" if mode=="eager_run" else "cuda_graph")
  with autotune_v2(mode="tune",measurement_policy=policy,cache_root=out/"managed-cache"):
   result=runner.run(inputs)
   chosen_runner,chosen_tactic=AutoTuner.get().choose_one("experimental_prefill_resource",[runner],runner.tuning_config,inputs)
  assert chosen_runner is runner and chosen_tactic in (-1,0,1), "Managed profiling fell back without a valid tactic"
  assert AutoTuner.get().stats.tuned_op_successful_configs.get("experimental_prefill_resource",0)>0,"No successfully profiled managed winner"
  assert not AutoTuner.get().stats.failed_tactics.get("experimental_prefill_resource::ResourceCapRunner"),"Offered tactic failed profiling"
  entries=[json.loads(p.read_text()) for p in (out/"managed-cache").glob("v2/*/entries/*.json")]
  matches=[e for e in entries if e.get("runner")=="ResourceCapRunner" and e.get("tactic")==chosen_tactic and runner.identity in e.get("key","") and receipt["checksum"] in e.get("key","")]
  assert len(matches)==1,"No exact source/certificate-bound persisted winner"
  torch.testing.assert_close(result,native,rtol=0,atol=0)
  autotune_v2_reload()
  with autotune_v2(mode="replay",measurement_policy=policy,cache_root=out/"managed-cache"):
   replayed=runner.run(inputs)
   _,reloaded_tactic=AutoTuner.get().choose_one("experimental_prefill_resource",[runner],runner.tuning_config,inputs)
   assert reloaded_tactic==chosen_tactic,"Disk-reloaded managed tactic changed"
  torch.testing.assert_close(replayed,native,rtol=0,atol=0)
  assert list((out/"managed-cache").glob("v2/*/entries/*.json")),"Managed entries were not persisted"
  if mode=='graph_replay':
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):captured=runner.run(inputs)
   for _ in range(3):graph.replay()
   torch.cuda.synchronize();torch.testing.assert_close(captured,native,rtol=0,atol=0)
  entry={'identity':runner.identity,'receipt':receipt,'managed_measurement_policy':policy.execution_mode,'managed_tactic':chosen_tactic,'persisted_winner':matches[0],'managed_output_sha256':digest_tensor(result),'native_output_sha256':digest_tensor(native),'resource_module_prepared':True,'managed_exact':True,'disk_reload_replay_exact':True,'reloaded_tactic':reloaded_tactic,'captured_managed_exact':mode=='graph_replay','scope':'preplanned eager wall-time or pure replay only; no metadata updates/model/HTTP','default_promotion':False,'serving_promotion':False}
  (out/(name+'.json')).write_text(json.dumps(entry,indent=2)+'\n');results[name]={k:v for k,v in entry.items() if k!='receipt'}|{'tactic':receipt['tactic'],'geomean':receipt.get('geomean'),'checks':receipt.get('checks')}
  print(name,json.dumps(results[name]),flush=True)
 assert len({x['identity'] for x in results.values()})==3
 p=torch.cuda.get_device_properties(0);summary={'pass':True,'three_distinct_execution_identities':True,'hardware':{'name':p.name,'uuid':str(p.uuid)},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'inputs_sha256':[digest_tensor(t) for t in inputs],'modes':results,'fresh_holdout_cases_consumed':0,'full_serving_qualified':False,'historical_token_divergence_resolved':False};(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);x=a.parse_args();main(x.out)
