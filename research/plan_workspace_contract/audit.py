from pathlib import Path
import argparse,hashlib,json

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(a):
 pre=(a.flashinfer/'prefill.py').read_text();backend=(a.sglang/'srt/layers/attention/flashinfer_backend.py').read_text();sched=(a.flashinfer/'data/include/flashinfer/attention/scheduler.cuh').read_text()
 facts={
  'pinned_workspace_allocated':'pin_memory=True' in pre,
  'async_plan_copy':'cudaMemcpyAsync(int_buffer, page_locked_int_buffer' in sched,
  'public_plan_has_blocking_host_read':'qo_indptr_host = qo_indptr.to("cpu")' in pre and 'paged_kv_indptr_host = paged_kv_indptr.to("cpu")' in pre,
  'fast_plan_direct_module_call':'def fast_prefill_plan(' in backend and 'self._cached_module.plan(*args)' in backend,
  'fast_plan_draft_scope':'forward_mode.is_draft_extend_v2()' in backend and 'w.begin_forward = partial(fast_prefill_plan, w)' in backend,
  'fast_plan_dflash_scope':'SpecInputType.DFLASH_VERIFY' in backend,
  'ordinary_extend_installs_fast_plan':False,
 }
 # The ordinary forward_mode.is_extend branch must precede no assignment of fast_prefill_plan in its bounded branch.
 i=backend.index('elif forward_mode.is_extend():');j=backend.index('if in_capture and forward_mode.is_decode_or_idle()',i);facts['ordinary_extend_installs_fast_plan']='fast_prefill_plan' in backend[i:j]
 if not all(v for k,v in facts.items() if k!='ordinary_extend_installs_fast_plan') or facts['ordinary_extend_installs_fast_plan']:raise ValueError(facts)
 out={'facts':facts,'source_sha256':{'prefill.py':sha(a.flashinfer/'prefill.py'),'flashinfer_backend.py':sha(a.sglang/'srt/layers/attention/flashinfer_backend.py'),'scheduler.cuh':sha(a.flashinfer/'data/include/flashinfer/attention/scheduler.cuh')},'historical_path_cause_claim':False,'fast_path_followup_required':True}
 a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--flashinfer',type=Path,required=True);p.add_argument('--sglang',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
