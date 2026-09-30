import ast,copy,hashlib,inspect,json,math,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import measurement_contract as mc
import evidence_contract as ec
from manifest import load,digest,tactic_identity,plan_guard
from measure import Runtime,verify_graph_outputs
from release_gate import evaluate
import native_policy as native
import prepare_guarded

class WindowTests(unittest.TestCase):
 def test_calibration_uses_fastest_pilot_and_power_of_two(self):
  self.assertEqual(mc.choose_eager_calls([16000.,20000.]),64)
  self.assertEqual(mc.choose_eager_calls([96000.,100000.]),16)
 def test_reject_invalid_calibration(self):
  for samples in ([],[0],[-1],[float('nan')],[float('inf')],[.001]):
   with self.subTest(samples=samples),self.assertRaises(ValueError):mc.choose_eager_calls(samples)
 def test_short_scored_window_is_error_not_retimed(self):
  with self.assertRaises(mc.EagerWindowTooShort) as e:mc.check_eager_window(11999.,16)
  self.assertEqual(e.exception.details['minimum_us'],12000.)
  mc.check_eager_window(12000.,16)
 def test_actual_treatment_sequences(self):
  self.assertEqual([a for a,r in mc.expected_sequence('paired','guarded',0)],['off','guarded','guarded','off'])
  self.assertEqual([a for a,r in mc.expected_sequence('paired','guarded',1)],['guarded','off','off','guarded'])
  with self.assertRaises(ValueError):mc.expected_sequence('pristine','guarded',0)

class RuntimeContractTests(unittest.TestCase):
 def test_hot_plan_does_not_read_ffi_vector(self):
  class Forbidden:
   @property
   def _plan_info(self):raise AssertionError('timed path read metadata')
  rt=Runtime.__new__(Runtime);rt.wrapper=Forbidden();seen=[];rt._plan=lambda:seen.append('plan')
  self.assertIsNone(rt.plan(inspect=False));self.assertEqual(seen,['plan'])
 def test_native_flags_are_checked_outside_timing(self):
  rt=Runtime.__new__(Runtime);rt.candidate=True;rt.arm='off';rt._plan=lambda:None
  rt.wrapper=SimpleNamespace(_plan_info=[0]*15+[1])
  with self.assertRaises(RuntimeError):rt.plan()
  rt.wrapper._plan_info=[0]*16;self.assertEqual(rt.plan(),([0]*16,False))
 def make_graphs(self,bad=None):
  class Tensor:
   def __init__(self):self.value=1.
   def fill_(self,x):self.value=x
   def cpu(self):return self.value
  rt=SimpleNamespace(out=Tensor(),lse=Tensor());rt.graphs={}
  for count in (1,16):
   def replay(c=count):
    if c!=bad:rt.out.value=1.;rt.lse.value=2.
   rt.graphs[count]=SimpleNamespace(replay=replay)
  torch=SimpleNamespace(equal=lambda x,y:x==y,cuda=SimpleNamespace(synchronize=lambda:None))
  return rt,torch
 def test_graph1_failure_cannot_be_overwritten_by_graph16(self):
  rt,t=self.make_graphs(1)
  with self.assertRaisesRegex(RuntimeError,'graph1'):verify_graph_outputs(rt,t,{'out':1.,'lse':2.},'off')
 def test_graph16_is_checked_separately(self):
  rt,t=self.make_graphs(16)
  with self.assertRaisesRegex(RuntimeError,'graph16'):verify_graph_outputs(rt,t,{'out':1.,'lse':2.},'off')
 def test_both_graphs_pass_only_after_writes(self):
  rt,t=self.make_graphs();self.assertEqual(verify_graph_outputs(rt,t,{'out':1.,'lse':2.},'off'),{'graph1_replay':True,'graph16_replay':True})

class NativePatchTests(unittest.TestCase):
 def test_native_default_off_and_original_signature_preserved(self):
  signature='Array<int64_t> BatchPrefillWithKVCachePlan(\n    '+', '.join('int64_t '+name for name in native.POLICY_ARGUMENTS)+') '
  src=signature+'{\n  initialize();\n  // Selector v3: frozen\n  plan_info.resource_cap = unchanged_rule;\n\n  return Array(plan_info.ToVector());\n}\n\nArray<int64_t> BatchPrefillWithKVCacheWorkspaceSize(\n'
  result=native.patch_native_entry(src)
  self.assertIn(signature,result);self.assertIn('return BatchPrefillWithKVCacheResourcePlan(0,',result)
  self.assertEqual(result.count('plan_info.resource_cap = unchanged_rule;'),1)
  self.assertIn('resource_policy >= 0 && resource_policy <= 2',result)
 def test_binding_keeps_old_and_adds_explicit_entry(self):
  src='Array<int64_t> BatchPrefillWithKVCachePlan(\n    int x);\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, BatchPrefillWithKVCachePlan);'
  result=native.patch_native_binding(src)
  self.assertIn('TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, BatchPrefillWithKVCachePlan);',result)
  self.assertIn('plan_resource, BatchPrefillWithKVCacheResourcePlan',result)
 def test_unreviewed_python_anchor_rejected(self):
  with self.assertRaises(ValueError):native.patch_python_dispatch('wrong source')

class EvidenceTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name);self.path=self.root/'runs'/'fixture';self.path.mkdir(parents=True);(self.root/'source').mkdir()
  self.case=copy.deepcopy(next(c for c in load()['cases'] if c['family']=='canary'));c=self.case
  hw='name, uuid, driver_version, power.limit [W]\nNVIDIA GeForce RTX 5090, GPU-fixture, 580.82.07, 575.00 W\n'
  self.env=dict(mode='paired',rep=0,stage='canary',gpu='NVIDIA GeForce RTX 5090',num_sm=170,torch='test',cuda='13.0',flashinfer='0.7.0',hardware={'out':hw},profiled=False,nsight_compute_excluded=True,measurement_contract_revision=mc.REVISION,cases=[c],case_hash=load()['case_hash'],source_archive_sha256='a'*64,official_overlay_sha256='b'*64,source={})
  for name in ('measure.py','manifest.py','measurement_contract.py','native_policy.py','prepare_guarded.py'):
   (self.root/'source'/name).write_text(name);self.env['source'][name]=hashlib.sha256(name.encode()).hexdigest()
  self.rows=[];self.quals=[]
  for dt in ('float16','bfloat16'):
   for layout in ('ragged','paged'):
    for split in ('auto','unsplit'):
     key=dict(case=c['id'],dtype=dt,layout=layout,split=split);core=[c['descriptor_count']]+[0]*14
     decision=plan_guard(c['q'],c['cached'],core[0],32,8,170,False);ordered=digest(list(zip(c['q'],c['cached'])))
     q=dict(key,measurement_contract_revision=mc.REVISION,eager_graph_exact=True,eager_calls=16,expected_selector=c['expected_selector'],ordered_pairs_sha256=ordered,guard_expected=decision,arms={},identity_payloads={},tactic_identities={})
     for arm in ('off','cap','guarded'):
      info=core+[int(arm=='cap' or (arm=='guarded' and decision))]
      q['arms'][arm]=dict(plan_info=info,pristine_exact=True,execution_checks=dict.fromkeys(ec.EXECUTIONS,True),out_sha256='1'*64,lse_sha256='2'*64);q['identity_payloads'][arm]={};q['tactic_identities'][arm]={}
      for ex in ec.EXECUTIONS:
       ei=dict(gpu_uuid='GPU-fixture',gpu_name=self.env['gpu'],num_sms=170,driver='580.82.07',torch='test',cuda='13.0',flashinfer='0.7.0',python='3.12',cudnn='test',nvcc='test',backend_sha256='c'*64,official_overlay_sha256='b'*64,source_archive_sha256='a'*64,selector_version=mc.REVISION,measurement_policy='fixture',profiling_mode='unprofiled')
       op=dict(execution_mode=ex,backend='fa2',causal=True,layout=layout,dtype=dt,requested_split=split,actual_split='unsplit',num_qo_heads=32,num_kv_heads=8,head_dim_qk=128,head_dim_vo=128,page_size=1 if layout=='ragged' else 16,q=c['q'],cached=c['cached'],ordered_pairs_sha256=ordered,plan_signature=info,window_repeats=1 if ex=='graph16_replay' else 16,candidate_binary_sha256='c'*64)
       payload=dict(environment=ei,operation=op);q['identity_payloads'][arm][ex]=payload;q['tactic_identities'][arm][ex]=tactic_identity(**payload)
     for block in range(2):
      for ex in ec.EXECUTIONS:
       for group in ('guarded','cap'):
        for pos,(arm,role) in enumerate(mc.expected_sequence('paired',group,block)):
         self.rows.append(dict(key,mode='paired',rep=0,block=block,execution_mode=ex,comparison_group=group,position=pos,arm=arm,role=role,wall_us=1000.,device_us=900.,window_elapsed_us=16000.,kernel_calls=16,tactic_identity=q['tactic_identities'][arm][ex]))
     self.quals.append(q)
  self.flush()
 def flush(self):
  memory=[{**{k:q[k] for k in ('case','dtype','layout','split')},'allocated_after_gc':0,'reserved_after_gc':0,'total_device_bytes':24000000000} for q in self.quals]
  files={'environment.json':self.env,'measurements.json':self.rows,'qualification.json':self.quals,'progress.json':{'rows':len(self.rows)},'memory.json':memory}
  for name,obj in files.items():(self.path/name).write_text(json.dumps(obj))
  complete=dict(complete=True,rows=len(self.rows),expected=len(self.rows),qualifications=len(self.quals),files={name:hashlib.sha256((self.path/name).read_bytes()).hexdigest() for name in files})
  (self.path/'complete.json').write_text(json.dumps(complete))
 def validate(self):return ec.validate_run(self.path,'paired',0,'canary',[self.case],2)
 def test_complete_synthetic_contract_passes(self):self.assertEqual(len(self.validate()[2]),384)
 def test_tampered_hash_rejected(self):
  (self.path/'measurements.json').write_text('[]')
  with self.assertRaisesRegex(ValueError,'digest mismatch'):self.validate()
 def test_old_revision_rejected(self):
  self.env['measurement_contract_revision']='3.2';self.flush()
  with self.assertRaisesRegex(ValueError,'revision'):self.validate()
 def test_mislabeled_abba_rejected(self):
  self.rows[0]['role']='B';self.flush()
  with self.assertRaisesRegex(ValueError,'ABBA'):self.validate()
 def test_missing_coordinate_rejected(self):
  self.rows.pop();self.flush()
  with self.assertRaisesRegex(ValueError,'incomplete timing'):self.validate()
 def test_short_window_rejected(self):
  self.rows[0]['wall_us']=100.;self.rows[0]['window_elapsed_us']=1600.;self.flush()
  with self.assertRaises(mc.EagerWindowTooShort):self.validate()
 def test_graph1_check_missing_rejected(self):
  del self.quals[0]['arms']['off']['execution_checks']['graph1_replay'];self.flush()
  with self.assertRaisesRegex(ValueError,'replay correctness'):self.validate()
 def test_gpu_identity_alias_rejected_even_with_digest_rebuilt(self):
  q=self.quals[0];payload=q['identity_payloads']['off']['eager_full_call'];payload['environment']['gpu_uuid']='other'
  q['tactic_identities']['off']['eager_full_call']=tactic_identity(**payload);self.flush()
  with self.assertRaisesRegex(ValueError,'GPU/driver'):self.validate()
 def test_source_drift_rejected(self):
  (self.root/'source'/'measure.py').write_text('changed')
  with self.assertRaisesRegex(ValueError,'measured source'):self.validate()
 def test_selector_flag_corruption_rejected(self):
  self.quals[0]['arms']['guarded']['plan_info'][-1]=1;self.flush()
  with self.assertRaisesRegex(ValueError,'selector rule'):self.validate()

class PairedGateTests(unittest.TestCase):
 def fixture(self,absolute=1.08,paired=1.08):
  cells=[];draws={}
  for i,ex in enumerate(ec.EXECUTIONS):
   comparisons={m:{'ratio':v,'controls_resolve':True} for m,v in [('guarded',absolute),('off',1.),('paired_guarded',paired)]}
   comparisons['off']['disabled_overlay_resolves_one_percent']=True
   cells.append(dict(selected=True,execution_mode=ex,comparisons=comparisons))
   for mode,c in comparisons.items():draws[i,mode]=np.full(10000,math.log(c['ratio']))
  n={arm:dict(qualifications=1,exact_full_outputs=1,max_abs_vs_pristine=0.) for arm in ('pristine','off','cap','guarded')}
  return cells,draws,n
 def test_good_absolute_does_not_override_bad_paired(self):
  result=evaluate(*self.fixture(paired=.98));self.assertFalse(result['pass']);self.assertFalse(result['requirements']['paired_selected_worst_point_at_least_0_99'])
 def test_joint_paired_lcb_blocks_unresolved_tail(self):
  c,d,n=self.fixture();d[0,'paired_guarded'][:1000]=math.log(.97)
  result=evaluate(c,d,n);self.assertFalse(result['requirements']['paired_selected_joint_worst_lcb_at_least_0_99'])
 def test_missing_disabled_equivalence_evidence_blocks(self):
  c,d,n=self.fixture();del c[0]['comparisons']['off']['disabled_overlay_resolves_one_percent']
  self.assertFalse(evaluate(c,d,n)['requirements']['disabled_overlay_all_controls_resolve'])
 def test_disabled_equivalence_failure_blocks(self):
  c,d,n=self.fixture();c[0]['comparisons']['off']['disabled_overlay_resolves_one_percent']=False
  self.assertFalse(evaluate(c,d,n)['pass'])
 def test_synthetic_pass_is_not_gpu_evidence(self):self.assertTrue(evaluate(*self.fixture())['pass'])

class ReleaseAuthorizationTests(unittest.TestCase):
 def test_missing_canary_blocks_release_without_submission(self):
  from authorize_release import authorize
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'receipts').mkdir();(root/'runs').mkdir()
   (root/'receipts'/'source-archive.sha256').write_text('a'*64)
   (root/'receipts'/'official-overlay.sha256').write_text('b'*64)
   with self.assertRaisesRegex(RuntimeError,'Missing unique completed'):authorize(root)

if __name__=='__main__':unittest.main()
