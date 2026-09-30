import copy,math,sys,unittest
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from release_gate_v322 import evaluate
from authorize_release_v322 import evaluate_authorization

class GateTests(unittest.TestCase):
 def cells(self,selected_paired=1.02,unselected_paired=1.0,absolute_unselected=.97,off=1.0):
  cells=[];draws={}
  coords=[(True,'eager_full_call'),(True,'graph1_replay'),(True,'graph16_replay'),(False,'eager_full_call'),(False,'graph1_replay'),(False,'graph16_replay')]
  for i,(selected,execution) in enumerate(coords):
   absolute=1.08 if selected else absolute_unselected;paired=selected_paired if selected else unselected_paired
   comparisons={
    'guarded':{'ratio':absolute,'controls_resolve':True},
    'off':{'ratio':off,'controls_resolve':True,'disabled_overlay_resolves_one_percent':True},
    'paired_guarded':{'ratio':paired,'controls_resolve':True}}
   cells.append({'selected':selected,'execution_mode':execution,'comparisons':comparisons})
   for mode,c in comparisons.items():draws[i,mode]=np.full(10000,math.log(c['ratio']))
  numerics={a:{'qualifications':10,'exact_full_outputs':10,'max_abs_vs_pristine':0.0} for a in ('pristine','off','cap','guarded')}
  return cells,draws,numerics
 def test_absolute_whole_policy_failure_is_retained_but_nonblocking(self):
  gate=evaluate(*self.cells());self.assertTrue(gate['pass']);self.assertLess(gate['absolute_policy_audit']['worst_point_ratio'],.99);self.assertFalse(gate['absolute_policy_audit_is_blocking'])
 def test_paired_unselected_regression_blocks(self):
  gate=evaluate(*self.cells(unselected_paired=.989));self.assertFalse(gate['pass']);self.assertFalse(gate['requirements']['paired_unselected_worst_point_at_least_0_99'])
 def test_paired_joint_tail_blocks(self):
  cells,draws,numerics=self.cells();draws[0,'paired_guarded'][:1000]=math.log(.97);self.assertFalse(evaluate(cells,draws,numerics)['pass'])
 def test_disabled_overlay_blocks(self):
  cells,draws,numerics=self.cells(off=.989);self.assertFalse(evaluate(cells,draws,numerics)['pass'])

class AuthorizationTests(unittest.TestCase):
 def fixture(self):
  identity={'case_hash':'c','source_archive_sha256':'s','official_overlay_sha256':'o'};canaries={}
  for gpu in ('gpu_4090','gpu_5090'):
   requirements={'numerical_exact':True,'selected_nonempty':True,'selected_point_worst_at_least_0_98':True,'policy_point_worst_at_least_0_98':False,'disabled_overlay_point_worst_at_least_0_98':True,'paired_policy_point_worst_at_least_0_98':False}
   canaries[gpu]={'measurement_contract_revision':'3.2.1','stage':'canary','gpu':gpu,'canary_gate':{'pass':False,'requirements':requirements},'hardware':{'0':{'uuid':gpu,'driver':'580.82.07'}},**identity}
  groups=[]
  for scenario,timer in (('independent','pooled_events'),('independent','wall_only'),('same_graph_null','pooled_events')):
   groups.append({'scenario':scenario,'timer':timer,'paired_wall_ratio':.999,'conditional_CI95':[.998,1.001]})
  diagnosis={'all_failed_canaries_retained':True,'release_gate_pass':False,'results':{gpu:{'noninjected_rows':8640,'positive_controls':15,'qualified_case_repeats':15,'fallback_graph_metadata_mismatches':0,'hardware':'driver 580.82.07','groups':copy.deepcopy(groups)} for gpu in canaries}}
  expected={'case_hash':'c','parent_source_archive_sha256':'s','official_overlay_sha256':'o'}
  return canaries,diagnosis,expected
 def test_valid_evidence_authorizes_only_release_qualification(self):
  result=evaluate_authorization(*self.fixture());self.assertTrue(result['release_qualification_authorized']);self.assertFalse(result['default_promotion']);self.assertFalse(result['historical_token_divergence_resolved']);self.assertTrue(result['old_canary_gate_remains_hold'])
 def test_bad_diagnostic_lower_bound_blocks(self):
  canaries,diagnosis,expected=self.fixture();diagnosis['results']['gpu_4090']['groups'][0]['conditional_CI95'][0]=.984
  with self.assertRaisesRegex(RuntimeError,'equivalence'):evaluate_authorization(canaries,diagnosis,expected)
 def test_rewritten_canary_pass_is_rejected(self):
  canaries,diagnosis,expected=self.fixture();canaries['gpu_4090']['canary_gate']['pass']=True
  with self.assertRaisesRegex(RuntimeError,'remain HOLD'):evaluate_authorization(canaries,diagnosis,expected)

if __name__=='__main__':unittest.main()
