import copy,math,unittest
import numpy as np
from release_gate import evaluate_release_gate,subset_gate_inputs

def cell(selected,guarded=1.05,off=1.0,resolved=True,calls=16):
 return {'calls':calls,'metric':'run_device_us','selected':selected,'comparisons':{'guarded':{'ratio':guarded,'controls_resolve':resolved},'off':{'ratio':off,'controls_resolve':True}}}
def family(selected,guarded=1.05,off=1.0,resolved=True):return [cell(selected,guarded,off,resolved,c) for c in (0,1,16)]
def draws(cells):
 out={}
 for i,c in enumerate(cells):
  for mode in ('guarded','off'):out[(i,mode)]=np.full(2000,math.log(c['comparisons'][mode]['ratio']))
 return out
NUMERICS={m:{'qualifications':10,'exact_full_outputs':10,'max_abs_vs_pristine':0.0} for m in ('pristine','off','cap','guarded')}
class ReleaseGateTests(unittest.TestCase):
 def test_passes_strong_selected_gain_and_identity_fallback(self):
  cells=family(True,1.08)+family(False,1.0);r=evaluate_release_gate(cells,draws(cells),NUMERICS);self.assertTrue(r['pass']);self.assertEqual(r['selected']['count'],1);self.assertEqual(r['all_execution_selected']['count'],3)
 def test_selected_regression_fails(self):
  cells=family(True,.985)+family(False,1.0);r=evaluate_release_gate(cells,draws(cells),NUMERICS);self.assertFalse(r['pass']);self.assertFalse(r['requirements']['graph16_selected_worst_point_at_least_0_99'])
 def test_eager_only_regression_fails(self):
  cells=family(True,1.08)+family(False,1.0);cells[0]['comparisons']['guarded']['ratio']=.985;r=evaluate_release_gate(cells,draws(cells),NUMERICS);self.assertFalse(r['requirements']['all_execution_selected_worst_point_at_least_0_99'])
 def test_unresolved_selected_cell_fails(self):
  cells=family(True,1.08,resolved=False)+family(False,1.0);self.assertFalse(evaluate_release_gate(cells,draws(cells),NUMERICS)['pass'])
 def test_no_selected_cell_fails(self):
  cells=family(False,1.0)+family(False,1.0);r=evaluate_release_gate(cells,draws(cells),NUMERICS);self.assertFalse(r['pass']);self.assertFalse(r['requirements']['selected_nonempty'])
 def test_numerical_mismatch_fails(self):
  cells=family(True,1.08)+family(False,1.0);n=copy.deepcopy(NUMERICS);n['guarded']['exact_full_outputs']=9;self.assertFalse(evaluate_release_gate(cells,draws(cells),n)['pass'])
 def test_policy_and_overlay_worst_cases_are_gated(self):
  cells=family(True,1.08)+family(False,.98);r=evaluate_release_gate(cells,draws(cells),NUMERICS);self.assertFalse(r['requirements']['policy_worst_point_at_least_0_99'])
  cells=family(True,1.08,off=.98)+family(False,1.0,off=1.0);r=evaluate_release_gate(cells,draws(cells),NUMERICS);self.assertFalse(r['requirements']['off_overlay_worst_point_at_least_0_99'])
 def test_sensitivity_subset_does_not_replace_primary(self):
  duplicate=family(True,.98);fresh=family(True,1.08)+family(False,1.0)
  for c in duplicate:c['case']='duplicate-boundary'
  for c in fresh:c['case']='strictly-fresh' if c['selected'] else 'fresh-fallback'
  cells=duplicate+fresh;cache=draws(cells)
  primary=evaluate_release_gate(cells,cache,NUMERICS);self.assertFalse(primary['pass'])
  subset,subset_draws=subset_gate_inputs(cells,cache,{'duplicate-boundary'})
  sensitivity=evaluate_release_gate(subset,subset_draws,NUMERICS);self.assertTrue(sensitivity['pass'])
  self.assertEqual(len(cells),9);self.assertEqual(len(subset),6)
if __name__=='__main__':unittest.main()
