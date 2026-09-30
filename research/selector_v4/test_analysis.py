import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from analyze_evaluation import summarize
from aggregate_dual_gpu import run as dual_run

class SummaryTests(unittest.TestCase):
 def cell(self,tactic='cap',ratio=1.2,regret=1.0,control=True):return {'tactic':tactic,'chosen_ratio':ratio,'regret':regret,'control_resolves':control}
 def test_safe_cap_passes(self):
  cells=[self.cell(),self.cell(ratio=1.1)];draws=[[1.2]*10000,[1.1]*10000];result=summarize(cells,draws,draws,require_cap=True)
  self.assertTrue(result['pass']);self.assertEqual(result['chosen_worst'],1.1)
 def test_bad_tail_or_regret_blocks(self):
  cells=[self.cell(ratio=.989),self.cell(regret=1.03)];draws=[[.989]*10000,[1.2]*10000];result=summarize(cells,draws,[draws[1]],require_cap=True)
  self.assertFalse(result['pass']);self.assertFalse(result['requirements']['chosen_worst_at_least_0_99']);self.assertFalse(result['requirements']['max_regret_at_most_1_02'])
 def test_no_cap_allowed_only_for_nonfinal_shard(self):
  cells=[self.cell(tactic='native',ratio=1.0)];draws=[[1.0]*10000]
  self.assertTrue(summarize(cells,draws,[],require_cap=False)['pass'])
  self.assertFalse(summarize(cells,draws,[],require_cap=True)['pass'])
 def test_unresolved_control_blocks(self):
  cells=[self.cell(control=False)];draws=[[1.2]*10000]
  self.assertFalse(summarize(cells,draws,draws,require_cap=True)['pass'])

class DualGateTests(unittest.TestCase):
 def test_requires_both_gpu_pass(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);paths=[]
   for gpu in ('gpu_4090','gpu_5090'):
    p=root/(gpu+'.json');p.write_text(json.dumps({'gpu':gpu,'stage':'release','pass':True,'chosen_worst':1.0,'chosen_joint_lcb':1.0,'max_regret':1.0,'p95_regret':1.0,'cap_geomean':1.2,'cap_ci95':[1.1,1.3],'provenance':{'case_hash':'c','release_hash':'r'}}));paths.append(p)
   out=root/'out.json';dual_run(SimpleNamespace(gpu_4090=paths[0],gpu_5090=paths[1],out=out));self.assertTrue(json.loads(out.read_text())['performance_release_pass'])
   bad=json.loads(paths[1].read_text());bad['pass']=False;paths[1].write_text(json.dumps(bad))
   with self.assertRaisesRegex(RuntimeError,'HOLD'):dual_run(SimpleNamespace(gpu_4090=paths[0],gpu_5090=paths[1],out=out))

if __name__=='__main__':unittest.main()
