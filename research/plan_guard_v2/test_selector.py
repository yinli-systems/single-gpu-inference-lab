import json,unittest
from pathlib import Path
from manifest import load,plan_guard,workload_gate
class SelectorTests(unittest.TestCase):
 def test_manifest(self):
  x=load();self.assertEqual(len([c for c in x['cases'] if c['family']=='release']),24);self.assertEqual(len(set(c['id'] for c in x['cases'])),28)
 def test_arch_boundaries(self):
  q=[32,96,128,320,512];k=[0,128,512,8192,16384]
  self.assertFalse(plan_guard(q,k,32,8,128,False));self.assertTrue(plan_guard(q,k,33,8,128,False))
  self.assertFalse(plan_guard(q,k,42,8,170,False));self.assertTrue(plan_guard(q,k,43,8,170,False));self.assertFalse(plan_guard(q,k,64,8,128,True))
 def test_controls(self):
  self.assertFalse(workload_gate([224]*7,[0,128,512,2048,4096,8192,16384]));self.assertFalse(workload_gate([64,128,192],[0,4096,8191]))
if __name__=='__main__':unittest.main()
