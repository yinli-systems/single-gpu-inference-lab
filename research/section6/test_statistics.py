import math,unittest
import numpy as np
from analyze import estimate,weights,require
class StatisticalContractTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.w=weights()
 def test_weights(self):
  self.assertEqual(self.w.shape,(10000,36));np.testing.assert_allclose(self.w.sum(axis=1),1)
  self.assertTrue((self.w>=0).all())
 def test_exact_noop(self):
  s=estimate([[0.]*12 for _ in range(3)],self.w,.9)
  self.assertEqual(s['ratio'],1.);self.assertEqual(s['CI'],[1.,1.])
 def test_constant_ratio(self):
  s=estimate([[math.log(1.25)]*12 for _ in range(3)],self.w)
  self.assertAlmostEqual(s['ratio'],1.25)
  for x in s['CI']:self.assertAlmostEqual(x,1.25)
 def test_reject_nonfinite(self):
  for v in (math.nan,math.inf,-math.inf):
   with self.assertRaises(ValueError):estimate([[v]*12 for _ in range(3)],self.w)
 def test_reject_incomplete_repeats(self):
  for x in ([[0.]*12]*2,[[0.]*11]*3,[[0.]*13]*3):
   with self.assertRaises(ValueError):estimate(x,self.w)
 def test_repeat_cluster_not_iid(self):
  x=[[-.05]*12,[0.]*12,[.05]*12];s=estimate(x,self.w)
  self.assertLess(s['CI'][0],.99);self.assertGreater(s['CI'][1],1.01)
if __name__=='__main__':unittest.main()
