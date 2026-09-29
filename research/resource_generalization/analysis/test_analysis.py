import unittest
import numpy as np
from analyze import weights,require
class Checks(unittest.TestCase):
 def test_weights_sum(self):
  for s in [0,1]:
   for mode in ['pristine','off','cap','wide']:
    w=weights(s,mode);self.assertEqual(w.shape,(10000,24));np.testing.assert_allclose(w.sum(axis=1),1);self.assertTrue((w>=0).all())
 def test_independent_arms(self):self.assertFalse(np.array_equal(weights(0,'pristine'),weights(0,'cap')))
 def test_repeat_exact(self):np.testing.assert_array_equal(weights(0,'cap'),weights(0,'cap'))
 def test_fail_closed(self):
  with self.assertRaises(ValueError):require(False,'missing')
if __name__=='__main__':unittest.main()
