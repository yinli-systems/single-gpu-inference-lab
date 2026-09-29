import unittest
from robustness import cases
class NumericalFixtures(unittest.TestCase):
 def test_count(self):self.assertEqual(len(cases()),72)
 def test_unique(self):self.assertEqual(len({x['id'] for x in cases()}),72)
 def test_heads(self):
  for c in cases():self.assertEqual(c['hq']%c['hkv'],0)
 def test_lengths(self):
  for c in cases():self.assertTrue(all(k>=q>0 for q,k in zip(c['q'],c['kv'])))
 def test_mask_both(self):self.assertEqual({c['causal'] for c in cases()},{True,False})
if __name__=='__main__':unittest.main()
