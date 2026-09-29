import unittest
from manifest import build,digest
from prepare import transform
class Contracts(unittest.TestCase):
 def test_frozen_repeat(self): self.assertEqual(build(),build())
 def test_disjoint_families(self):
  m=build();a={len(x['q']) for x in m['cases'] if x['family']=='development'};b={len(x['q']) for x in m['cases'] if x['family']=='confirmatory'}
  self.assertFalse(a&b)
 def test_matrix_counts(self):
  m=build();self.assertEqual(len(m['cases']),43);self.assertEqual(sum(x['family']=='development' for x in m['cases']),16);self.assertEqual(sum(x['family']=='confirmatory' for x in m['cases']),24)
 def test_bounded_memory(self):
  for x in build()['cases']:
   self.assertGreater(min(x['q']),0);self.assertEqual(len(x['q']),len(x['cached']));self.assertLess(sum(x['q']),8192);self.assertLess(sum(x['q'])+sum(x['cached']),262144)
 def test_hash_scope(self):self.assertEqual(build()['case_hash'],digest(build()['cases']))
 def test_fail_closed_unknown_source(self):
  for mode in ['off','cap','wide']:
   with self.assertRaises(ValueError):transform('not actual upstream source',mode)
 def test_invalid_mode(self):
  with self.assertRaises(ValueError):transform('anything','unknown')
if __name__=='__main__':unittest.main()
