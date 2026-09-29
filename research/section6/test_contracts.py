import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import measure
from order_guard import Geometry,propose
from metadata_adapter import expected_descriptors,validate_permutation
class Contracts(unittest.TestCase):
 def test_matrix(self):
  self.assertEqual(len(measure.CASES),6)
  self.assertEqual(2*len(measure.CASES)*2*len(measure.POLICIES)*len(measure.REGIMES)*measure.BLOCKS*4,23040)
 def test_all_case_orders(self):
  for _,q,k in measure.CASES:
   ls=[a+b for a,b in zip(q,k)]
   for split in (False,True):
    g=Geometry(tuple(q),tuple(ls),4,128,128 if split else -1,split)
    desc=g.descriptors()
    for pol in ('identity','locality_packet8','causal_heavy'):
     order=propose(g,desc,pol);validate_permutation(list(order),len(desc))
 def test_invalid_permutation(self):
  for x in ([0,0],[1],[0,1,3]):
   with self.assertRaises(ValueError):validate_permutation(x,2)
 def test_collision(self):
  a=measure.CASES[-2];b=measure.CASES[-1]
  self.assertEqual(sorted(a[1]),sorted(b[1]));self.assertEqual(sorted(a[2]),sorted(b[2]))
  self.assertEqual(sorted(x+y for x,y in zip(a[1],a[2])), sorted(x+y for x,y in zip(b[1],b[2])))
if __name__=='__main__':unittest.main()
