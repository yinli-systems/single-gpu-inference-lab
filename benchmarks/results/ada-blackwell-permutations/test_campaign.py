import unittest
from campaign import CONFIGS,states,geometry,schema_check
class Contracts(unittest.TestCase):
 def test_marginals(self):
  for n in CONFIGS:
   gs=[geometry(n,l,p) for l,p in states(n)]
   self.assertTrue(all(g['marginal_key']==gs[0]['marginal_key'] for g in gs))
 def test_same_opposite_extrema(self):
  for n in CONFIGS:
   gs=[geometry(n,l,p) for l,p in states(n)]
   self.assertEqual(gs[0]['C'],max(g['C'] for g in gs))
   self.assertEqual(gs[1]['C'],min(g['C'] for g in gs))
 def test_equal_work(self):
  for n in ('n4','n8','n16'):
   gs={l:geometry(n,l,p) for l,p in states(n)}
   self.assertNotEqual(gs['eqC-a']['query'],gs['eqC-b']['query'])
   self.assertEqual(gs['eqC-a']['W'],gs['eqC-b']['W'])
 def test_order_null_pairs(self):
  for n in CONFIGS:
   gs={l:geometry(n,l,p) for l,p in states(n)}
   pairs=lambda g:sorted(zip(g['query'],g['cached']))
   self.assertEqual(pairs(gs['same']),pairs(gs['order-null']))
 def test_unique_labels(self):
  for n in CONFIGS:
   st=states(n);self.assertEqual(len(st),len(set(l for l,p in st)))
 def test_expected_rows(self):
  self.assertEqual(sum(len(states(n)) for n in CONFIGS)*2*12*2*2,3456)
 def test_permutation_invalid(self):
  with self.assertRaises(ValueError): geometry('n2','same',[1,1])
 def test_schema_valid(self):
  schema_check(dict(geometry('n2','same',CONFIGS['n2'][1]),arm='auto',mode='eager',block=0,device_us=1.,wall_us=2.,plan_us=3.))
 def test_nonfinite_rejected(self):
  for val in (float('nan'),float('inf'),0,-1):
   with self.assertRaises(ValueError): schema_check(dict(geometry('n2','same',CONFIGS['n2'][1]),arm='auto',mode='eager',block=0,device_us=val,wall_us=2.,plan_us=3.))
 def test_boolean_block(self):
  with self.assertRaises(ValueError): schema_check(dict(geometry('n2','same',CONFIGS['n2'][1]),arm='auto',mode='eager',block=True,device_us=1.,wall_us=2.,plan_us=3.))
if __name__=='__main__': unittest.main()
