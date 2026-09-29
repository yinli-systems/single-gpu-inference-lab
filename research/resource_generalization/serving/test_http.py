import unittest
from http_bench import update,workloads,quantile
class HttpContracts(unittest.TestCase):
 def test_delta_stream(self):self.assertEqual(update([1,2],2,[3],3),([1,2,3],3,1))
 def test_cumulative_stream(self):self.assertEqual(update([1,2],2,[1,2,3],3),([1,2,3],3,1))
 def test_final_empty(self):self.assertEqual(update([1,2],2,[],2),([1,2],2,0))
 def test_rewind_rejected(self):
  with self.assertRaises(ValueError):update([1,2],2,[1],1)
 def test_prefix_rejected(self):
  with self.assertRaises(ValueError):update([1,2],2,[1,9,3],3)
 def test_missing_output_rejected(self):
  with self.assertRaises(ValueError):update([],0,[],1)
 def test_unique_frozen_workloads(self):
  a=workloads();self.assertEqual(a,workloads());self.assertEqual(sum(len(w['cells']) for w in a),48)
  ids=[x['id'] for w in a for x in w['cells']];self.assertEqual(len(ids),len(set(ids)))
 def test_quantiles(self):self.assertEqual(quantile([1,2,3],.5),2)
if __name__=='__main__':unittest.main()
