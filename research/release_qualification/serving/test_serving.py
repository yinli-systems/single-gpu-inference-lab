import unittest
from workloads import prefix_tokens,work_specs,update
class ServingTests(unittest.TestCase):
 def test_prefix(self):self.assertEqual(len(prefix_tokens()),8448)
 def test_workloads(self):
  w=work_specs(prefix_tokens(),0);self.assertEqual(set(w),{'guarded_prefix','balanced_prefix','short_prefill','decode','mixed'})
  self.assertEqual(sum(len(x['input_ids'])-8448 for x in w['guarded_prefix']['cells']),1008)
  self.assertEqual(len({len(x['input_ids']) for x in w['balanced_prefix']['cells']}),1)
  self.assertTrue(all(x['expect_cached']==8192 for x in w['guarded_prefix']['cells']))
 def test_delta(self):self.assertEqual(update([1,2],2,[3],3),([1,2,3],3,1))
 def test_cumulative(self):self.assertEqual(update([1,2],2,[1,2,3],3),([1,2,3],3,1))
if __name__=='__main__':unittest.main()
