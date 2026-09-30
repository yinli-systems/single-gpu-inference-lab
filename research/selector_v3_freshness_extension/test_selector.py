import unittest
from manifest import strict_opposite,descriptor_threshold,plan_guard,tactic_identity
class SelectorV3Tests(unittest.TestCase):
 def test_strict_opposite(self):
  self.assertTrue(strict_opposite([10,20,30,40,50],[100,80,60,40,20]))
 def test_same_rejected(self):
  self.assertFalse(strict_opposite([10,20,30,40,50],[20,40,60,80,100]))
 def test_one_concordant_rejected(self):
  self.assertFalse(strict_opposite([10,20,30,40,50],[100,60,80,40,20]))
 def test_ties_allowed_with_strict_support(self):
  self.assertTrue(strict_opposite([10,20,30,40,50],[100,80,80,40,20]))
 def test_too_small_rejected(self):
  self.assertFalse(strict_opposite([10,20,30,40],[100,80,40,20]))
 def test_threshold(self):
  self.assertEqual(descriptor_threshold(32,8,128),40)
  self.assertEqual(descriptor_threshold(32,8,170),43)
 def test_plan_gate(self):
  q=[10,20,30,40,50];k=[16384,8192,4096,512,0]
  self.assertTrue(plan_guard(q,k,43,32,8,170,False))
  self.assertFalse(plan_guard(q,k,42,32,8,170,False))
  self.assertFalse(plan_guard(q,k,43,32,8,170,True))
 def test_identity_is_order_sensitive_and_environment_bound(self):
  env=dict(gpu_uuid='u',gpu_name='g',num_sms=1,driver='d',cuda='c',torch='t',flashinfer='f',python='p',cudnn='cd',nvcc='n',backend_sha256='h',official_overlay_sha256='o',source_archive_sha256='s',selector_version=3,measurement_policy='graph-replay',profiling_mode='unprofiled')
  op=dict(execution_mode='graph16',backend='fa2',causal=True,layout='paged',dtype='bfloat16',requested_split='auto',actual_split='unsplit',num_qo_heads=32,num_kv_heads=8,head_dim_qk=128,head_dim_vo=128,page_size=16,q=[1,2],cached=[4,3],ordered_pairs_sha256='x',plan_signature=[1,2])
  a=tactic_identity(environment=env,operation=op)
  op2=dict(op,q=[2,1]);self.assertNotEqual(a,tactic_identity(environment=env,operation=op2))
  env2=dict(env,driver='e');self.assertNotEqual(a,tactic_identity(environment=env2,operation=op))
if __name__=='__main__':unittest.main()
