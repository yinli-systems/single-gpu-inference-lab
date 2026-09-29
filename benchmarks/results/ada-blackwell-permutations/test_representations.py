import random,unittest
from representations import reconstruct_work,features,audit
from campaign import CONFIGS,states,geometry
class RepresentationContracts(unittest.TestCase):
 def test_identity_random(self):
  rng=random.Random(20260929)
  for _ in range(2000):
   n=rng.randrange(1,33);q=[rng.randrange(1,2049) for _ in range(n)];k=[rng.randrange(0,32769) for _ in range(n)]
   actual=reconstruct_work(sum(x*x for x in q),sum(x*x for x in k),sum((a+b)**2 for a,b in zip(q,k)),sum(q))
   self.assertEqual(actual,sum(a*b+a*(a+1)//2 for a,b in zip(q,k)))
 def test_q_k_marginals_alias_different_work(self):
  rows=[r for r in audit()['rows'] if r['feature']=='query_cached_marginals']
  self.assertTrue(all(r['same_opposite_alias'] and r['distinct_work_alias_classes']==1 for r in rows))
 def test_augmented_moments_recover_work(self):
  rows=[r for r in audit()['rows'] if r['feature']=='augmented_marginal_moments']
  self.assertTrue(all(not r['same_opposite_alias'] and r['distinct_work_alias_classes']==0 for r in rows))
 def test_equal_work_does_not_imply_equal_pairs(self):
  for n in ('n4','n8','n16'):
   fs={l:features(geometry(n,l,p)) for l,p in states(n)}
   self.assertEqual(fs['eqC-a']['augmented_marginal_moments'],fs['eqC-b']['augmented_marginal_moments'])
   self.assertNotEqual(fs['eqC-a']['paired_set'],fs['eqC-b']['paired_set'])
 def test_boolean_rejected(self):
  with self.assertRaises(ValueError):reconstruct_work(True,0,1,1)
 def test_bad_moments(self):
  with self.assertRaises(ValueError):reconstruct_work(1,2,2,1)
if __name__=='__main__':unittest.main()
