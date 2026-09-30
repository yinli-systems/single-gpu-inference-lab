import unittest
from policy import candidate_pool, descriptor_threshold, anti_monotone

class PolicyTests(unittest.TestCase):
    def test_hardware_wave_thresholds(self):
        self.assertEqual(descriptor_threshold(128, 8), 40)
        self.assertEqual(descriptor_threshold(170, 8), 51)
    def test_opposite_and_ties_are_candidate_pool(self):
        q=[3,71,139,207,207,311,443]; c=[23040,16896,12032,8320,8320,704,96]
        self.assertTrue(anti_monotone(q,c))
        d=candidate_pool(q=q,cached=c,padded_batch_size=52,num_qo_heads=32,
                         num_kv_heads=8,num_sms=170,split_kv=False)
        self.assertTrue(d.eligible); self.assertGreaterEqual(d.block_waves,2.4)
    def test_5090_old_47_descriptor_failure_is_excluded(self):
        q=[3,71,107,175,275,311,443]; c=[23040,16896,12032,8320,2688,704,96]
        d=candidate_pool(q=q,cached=c,padded_batch_size=47,num_qo_heads=32,
                         num_kv_heads=8,num_sms=170,split_kv=False)
        self.assertFalse(d.eligible); self.assertEqual(d.reason,'insufficient-waves')
    def test_split_and_concordant_fail_closed(self):
        q=[1,2,3,4,5]; c=[9000,10000,11000,12000,13000]
        self.assertEqual(candidate_pool(q=q,cached=c,padded_batch_size=80,
            num_qo_heads=32,num_kv_heads=8,num_sms=128,split_kv=False).reason,'pairing')
        c=list(reversed(c))
        self.assertEqual(candidate_pool(q=q,cached=c,padded_batch_size=80,
            num_qo_heads=32,num_kv_heads=8,num_sms=128,split_kv=True).reason,'split-plan')

if __name__=='__main__': unittest.main()
