import unittest
from diagnose import anomaly,signature,normalized_graph

class DiagnosisTests(unittest.TestCase):
 def test_gap_does_not_claim_cause(self):
  row=anomaly({'wall_us':1289.334125,'device_us':1196.938038})
  self.assertTrue(row['host_span_separation']);self.assertFalse(row['cause_proven'])
 def test_wall_only_has_no_invented_device_span(self):
  self.assertIsNone(anomaly({'wall_us':1200.,'device_us':None})['wall_minus_device_us'])
 def test_near_equal_does_not_flag(self):
  self.assertFalse(anomaly({'wall_us':1200.,'device_us':1199.})['host_span_separation'])
 def test_graph_signature_ignores_instance_id_not_launch(self):
  a={'nodes':[{'index':0,'node_type':'kernel','kernel_name':'x','grid_dim':[42,1,8],'block_dim':[32,4,1],'shared_mem_bytes':49152,'dependencies':[],'dependents':[],'graph_id':1}]}
  b={'nodes':[dict(a['nodes'][0],graph_id=2)]}
  self.assertEqual(signature(a),signature(b))
  b['nodes'][0]['shared_mem_bytes']=65536;self.assertNotEqual(signature(a),signature(b))
class FastRejectTests(unittest.TestCase):
 def test_equivalence_for_all_boolean_boundaries(self):
  from fast_reject_patch import equivalence_witness
  import itertools
  for pairing,split,descriptor,batch in itertools.product((False,True),(False,True),(False,True),range(10)):
   self.assertTrue(equivalence_witness(pairing,split,descriptor,batch))
 def test_source_rejects_unknown_anchor(self):
  from fast_reject_patch import apply
  with self.assertRaises(ValueError):apply('wrong source')
 def test_skips_only_logically_false_region(self):
  from fast_reject_patch import apply
  text='  } else {\n    // Selector v3: unchanged predicate'
  patched=apply(text);self.assertIn('!plan_info.split_kv && batch_size >= 5',patched)
if __name__=='__main__':unittest.main()
