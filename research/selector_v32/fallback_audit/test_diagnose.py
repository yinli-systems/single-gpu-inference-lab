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
if __name__=='__main__':unittest.main()
