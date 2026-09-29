import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from evidence_contract import first_difference,read_verified,validate_requests,compare_batches,release_gate

def batch():
 return dict(requests=[dict(id='a',tokens=[1,2,3]),dict(id='b',tokens=[4,5,6])],errors=[],elapsed=2.,output_tokens=6,output_tokens_per_second=3.,workload_sha256='fixed')

class EvidenceTests(unittest.TestCase):
 def test_identical(self):self.assertIsNone(first_difference([1,2],[1,2]))
 def test_positions(self):self.assertEqual(first_difference([1,2],[1,3]),1)
 def test_truncation(self):self.assertEqual(first_difference([1,2],[1]),1)
 def test_tokens_reject(self):
  for xs in [[True],[-1],[1.0],None]:
   with self.assertRaises(ValueError):first_difference(xs,[1])
 def test_denominator_is_candidate(self):self.assertEqual(compare_batches(batch(),batch())['candidate_requests'],2)
 def test_changed_workload(self):
  b=batch();b['workload_sha256']='different'
  with self.assertRaises(ValueError):compare_batches(batch(),b)
 def test_id_set(self):
  b=batch();b['requests'][0]['id']='other'
  with self.assertRaises(ValueError):compare_batches(batch(),b)
 def test_duplicate(self):
  b=batch();b['requests'][0]['id']='b'
  with self.assertRaises(ValueError):validate_requests(b)
 def test_units(self):
  b=batch();b['output_tokens_per_second']=3000.
  with self.assertRaises(ValueError):validate_requests(b)
 def test_failed(self):
  b=batch();b['errors']=['timeout']
  with self.assertRaises(ValueError):validate_requests(b)
 def test_hash(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'x.json';p.write_text('{}')
   (p.parent/'complete.json').write_text(json.dumps(dict(complete=True,files={'x.json':hashlib.sha256(p.read_bytes()).hexdigest()})))
   self.assertEqual(read_verified(p),{})
   p.write_text('{"bad":true}')
   with self.assertRaises(ValueError):read_verified(p)
 def good(self):return dict(token_parity=True,all_controls_resolve=True,all_supported_paths_qualified=True,worst_ratio_lower_bound=1.,simultaneous_bound=True,net_gain_lower_bound=1.02,independent_validation=True,old_failure_attributed=True)
 def test_gate_all(self):self.assertTrue(release_gate(**self.good())['promote'])
 def test_gate_each(self):
  for k in ['token_parity','all_controls_resolve','all_supported_paths_qualified','simultaneous_bound','independent_validation','old_failure_attributed']:
   a=self.good();a[k]=False;self.assertFalse(release_gate(**a)['promote'])
 def test_regression(self):
  a=self.good();a['worst_ratio_lower_bound']=.94;self.assertIn('bounded_regression',release_gate(**a)['failed_gates'])
 def test_no_gain(self):
  a=self.good();a['net_gain_lower_bound']=1.;self.assertFalse(release_gate(**a)['promote'])
 def test_unknown(self):
  a=self.good();a['token_parity']=None
  with self.assertRaises(ValueError):release_gate(**a)
 def test_nonfinite(self):
  for v in [float('nan'),float('inf'),0,-1]:
   a=self.good();a['net_gain_lower_bound']=v
   with self.assertRaises(ValueError):release_gate(**a)
if __name__=='__main__':unittest.main()
