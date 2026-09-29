import json,tempfile,unittest
from pathlib import Path
from audit_vidur import audit
BASE=Path(__file__).resolve().parent
class UpstreamContracts(unittest.TestCase):
 def test_matches_frozen_source_audit(self):
  self.assertEqual(audit(BASE/'reference_sources/vidur/sklearn_execution_time_predictor.py'),json.loads((BASE/'vidur-source-audit.json').read_text()))
 def test_json_roundtrip_preserves_types(self):
  result=audit(BASE/'reference_sources/vidur/sklearn_execution_time_predictor.py')
  self.assertEqual(result,json.loads(json.dumps(result)))
  self.assertEqual(result['calls'],108)
 def test_unreviewed_source_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'fake.py';p.write_text('raise RuntimeError("must not execute")')
   with self.assertRaises(ValueError):audit(p)
if __name__=='__main__':unittest.main()
