import json,unittest
from pathlib import Path
from manifest import load,guarded,digest
class ReleaseTests(unittest.TestCase):
 def setUp(self):self.m=load()
 def test_manifest(self):
  self.assertEqual(len(self.m['cases']),50);self.assertEqual(sum(x['family']=='release' for x in self.m['cases']),48)
  self.assertEqual(digest(self.m['cases']),self.m['case_hash'])
 def test_guard_frozen(self):
  for c in self.m['cases']:self.assertEqual(guarded(c['q'],c['cached']),c['guarded_expected'])
  self.assertEqual(sum(c['guarded_expected'] for c in self.m['cases'] if c['family']=='release'),24)
 def test_no_old_case_collision(self):
  old=json.loads((Path(__file__).with_name('exposed-old-manifest.json')).read_text())
  a={(tuple(c['q']),tuple(c['cached'])) for c in self.m['cases'] if c['family']=='release'}
  b={(tuple(c['q']),tuple(c['cached'])) for c in old['cases']}
  self.assertFalse(a&b)
 def test_boundary_native(self):
  for c in self.m['cases']:
   if c['regime'] in ('boundary','balanced','control'):self.assertFalse(c['guarded_expected'])
if __name__=='__main__':unittest.main()
