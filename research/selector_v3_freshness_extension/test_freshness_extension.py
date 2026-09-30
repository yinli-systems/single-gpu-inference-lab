import json,math,unittest
from pathlib import Path
from manifest import load,strict_opposite

class FreshnessExtensionTests(unittest.TestCase):
 def test_manifest_pair_is_new_and_frozen(self):
  m=load(); self.assertTrue(m['strict_freshness_extension']); self.assertEqual(len(m['cases']),2)
  primary=json.loads(Path(__file__).with_name('PRIMARY_MANIFEST.json').read_text())
  old={(tuple(c['q']),tuple(c['cached'])) for c in primary['cases']}
  for c in m['cases']:
   self.assertNotIn((tuple(c['q']),tuple(c['cached'])),old)
   self.assertEqual(sum(math.ceil(v/32) for v in c['q']),c['descriptor_count'])
  opp,near=m['cases']; self.assertTrue(strict_opposite(opp['q'],opp['cached'])); self.assertFalse(strict_opposite(near['q'],near['cached']))
  self.assertEqual(sorted(opp['cached']),sorted(near['cached']))

if __name__=='__main__':unittest.main()
