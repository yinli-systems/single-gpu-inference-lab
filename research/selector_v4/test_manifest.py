import unittest
from manifest import load,digest

class ManifestTests(unittest.TestCase):
    def test_frozen_counts_hashes_and_no_duplicates(self):
        m=load();self.assertEqual(len([c for c in m['cases'] if c['family']=='canary']),8)
        release=[c for c in m['cases'] if c['family']=='release'];self.assertEqual(len(release),48)
        self.assertEqual(digest(release),m['release_hash'])
        signatures={(tuple(c['q']),tuple(c['cached'])) for c in m['cases']}
        self.assertEqual(len(signatures),56)
    def test_broad_batches_and_regimes(self):
        m=load();r=[c for c in m['cases'] if c['family']=='release']
        self.assertGreaterEqual(len({c['batch_size'] for c in r}),5)
        self.assertEqual({c['regime'] for c in r},{'opposite','tied-opposite','near-opposite','same-paired','mixed','bimodal'})

if __name__=='__main__': unittest.main()
