import unittest
from collisions import family, representation, audit
from manifest import build


class CollisionTests(unittest.TestCase):
    def test_stronger_three_marginal_collision(self):
        a,b=family(64,64,0)
        self.assertEqual(representation(a),representation(b))
        self.assertEqual([len(a.descriptors()),len(b.descriptors())],[22,20])
        self.assertNotEqual(list(zip(a.query,a.total_kv)),list(zip(b.query,b.total_kv)))

    def test_full_family(self):
        r=audit()
        self.assertEqual(r['checked_pairs'],1080)
        self.assertGreater(r['different_descriptor_counts'],0)
        self.assertFalse(r['GPU_used'])

    def test_manifest_unmeasured_and_disjoint(self):
        m=build();self.assertEqual(m,build())
        self.assertEqual(m['GPU_observations'],0)
        self.assertEqual(len(m['cases']),57)
        self.assertEqual(sum(r['exposed'] for r in m['cases']),1)
        self.assertEqual(sum(r['kind']=='fresh' for r in m['cases']),48)


if __name__=='__main__':unittest.main()
