"""CPU contract tests; these are not GPU correctness or performance results."""
import math
import unittest
from geometry import Shape, POLICIES, corpus, corpus_json, features, plan, workspace_bytes


class GeometryTests(unittest.TestCase):
    def test_workspace_no_split(self):
        self.assertEqual(workspace_bytes(corpus()[0],16,4,128,'none'),16)
    def test_workspace_covers_partial_rows(self):
        s=next(s for s in corpus() if s.name=='discovery-eq255-A')
        p=plan(s,32,8,128,'s512')
        self.assertGreaterEqual(workspace_bytes(s,32,8,128,'s512'),32*p['grid_x']*p['tile']*129*4)
    def test_canary_names_exist(self):
        names={s.name for s in corpus()}
        self.assertTrue({'Q512-K1536-eq127-A','Q2048-K6144-n8-A'}<=names)
    def test_empty(self):
        with self.assertRaises(ValueError):Shape('x','x','train',(),())
    def test_alignment(self):
        with self.assertRaises(ValueError):Shape('x','x','train',(1,2),(1,))
    def test_bool_rejected(self):
        with self.assertRaises(TypeError):Shape('x','x','train',(True,),(2,))
    def test_negative(self):
        with self.assertRaises(ValueError):Shape('x','x','train',(1,),(-1,))
    def test_zero_query(self):
        with self.assertRaises(ValueError):Shape('x','x','train',(0,),(1,))
    def test_corpus_deterministic(self):self.assertEqual(corpus_json(),corpus_json())
    def test_split_sizes(self):
        ss=corpus()
        self.assertEqual(sum(s.split=='train' for s in ss),72)
        self.assertEqual(sum(s.split=='test' for s in ss),72)
        self.assertEqual(sum(s.split=='diagnostic' for s in ss),8)
    def test_no_duplicate_geometry(self):
        ss=corpus();self.assertEqual(len(ss),len(set((s.q,s.k) for s in ss)))
    def test_family_isolation(self):
        sets={split:{s.family for s in corpus() if s.split==split} for split in ('train','test','diagnostic')}
        self.assertFalse(sets['train'] & sets['test'])
        self.assertFalse(sets['diagnostic'] & (sets['train']|sets['test']))
    def test_equal_work_pairs(self):
        pairs={s.name:s for s in corpus()}
        for s in pairs.values():
            if '-eq' in s.name and s.name.endswith('-A'):
                other=pairs[s.name[:-1]+'B']
                self.assertEqual(s.work,other.work)
                self.assertEqual(sum(s.q),sum(other.q))
                self.assertEqual(sum(s.k),sum(other.k))
    def test_swap_identity(self):
        for q,k in [((7,13),(2,11)),((255,769),(4096,16384))]:
            a=Shape('a','x','train',q,k);b=Shape('b','x','train',q[::-1],k)
            self.assertEqual(a.work-b.work,(q[0]-q[1])*(k[0]-k[1]))
    def test_invalid_policy(self):
        with self.assertRaises(ValueError):plan(corpus()[0],16,4,128,'oracle')
    def test_invalid_heads(self):
        with self.assertRaises(ValueError):plan(corpus()[0],15,4,128,'auto')
    def test_no_split(self):
        s=corpus()[0];p=plan(s,16,4,128,'none')
        self.assertFalse(p['split']);self.assertEqual(p['merge_rows'],0)
        self.assertEqual(p['grid_x'],sum((q*4+127)//128 for q in s.q))
    def test_fixed_split(self):
        s=corpus()[0];p=plan(s,16,4,128,'s512')
        self.assertEqual(p['kv_chunk'],512)
    def test_causal_iterations_bounded(self):
        for s in corpus():
            for policy in POLICIES:
                p=plan(s,16,4,128,policy)
                self.assertEqual(p['grid_x'],len(p['task_iterations']))
                self.assertTrue(all(0<=t<=r for t,r in zip(p['task_iterations'],p['rectangular_iterations'])))
    def test_feature_schema(self):
        lengths={}
        for kind in ('marginal','joint','plan','causal'):
            for s in corpus()[::11]:
                for mode in POLICIES:
                    f=features(s,16,4,128,mode,kind)
                    self.assertTrue(all(math.isfinite(x) for x in f))
                    lengths.setdefault(kind,len(f));self.assertEqual(lengths[kind],len(f))
        self.assertGreater(lengths['causal'],lengths['plan'])
    def test_zero_cached_depth(self):
        s=Shape('x','x','train',(128,),(0,));self.assertEqual(s.work,128*129//2)
    def test_diagnostic_not_train(self):
        self.assertTrue(all(s.split=='diagnostic' for s in corpus() if s.family=='discovery'))


if __name__=='__main__':unittest.main()
