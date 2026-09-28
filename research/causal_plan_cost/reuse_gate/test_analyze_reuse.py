"""Synthetic contract tests; no synthetic measurements enter reported results."""
import tempfile
from pathlib import Path
import unittest
from analyze_reuse import ratio_stats,check_counters,load


class AnalysisContracts(unittest.TestCase):
    def test_identical(self):
        r=ratio_stats([1,2,3,4],[1,2,3,4],['a','a','b','b'])
        self.assertEqual(r['ratio'],1.)
        self.assertEqual(r['CI95'],[1.,1.])
        self.assertEqual(r['regressions_gt5pct'],0)
    def test_ratio_direction(self):
        r=ratio_stats([1,1],[2,2],['a','b'])
        self.assertAlmostEqual(r['ratio'],2.)
        self.assertEqual(r['CI95'],[2.,2.])
    def test_family_macro(self):
        r=ratio_stats([1,1,1,2],[2,2,2,2],['a','a','a','b'])
        self.assertAlmostEqual(r['ratio'],2**.5)
    def test_regressions_retained(self):
        r=ratio_stats([1,2,3],[1,1,1],['a','b','c'])
        self.assertEqual(r['regressions_gt5pct'],2)
        self.assertEqual(r['max_slowdown'],3.)
    def test_deterministic_interval(self):
        self.assertEqual(ratio_stats([1,2],[2,1],['a','b']),ratio_stats([1,2],[2,1],['a','b']))
    def test_invalid_timing(self):
        for x in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):ratio_stats([x],[1],['a'])
    def test_misaligned(self):
        with self.assertRaises(ValueError):ratio_stats([1],[1,2],['a'])
    def test_empty(self):
        with self.assertRaises(ValueError):ratio_stats([],[],[])
    def test_all_lifetimes(self):
        fixtures={'unchanged':(1,3,1,0),'alternating':(4,2,2,0),'eviction':(4,0,4,2)}
        for name,counts in fixtures.items():
            c=dict(zip(('plans','hits','misses','evictions'),counts),runs=144,policies=['auto']*4)
            check_counters(c,36,name)
    def test_stale_active_plan_rejected(self):
        c=dict(plans=1,hits=2,misses=2,evictions=0,runs=144,policies=['auto']*4)
        with self.assertRaises(ValueError):check_counters(c,36,'alternating')
    def test_insufficient_actual_layer_runs_rejected(self):
        c=dict(plans=1,hits=3,misses=1,evictions=0,runs=36,policies=['auto']*4)
        with self.assertRaises(ValueError):check_counters(c,36,'unchanged')
    def test_partial_matrix_not_reported(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'runs').mkdir()
            with self.assertRaises(ValueError):load(Path(d))


if __name__=='__main__':unittest.main()
