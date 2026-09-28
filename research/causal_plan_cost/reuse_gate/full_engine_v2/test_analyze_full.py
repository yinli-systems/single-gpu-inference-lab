"""Synthetic unit inputs only; these never enter GPU experiment evidence."""
import math
from pathlib import Path
import tempfile
import unittest
from analyze_full import confidence,load


class AnalysisContracts(unittest.TestCase):
    def test_equal_cost_no_improvement(self):
        r=confidence([0.]*6)
        self.assertEqual(r['ratio'],1.)
        self.assertEqual(r['CI95'],[1.,1.])
    def test_ratio_direction(self):
        r=confidence([math.log(2.)]*6)
        self.assertAlmostEqual(r['ratio'],2.)
        self.assertAlmostEqual(r['CI95'][0],2.)
    def test_single_screen_no_interval(self):
        with self.assertRaises(ValueError):confidence([math.log(1.1)])
    def test_missing_replica_rejected(self):
        with self.assertRaises(ValueError):confidence([0.]*5)
    def test_nonfinite_rejected(self):
        for value in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):confidence([0.]*5+[value])
    def test_interval_reproducible(self):
        values=[-.05,0.,.01,.02,.1,-.03]
        self.assertEqual(confidence(values),confidence(values))
    def test_incomplete_matrix_not_best_subset(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path);(root/'runs').mkdir()
            for i in range(5):(root/'runs'/f'test-123-{i}').mkdir()
            with self.assertRaises(ValueError):load(root,'test')
    def test_no_screen_is_not_success(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path);(root/'runs').mkdir()
            with self.assertRaises(ValueError):load(root,'screen')


if __name__=='__main__':unittest.main()
