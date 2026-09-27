"""Accounting/statistics unit fixtures, NOT measured performance evidence."""
import unittest
from analyze import best_fixed, cost_metrics, policy_metrics, select_model, POLICIES


class AnalysisTests(unittest.TestCase):
    def test_identical_policy(self):
        r=policy_metrics([1,2,4,8],[1,2,4,8],['a','a','b','b'],[1,2,4,8])
        self.assertEqual(r['family_geomean_speed_ratio'],1.)
        self.assertEqual(r['family_cluster_bootstrap_95pct_CI'],[1.,1.])
        self.assertEqual(r['regressions_over_5pct'],0)
    def test_known_ratio(self):
        r=policy_metrics([1,2,4,8],[2,4,8,16],['a','a','b','b'],[1,2,4,8])
        self.assertAlmostEqual(r['family_geomean_speed_ratio'],2.)
        self.assertEqual(r['family_cluster_bootstrap_95pct_CI'],[2.,2.])
    def test_family_macro_not_micro(self):
        r=policy_metrics([1,1,1,1],[2,2,2,.5],['a','a','a','b'],[1,1,1,.5])
        self.assertAlmostEqual(r['family_geomean_speed_ratio'],1.)
        self.assertEqual(r['clusters'],2)
    def test_regressions_included(self):
        r=policy_metrics([1.06,1,.8],[1,1,1],['a','b','c'],[1,1,.8])
        self.assertEqual(r['regressions_over_5pct'],1)
    def test_mae_not_mape(self):
        r=cost_metrics([1.,10.],[2.,8.])
        self.assertEqual(r['mae_ms'],1.5)
        self.assertAlmostEqual(r['p95_underprediction_ms'],1.9)
    def test_deterministic_bootstrap(self):
        a=policy_metrics([1,2,3],[2,2,2],['a','b','c'],[1,1,1])
        b=policy_metrics([1,2,3],[2,2,2],['a','b','c'],[1,1,1])
        self.assertEqual(a,b)
    def test_fit_rejects_test(self):
        with self.assertRaises(ValueError):
            select_model([{'shape':{'split':'test'}}],'causal','median_ms')
    def test_fixed_uses_all_candidates(self):
        rows=[{'policy':p,'median_ms':1+i} for i,p in enumerate(POLICIES)]
        p,cost=best_fixed(rows,'median_ms')
        self.assertEqual(p,'auto')
        self.assertEqual(set(cost),set(POLICIES))


if __name__=='__main__':unittest.main()
