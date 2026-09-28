"""CPU-only bookkeeping tests; fixtures are NOT inference results."""
import copy
import unittest
from measure_engine import observe,summarize,quantiles


def fixture():
    cells=[dict(id='r',arrival=0.,max_tokens=3)]
    records={'r':dict(id='r',admission=.2,token_ids=[1,2,3],token_times=[.5,.6,.7],finished=True)}
    return cells,records


class MeasurementContracts(unittest.TestCase):
    def test_offered_and_admitted_are_separate(self):
        cells,records=fixture();s=summarize(cells,records,1.)
        self.assertAlmostEqual(s['ttft_seconds']['p50'],.3)
        self.assertAlmostEqual(s['offered_ttft_seconds']['p50'],.5)
        self.assertAlmostEqual(s['tpot_seconds']['p50'],.1)
        self.assertEqual(s['output_tokens'],3)
    def test_empty_output_not_success(self):
        c,r=fixture();r['r']['token_ids']=[];r['r']['token_times']=[]
        with self.assertRaises(ValueError):summarize(c,r,1.)
    def test_missing_planned_request_rejected(self):
        c,r=fixture();c.append(dict(id='missing',arrival=0.,max_tokens=3))
        with self.assertRaises(ValueError):summarize(c,r,1.)
    def test_extra_request_rejected(self):
        c,r=fixture();r['extra']=copy.deepcopy(r['r'])
        with self.assertRaises(ValueError):summarize(c,r,1.)
    def test_duplicate_plan_rejected(self):
        c,r=fixture();c.append(c[0].copy())
        with self.assertRaises(ValueError):summarize(c,r,1.)
    def test_unfinished_rejected(self):
        c,r=fixture();r['r']['finished']=False
        with self.assertRaises(ValueError):summarize(c,r,1.)
    def test_truncated_length_rejected(self):
        c,r=fixture();c[0]['max_tokens']=4
        with self.assertRaises(ValueError):summarize(c,r,1.)
    def test_prefix_cannot_change(self):
        _,r=fixture()
        with self.assertRaises(ValueError):observe(r['r'],[1,9,3,4],True,.8)
    def test_tokens_cannot_shrink(self):
        _,r=fixture()
        with self.assertRaises(ValueError):observe(r['r'],[1,2],True,.8)
    def test_clock_cannot_rewind(self):
        _,r=fixture()
        with self.assertRaises(ValueError):observe(r['r'],[1,2,3,4],True,.6)
    def test_append_cumulative(self):
        _,r=fixture();self.assertEqual(observe(r['r'],[1,2,3,4],True,.8),1)
        self.assertEqual(r['r']['token_times'],[.5,.6,.7,.8])
    def test_repeated_observation_not_new_token(self):
        _,r=fixture();self.assertEqual(observe(r['r'],[1,2,3],True,.8),0)
    def test_end_timestamp_bound(self):
        c,r=fixture()
        with self.assertRaises(ValueError):summarize(c,r,.4)
    def test_invalid_walltime(self):
        for t in (0,-1,float('inf'),float('nan')):
            c,r=fixture()
            with self.assertRaises(ValueError):summarize(c,r,t)
    def test_quantiles_identical(self):self.assertEqual(quantiles([2,2,2]),dict(p50=2.,p95=2.,p99=2.))
    def test_quantiles_empty_rejected(self):
        with self.assertRaises(ValueError):quantiles([])
    def test_tpot_not_ttft(self):
        c,r=fixture();r['r']['token_times']=[1.9,1.91,1.92]
        s=summarize(c,r,2.);self.assertTrue(r['r']['strict_slo']);self.assertAlmostEqual(s['tpot_seconds']['p50'],.01)
    def test_offered_slo_includes_admission_lag(self):
        c,r=fixture();r['r']['admission']=2.;r['r']['token_times']=[2.1,2.11,2.12]
        s=summarize(c,r,3.);self.assertFalse(r['r']['strict_slo']);self.assertTrue(r['r']['lenient_slo'])


if __name__=='__main__':unittest.main()
