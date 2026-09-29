"""CPU contracts. Artificial timing fixtures below are unit tests, not benchmarks."""
from dataclasses import replace
import math
import random
import unittest

from order_guard import Context, Geometry, Probe, admit, causal_pairs, digest, propose, select, transitions


def context():
    return Context('test-device', 'test-driver', 'test-torch', 'test-cuda',
                   'a'*64, 'b'*64, 'c'*64, 'float16', 'graph', 'warm', 'run_only', 1)


def probes(c, policy, prefix, candidate=90.):
    return [Probe(prefix+str(i), c.key, policy, 100., candidate, True, True) for i in range(6)]


class GeometryTests(unittest.TestCase):
    def test_exhaustive_pair_counts(self):
        for q in range(1, 9):
            for k in range(q, q+9):
                for group in (1, 2, 3, 4):
                    for tile in (1, 3, 8):
                        for chunk in (1, 4, 17):
                            g = Geometry((q,), (k,), group, tile, chunk, True)
                            total = 0
                            for desc in g.descriptors():
                                _, t, s = desc
                                brute = sum(max(0, min(k-q+u//group+1, min(k, (s+1)*chunk))-s*chunk)
                                            for u in range(t*tile, min((t+1)*tile, q*group)))
                                self.assertEqual(causal_pairs(g, desc), brute)
                                total += brute
                            self.assertEqual(total, group*(q*(k-q)+q*(q+1)//2))

    def test_unsplit_sentinel(self):
        g = Geometry((672, 176, 96), (8864, 17072, 24160), 4, 128, -1, False)
        self.assertEqual(len(g.descriptors()), 30)
        self.assertEqual(sum(causal_pairs(g, d) for d in g.descriptors()),
                         4*sum(q*(k-q)+q*(q+1)//2 for q,k in zip(g.query, g.total_kv)))

    def test_reject_invalid_geometry(self):
        for change in ({'query': ()}, {'query': [1]}, {'total_kv': (0,)},
                       {'query': (True,)}, {'group': 0}, {'split': 1},
                       {'chunk': 0, 'split': True}, {'query': (2**31-1,), 'total_kv': (2**31-1,), 'group': 256}):
            with self.assertRaises(ValueError):
                replace(Geometry((1,), (2,), 1, 8, -1, False), **change)

    def test_proposal_bijection_and_locality(self):
        rng = random.Random(178)
        for _ in range(300):
            n = rng.randint(1, 10)
            q = tuple(rng.randint(1, 256) for _ in range(n))
            k = tuple(x+rng.randint(0, 512) for x in q)
            g = Geometry(q, k, 4, 64, 128, rng.choice([True, False]))
            desc = g.descriptors()
            for policy in ('identity', 'causal_heavy', 'locality_packet8'):
                order = propose(g, desc, policy)
                self.assertEqual(sorted(order), list(range(len(desc))))
                self.assertEqual(order, propose(g, desc, policy))
                if policy == 'locality_packet8':
                    self.assertEqual(order[:8], tuple(range(min(8, len(desc)))))
                    self.assertLessEqual(transitions(desc, order), transitions(desc, range(len(desc))))
                    self.assertLessEqual(max(abs(i-j) for i,j in enumerate(order)), 31)
                    for begin in range(8, len(desc), 32):
                        self.assertEqual(set(order[begin:begin+32]), set(range(begin,min(begin+32,len(desc)))))

    def test_corrupt_descriptor_rejected(self):
        g = Geometry((32,), (128,), 4, 64, 64, True)
        ds = g.descriptors()
        for desc in (ds[:-1], ds+(ds[0],), ((True,0,0),)+ds[1:], ((3,0,0),)+ds[1:]):
            with self.assertRaises(ValueError):propose(g, desc, 'identity')
        with self.assertRaises(ValueError):propose(g, ds, 'mystery')
        with self.assertRaises(ValueError):causal_pairs(g, (0, 10, 0))


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.c = context()
        self.sel = probes(self.c, 'heavy', 's')
        self.ids = {p.trial_id for p in self.sel}
        self.conf = probes(self.c, 'heavy', 'v')
        self.aa = probes(self.c, 'identity_repeat', 'a', 100.)
        self.costs = dict(probe_cost_us=1000., extra_setup_us=50., dispatch_us=1., expected_reuses=200)

    def call(self, **kw):
        args = dict(context=self.c, selected='heavy', selection_ids=self.ids,
                    confirmation=self.conf, aa=self.aa, **self.costs)
        args.update(kw)
        return admit(**args)

    def test_exact_context_sensitivity(self):
        for key, value in vars(self.c).items():
            v = value+1 if type(value) is int else value+'-other'
            if key.endswith('sha256'):v = 'd'*64
            if key == 'mode':v='eager'
            if key == 'timing_scope':v='plan_plus_run'
            self.assertNotEqual(replace(self.c, **{key:v}).key, self.c.key)

    def test_selection_and_admission(self):
        self.assertEqual(select(self.c, self.sel), 'heavy')
        out = self.call()
        self.assertEqual(out['policy'], 'heavy')
        self.assertEqual(out['break_even_reuses'], 117)
        self.assertFalse(out['production_qualified'])

    def test_cost_and_reuse_fallbacks(self):
        for values, reason in [({'expected_reuses':None},'unknown_reuse'),
                               ({'expected_reuses':36},'insufficient_reuse'),
                               ({'dispatch_us':12.},'dispatch_exceeds_gain')]:
            self.assertEqual(self.call(**values)['reason'], reason)

    def test_nonfinite_cost_rejected(self):
        for field in ('probe_cost_us','extra_setup_us','dispatch_us'):
            for v in (-1., math.nan, math.inf, True):
                with self.assertRaises(ValueError):self.call(**{field:v})

    def test_wrong_context_falls_back(self):
        c2 = replace(self.c, mode='eager')
        self.assertEqual(self.call(confirmation=probes(c2,'heavy','v'))['reason'], 'context_mismatch')

    def test_disjointness_all_phases(self):
        with self.assertRaises(ValueError):self.call(confirmation=self.sel)
        with self.assertRaises(ValueError):self.call(aa=probes(self.c,'identity_repeat','v',100.))
        with self.assertRaises(ValueError):select(self.c,self.sel+self.sel)

    def test_exact_six_confirmation_no_optional_stopping(self):
        for conf in (self.conf[:5], self.conf+[replace(self.conf[0],trial_id='v-new')]):
            self.assertEqual(self.call(confirmation=conf)['policy'], 'identity')

    def test_no_runner_up_on_failure(self):
        conf = probes(self.c,'heavy','v',105.)
        self.assertEqual(self.call(confirmation=conf)['reason'], 'confirmation_gain_failed')

    def test_AA_failure(self):
        self.assertEqual(self.call(aa=probes(self.c,'identity_repeat','a',103.1))['reason'], 'AA_observed_envelope_failed')

    def test_bad_numeric_receipt(self):
        for field,v in [('exact_output',False),('exact_lse',False),('baseline_us',math.inf),('candidate_us',0.)]:
            with self.assertRaises(ValueError):replace(self.sel[0],**{field:v})

    def test_selection_insufficient_or_no_gain(self):
        self.assertEqual(select(self.c, self.sel[:5]), 'identity')
        self.assertEqual(select(self.c, probes(self.c,'heavy','s',100.)), 'identity')
        with self.assertRaises(ValueError):select(replace(self.c,driver='other'),self.sel)


if __name__ == '__main__':
    unittest.main()
