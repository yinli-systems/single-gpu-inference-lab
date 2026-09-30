import json,tempfile,unittest
from pathlib import Path
from autotune import decide,TacticStore,canonical_hash

def grid(value): return [[value]*8 for _ in range(3)]

class AutotuneTests(unittest.TestCase):
    def test_strong_gain_selects_cap(self):
        e=decide(native_over_cap=grid(1.20),null_controls=grid(1.0),
                 exact_outputs=True,clock_stable=True,candidate_pool=True)
        self.assertEqual(e.tactic,'cap');self.assertGreater(e.ci95[0],1.01)
    def test_old_5090_regression_rejects_cap(self):
        e=decide(native_over_cap=grid(.9874),null_controls=grid(1.0),
                 exact_outputs=True,clock_stable=True,candidate_pool=True)
        self.assertEqual(e.tactic,'native');self.assertEqual(e.reason,'margin-not-met')
    def test_unresolved_control_and_numerics_fail_closed(self):
        e=decide(native_over_cap=grid(1.3),null_controls=grid(.99),
                 exact_outputs=True,clock_stable=True,candidate_pool=True)
        self.assertEqual(e.tactic,'native');self.assertEqual(e.reason,'control-unresolved')
        e=decide(native_over_cap=grid(1.3),null_controls=grid(1.0),
                 exact_outputs=False,clock_stable=True,candidate_pool=True)
        self.assertEqual(e.tactic,'native')
    def test_outside_pool_and_insufficient_evidence_fail_closed(self):
        e=decide(native_over_cap=[],null_controls=[],exact_outputs=True,
                 clock_stable=True,candidate_pool=False)
        self.assertEqual(e.tactic,'native')
        e=decide(native_over_cap=[[1.2]*8],null_controls=[[1.0]*8],
                 exact_outputs=True,clock_stable=True,candidate_pool=True)
        self.assertEqual(e.tactic,'native');self.assertTrue(e.reason.startswith('invalid-evidence'))
    def test_store_atomic_hydration_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tactics.json';identity={'environment':{'gpu':'x'},'operation':{'mode':'graph16'}}
            evidence=decide(native_over_cap=grid(1.2),null_controls=grid(1.0),
                            exact_outputs=True,clock_stable=True,candidate_pool=True)
            store=TacticStore(path);self.assertEqual(store.lookup(identity),'native')
            store.publish(identity,evidence);self.assertEqual(store.lookup(identity),'cap')
            fresh=TacticStore(path);self.assertEqual(fresh.lookup(identity),'cap')
            data=json.loads(path.read_text());self.assertIn(canonical_hash(identity),data['records'])
            path.write_text('{broken');self.assertEqual(TacticStore(path).lookup(identity),'native')

if __name__=='__main__': unittest.main()

class StoreConcurrencyTests(unittest.TestCase):
    def test_stale_writer_merges_instead_of_clobbering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tactics.json'
            a=TacticStore(path);b=TacticStore(path)
            # Hydrate both before either publication to emulate stale processes.
            self.assertEqual(a.lookup({'environment':{'gpu':'a'},'operation':{'mode':'eager'}}),'native')
            self.assertEqual(b.lookup({'environment':{'gpu':'b'},'operation':{'mode':'graph'}}),'native')
            evidence=decide(native_over_cap=grid(1.2),null_controls=grid(1.0),exact_outputs=True,clock_stable=True,candidate_pool=True)
            ia={'environment':{'gpu':'a'},'operation':{'mode':'eager'}}
            ib={'environment':{'gpu':'b'},'operation':{'mode':'graph'}}
            a.publish(ia,evidence);b.publish(ib,evidence)
            data=json.loads(path.read_text())
            self.assertEqual(set(data['records']),{canonical_hash(ia),canonical_hash(ib)})
