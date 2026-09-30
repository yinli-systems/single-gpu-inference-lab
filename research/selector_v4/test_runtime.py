from __future__ import annotations
import tempfile, unittest
from types import SimpleNamespace
from research.selector_v4.cache import SafeTacticCache
from research.selector_v4.eligibility import evaluate_eligibility
from research.selector_v4.runtime import apply_cached_tactic, publish_and_apply
from research.selector_v4.safe_tuner import choose_tactic
from research.selector_v4.schema import TACTIC_CAP, TACTIC_NATIVE
from research.selector_v4.test_core import identity

class RuntimeTests(unittest.TestCase):
    def wrapper(self, valid=True):
        return SimpleNamespace(_backend='fa2' if valid else 'fa3',_jit_module=None,_plan_info=list(identity().operation["plan_signature"]),
            run=lambda: 'native',run_resource=lambda: 'resource',
            _cached_module=SimpleNamespace(paged_run_resource=lambda: None,ragged_run_resource=lambda: None,resource_kernel_isolation=True),_sgi_resource_policy=99)
    def test_miss_is_native_before_plan(self):
        x=identity()
        with tempfile.TemporaryDirectory() as tmp:
            w=self.wrapper();r=apply_cached_tactic(w,x,SafeTacticCache(tmp,x));self.assertEqual(r.tactic,TACTIC_NATIVE);self.assertEqual(w._sgi_resource_policy,0)
    def test_valid_cap_hit_applies_one(self):
        x=identity();receipt=choose_tactic(x,evaluate_eligibility(x),[1.2]*32,[1.]*32,[1.]*32,exact_outputs=True).to_dict()
        with tempfile.TemporaryDirectory() as tmp:
            w=self.wrapper();r=publish_and_apply(w,x,SafeTacticCache(tmp,x),receipt,{'job':'test'});self.assertEqual(r.tactic,TACTIC_CAP);self.assertEqual(w._sgi_resource_policy,1)
    def test_missing_isolation_marker_falls_back_native(self):
        x=identity();receipt=choose_tactic(x,evaluate_eligibility(x),[1.2]*32,[1.]*32,[1.]*32,exact_outputs=True).to_dict()
        with tempfile.TemporaryDirectory() as tmp:
            cache=SafeTacticCache(tmp,x);cache.publish(x,receipt,provenance={});cache.reload();w=self.wrapper();w._cached_module.resource_kernel_isolation=False
            r=apply_cached_tactic(w,x,cache);self.assertEqual(r.tactic,TACTIC_NATIVE);self.assertEqual(w._sgi_resource_policy,0)
    def test_changed_or_unplanned_wrapper_fails_closed(self):
        x=identity();receipt=choose_tactic(x,evaluate_eligibility(x),[1.2]*32,[1.]*32,[1.]*32,exact_outputs=True).to_dict()
        with tempfile.TemporaryDirectory() as tmp:
            cache=SafeTacticCache(tmp,x);cache.publish(x,receipt,provenance={});cache.reload()
            for plan in ([],[1]*15):
                w=self.wrapper();w._plan_info=plan
                r=apply_cached_tactic(w,x,cache)
                self.assertEqual(r.tactic,TACTIC_NATIVE)
                self.assertIs(w._sgi_tactic_run,w.run)

    def test_unsupported_wrapper_revalidates_to_native(self):
        x=identity();receipt=choose_tactic(x,evaluate_eligibility(x),[1.2]*32,[1.]*32,[1.]*32,exact_outputs=True).to_dict()
        with tempfile.TemporaryDirectory() as tmp:
            cache=SafeTacticCache(tmp,x);cache.publish(x,receipt,provenance={});cache.reload();w=self.wrapper(False);r=apply_cached_tactic(w,x,cache);self.assertEqual(r.tactic,TACTIC_NATIVE);self.assertEqual(w._sgi_resource_policy,0)
