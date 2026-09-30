from __future__ import annotations
import json, tempfile, unittest
from pathlib import Path
from research.selector_v4.cache import SafeTacticCache
from research.selector_v4.crossfit import RepeatEvidence, crossfit_cell
from research.selector_v4.eligibility import evaluate_eligibility
from research.selector_v4.identity import TacticIdentity
from research.selector_v4.safe_tuner import choose_tactic
from research.selector_v4.schema import DEFAULT_THRESHOLDS, TACTIC_CAP, TACTIC_NATIVE


def identity(execution="graph16_replay", split="unsplit", uuid="GPU-test"):
    return TacticIdentity(
        environment={"gpu_name":"NVIDIA Test","gpu_uuid":uuid,"num_sms":170,"driver":"580.82.07",
            "cuda":"13.0","torch":"2.13.0","flashinfer":"0.7.0","nvcc":"13.0",
            "backend_source_sha256":"a"*64,"official_overlay_sha256":"b"*64,
            "max_smem_per_sm":102400,"max_smem_per_block_optin":101376},
        operation={"execution_mode":execution,"backend":"fa2","causal":True,"layout":"paged",
            "dtype":"float16","actual_split":split,"num_qo_heads":32,"num_kv_heads":8,
            "head_dim_qk":128,"head_dim_vo":128,"page_size":16,"q":[3,35,99,163,259],
            "cached":[32768,16384,8192,2048,64],"plan_signature":[40]+[0]*14},
        measurement_policy={"execution_mode":execution,"timer":"deployment_wall","qualification_revision":"4.0.0"},
    )

class IdentityEligibilityTests(unittest.TestCase):
    def test_identity_binds_execution_and_environment(self):
        self.assertNotEqual(identity().key, identity("eager_full_call").key)
        self.assertNotEqual(identity().key, identity(uuid="GPU-other").key)

    def test_tactic_flag_is_rejected_from_operation_identity(self):
        a=identity();op=dict(a.operation);op["plan_signature"]=[40]+[0]*15
        with self.assertRaisesRegex(ValueError,"exclude tactic-specific"):
            TacticIdentity(a.environment,op,a.measurement_policy).validate()
    def test_eligibility_fail_closed(self):
        self.assertTrue(evaluate_eligibility(identity()).eligible)
        self.assertFalse(evaluate_eligibility(identity(split="split")).eligible)
        x=identity(); op=dict(x.operation);op["cached"]=[4096,2048,1024,512,64]
        self.assertFalse(evaluate_eligibility(TacticIdentity(x.environment,op,x.measurement_policy)).eligible)

class SafeTunerTests(unittest.TestCase):
    def test_strong_evidence_selects_cap(self):
        x=identity(); e=evaluate_eligibility(x)
        r=choose_tactic(x,e,[1.20]*16,[1.0]*16,[1.0]*16,exact_outputs=True)
        self.assertEqual(r.tactic,TACTIC_CAP);self.assertTrue(r.passed)
    def test_control_or_gain_failure_falls_back(self):
        x=identity();e=evaluate_eligibility(x)
        self.assertEqual(choose_tactic(x,e,[1.02]*16,[1.0]*16,[1.0]*16,exact_outputs=True).tactic,TACTIC_NATIVE)
        self.assertEqual(choose_tactic(x,e,[1.20]*16,[1.02]*16,[1.0]*16,exact_outputs=True).tactic,TACTIC_NATIVE)
        self.assertEqual(choose_tactic(x,e,[1.20]*16,[1.0]*16,[1.0]*16,exact_outputs=False).tactic,TACTIC_NATIVE)
    def test_crossfit_never_applies_failed_tactic(self):
        x=identity();e=evaluate_eligibility(x)
        reps={0:RepeatEvidence((1.2,)*8,(1.,)*8,(1.,)*8),1:RepeatEvidence((1.2,)*8,(1.,)*8,(1.,)*8),2:RepeatEvidence((.98,)*8,(1.,)*8,(1.,)*8)}
        folds=crossfit_cell(x,e,reps,exact_outputs=True)
        self.assertEqual(folds[2].receipt.tactic,TACTIC_CAP)
        self.assertLess(folds[2].policy_geomean,1.0)
        self.assertEqual(folds[0].receipt.tactic,TACTIC_NATIVE)
        self.assertEqual(folds[0].policy_geomean,1.0)

class CacheTests(unittest.TestCase):
    def test_atomic_publish_lookup_corruption_and_env_isolation(self):
        x=identity();e=evaluate_eligibility(x);receipt=choose_tactic(x,e,[1.2]*16,[1.]*16,[1.]*16,exact_outputs=True).to_dict()
        with tempfile.TemporaryDirectory() as tmp:
            cache=SafeTacticCache(tmp,x);path=cache.publish(x,receipt,provenance={"job":"test"})
            self.assertEqual(cache.lookup(x)["tactic"],TACTIC_CAP)
            self.assertFalse(any(p.suffix==".tmp" for p in path.parent.iterdir()))
            path.write_text("not json");cache.reload();self.assertIsNone(cache.lookup(x))
            with self.assertRaises(ValueError):SafeTacticCache(tmp,identity(uuid="other")).publish(x,receipt,provenance={})
    def test_invalid_cap_receipt_not_published(self):
        x=identity();bad={"identity_key":x.key,"tactic":TACTIC_CAP,"passed":False,"reason":"no","thresholds":DEFAULT_THRESHOLDS.to_dict()}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):SafeTacticCache(tmp,x).publish(x,bad,provenance={})
