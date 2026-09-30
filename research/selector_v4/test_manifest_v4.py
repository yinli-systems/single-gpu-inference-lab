from __future__ import annotations
import json, unittest
from pathlib import Path
from research.selector_v4.manifest_v4 import digest, load

class ManifestTests(unittest.TestCase):
    def test_counts_hash_and_internal_freshness(self):
        data=load();self.assertEqual(data["families"],{"dev":2,"canary":10,"release":48,"stress":12})
        self.assertEqual(data["case_hash"],digest(data["cases"]));self.assertEqual(data["qualification_revision"],"4.1.0");self.assertEqual(data["blocks"],16)
        fresh=[c for c in data["cases"] if not c.get("exposed_development")]
        keys=[(tuple(c["q"]),tuple(c["cached"])) for c in fresh]
        self.assertEqual(len(keys),len(set(keys)))
        self.assertEqual(len(fresh),len({tuple(c["q"]) for c in fresh}))
        self.assertEqual(len(fresh),len({tuple(c["cached"]) for c in fresh}))
    def test_descriptor_and_shape_contracts(self):
        for case in load()["cases"]:
            self.assertEqual(case["descriptor_count"],sum((x+31)//32 for x in case["q"]))
            self.assertEqual(case["batch_size"],len(case["q"]));self.assertEqual(len(case["q"]),len(case["cached"]))
            self.assertEqual(case["shape_eligible"],case["batch_size"]>=5 and max(case["cached"])>=8192)
    def test_exact_freshness_against_recorded_manifests(self):
        root=Path(__file__).resolve().parents[2];data=load();new={(tuple(c["q"]),tuple(c["cached"])) for c in data["cases"] if not c.get("exposed_development")}
        self.assertNotIn("research/selector_v4/manifest.json",data["historical_manifest_hashes"])
        for rel,expected in data["historical_manifest_hashes"].items():
            path=root/rel;self.assertTrue(path.exists());import hashlib
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),expected)
            old=json.loads(path.read_text())
            oldcases=[c for c in old.get("cases",[]) if "q" in c and "cached" in c]
            oldkeys={(tuple(c["q"]),tuple(c["cached"])) for c in oldcases}
            oldq={tuple(c["q"]) for c in oldcases}; oldcached={tuple(c["cached"]) for c in oldcases}
            self.assertFalse(new & oldkeys,rel)
            self.assertFalse({tuple(c["q"]) for c in data["cases"] if not c.get("exposed_development")} & oldq,rel+":q")
            self.assertFalse({tuple(c["cached"]) for c in data["cases"] if not c.get("exposed_development")} & oldcached,rel+":cached")
