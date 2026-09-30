from __future__ import annotations
import hashlib, json, shutil, tempfile, unittest
from pathlib import Path
from research.selector_v4.authorize_v4 import NATIVE_OVERLAY_REQUIREMENTS, authorize
from research.selector_v4.manifest_v4 import load
from research.selector_v4.schema import QUALIFICATION_REVISION

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

class AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        source=self.root/'source/research/selector_v4';source.mkdir(parents=True);(self.root/'receipts').mkdir();(self.root/'analysis').mkdir()
        here=Path(__file__).parent
        shutil.copyfile(here/'manifest.json',source/'manifest.json');shutil.copyfile(here/'analyze_v4.py',source/'analyze_v4.py')
        (self.root/'source/SOURCE_COMMIT.txt').write_text('commit-test\n')
        (self.root/'receipts/source-archive.sha256').write_text('a'*64+'\n');(self.root/'receipts/official-overlay.sha256').write_text('b'*64+'\n')
        self.manifest=load(source/'manifest.json')
        for gpu in ('gpu_4090','gpu_5090'):self.write_summary('canary',gpu)
    def value(self,stage,gpu):
        return {'pass':True,'qualification_revision':QUALIFICATION_REVISION,'stage':stage,'gpu':gpu,
            'case_hash':self.manifest['case_hash'],'stage_hash':self.manifest['stage_hashes'][stage],
            'source_archive_sha256':'a'*64,'official_overlay_sha256':'b'*64,
            'measurement_manifest_sha256':sha(self.root/'source/research/selector_v4/manifest.json'),
            'analysis_source_sha256':sha(self.root/'source/research/selector_v4/analyze_v4.py'),
            'campaign_source_commit':'commit-test','requirements':{key:True for key in NATIVE_OVERLAY_REQUIREMENTS},
            'metrics':{'policy_worst':1.0},'hardware':{'0':{'gpu_name':gpu}}}
    def write_summary(self,stage,gpu,value=None):
        d=self.root/f'analysis/{stage}-{gpu}';d.mkdir(exist_ok=True);(d/'summary.json').write_text(json.dumps(value or self.value(stage,gpu)))
    def test_dual_pass_authorizes_release_only(self):
        result=authorize(self.root,'release');self.assertEqual(result['authorized_stage'],'release');self.assertFalse(result['default_promotion']);self.assertFalse(result['serving_promotion'])
    def test_hold_or_provenance_drift_rejected(self):
        p=self.root/'analysis/canary-gpu_5090/summary.json';x=json.loads(p.read_text());x['pass']=False;p.write_text(json.dumps(x))
        with self.assertRaisesRegex(RuntimeError,'HOLD'):authorize(self.root,'release')
        x=self.value('canary','gpu_5090');x['source_archive_sha256']='c'*64;p.write_text(json.dumps(x))
        with self.assertRaisesRegex(RuntimeError,'provenance'):authorize(self.root,'release')
    def test_analyzer_manifest_and_commit_are_bound(self):
        p=self.root/'analysis/canary-gpu_5090/summary.json';x=json.loads(p.read_text());x['analysis_source_sha256']='0'*64;p.write_text(json.dumps(x))
        with self.assertRaisesRegex(RuntimeError,'analyzer source'):authorize(self.root,'release')
        x=self.value('canary','gpu_5090');x['measurement_manifest_sha256']='0'*64;p.write_text(json.dumps(x))
        with self.assertRaisesRegex(RuntimeError,'measurement manifest'):authorize(self.root,'release')
        x=self.value('canary','gpu_5090');x['campaign_source_commit']='old';p.write_text(json.dumps(x))
        with self.assertRaisesRegex(RuntimeError,'campaign source'):authorize(self.root,'release')
    def test_old_summary_without_native_overlay_gate_is_rejected(self):
        p=self.root/'analysis/canary-gpu_5090/summary.json';x=json.loads(p.read_text());x['requirements'].pop(NATIVE_OVERLAY_REQUIREMENTS[0]);p.write_text(json.dumps(x))
        with self.assertRaisesRegex(RuntimeError,'native-overlay'):authorize(self.root,'release')
    def test_stress_requires_dual_release_and_full_binding(self):
        with self.assertRaisesRegex(RuntimeError,'missing release'):authorize(self.root,'stress')
        for gpu in ('gpu_4090','gpu_5090'):self.write_summary('release',gpu)
        result=authorize(self.root,'stress');self.assertTrue(result['release_passed_both_gpus'])
