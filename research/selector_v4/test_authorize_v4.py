from __future__ import annotations
import json, shutil, tempfile, unittest
from pathlib import Path
from research.selector_v4.authorize_v4 import authorize
from research.selector_v4.manifest_v4 import load
from research.selector_v4.schema import QUALIFICATION_REVISION

class AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        (self.root/'source/research/selector_v4').mkdir(parents=True);(self.root/'receipts').mkdir();(self.root/'analysis').mkdir()
        manifest=Path(__file__).with_name('manifest.json');shutil.copyfile(manifest,self.root/'source/research/selector_v4/manifest.json')
        (self.root/'receipts/source-archive.sha256').write_text('a'*64+'\n');(self.root/'receipts/official-overlay.sha256').write_text('b'*64+'\n')
        self.case_hash=load(manifest)['case_hash']
        for gpu in ('gpu_4090','gpu_5090'):
            d=self.root/f'analysis/canary-{gpu}';d.mkdir();(d/'summary.json').write_text(json.dumps({
                'pass':True,'qualification_revision':QUALIFICATION_REVISION,'case_hash':self.case_hash,
                'source_archive_sha256':'a'*64,'official_overlay_sha256':'b'*64,
                'metrics':{'policy_worst':1.0},'hardware':{'0':{'gpu_name':gpu}}
            }))
    def test_dual_pass_authorizes_release_only(self):
        result=authorize(self.root,'release');self.assertTrue(result['authorized_stage']=='release');self.assertFalse(result['default_promotion']);self.assertFalse(result['serving_promotion'])
    def test_hold_or_provenance_drift_rejected(self):
        p=self.root/'analysis/canary-gpu_5090/summary.json';x=json.loads(p.read_text());x['pass']=False;p.write_text(json.dumps(x))
        with self.assertRaisesRegex(RuntimeError,'HOLD'):authorize(self.root,'release')
        x['pass']=True;x['source_archive_sha256']='c'*64;p.write_text(json.dumps(x))
        with self.assertRaisesRegex(RuntimeError,'provenance'):authorize(self.root,'release')
    def test_stress_requires_dual_release(self):
        with self.assertRaisesRegex(RuntimeError,'missing release'):authorize(self.root,'stress')
