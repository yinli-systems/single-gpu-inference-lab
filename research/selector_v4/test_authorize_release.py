import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from authorize_release import run
class AuthorizationTests(unittest.TestCase):
 def fixture(self):
  tmp=tempfile.TemporaryDirectory();root=Path(tmp.name);(root/'receipts').mkdir();(root/'runs').mkdir();paths=[]
  prov={'case_hash':'c','release_hash':'r','source_archive_sha256':'s','overlay_sha256':'o','measurement_revision':'4.0'}
  for gpu in ('gpu_4090','gpu_5090'):
   p=root/(gpu+'.json');p.write_text(json.dumps({'gpu':gpu,'stage':'canary','pass':True,'requirements':{'numerical_exact':True},'provenance':prov,'chosen_worst':1.0,'chosen_joint_lcb':1.0,'max_regret':1.0,'p95_regret':1.0,'cap_geomean':1.2,'cap_ci95':[1.1,1.3]}));paths.append(p)
  for i in range(4):(root/'receipts'/f'binary-audit-{i}.json').write_text(json.dumps({'kernel_symbol_isolation_compiled':True}))
  return tmp,root,paths
 def test_valid_canary_authorizes_release_only(self):
  tmp,root,paths=self.fixture();self.addCleanup(tmp.cleanup);out=root/'auth.json';run(SimpleNamespace(root=root,gpu_4090=paths[0],gpu_5090=paths[1],out=out));data=json.loads(out.read_text());self.assertTrue(data['release_qualification_authorized']);self.assertFalse(data['default_promotion'])
 def test_hold_or_missing_audit_blocks(self):
  tmp,root,paths=self.fixture();self.addCleanup(tmp.cleanup);bad=json.loads(paths[1].read_text());bad['pass']=False;paths[1].write_text(json.dumps(bad))
  with self.assertRaisesRegex(RuntimeError,'HOLD'):run(SimpleNamespace(root=root,gpu_4090=paths[0],gpu_5090=paths[1],out=root/'x'))
  bad['pass']=True;paths[1].write_text(json.dumps(bad));(root/'receipts/binary-audit-3.json').unlink()
  with self.assertRaisesRegex(RuntimeError,'binary audits'):run(SimpleNamespace(root=root,gpu_4090=paths[0],gpu_5090=paths[1],out=root/'x'))
 def test_consumed_release_blocks(self):
  tmp,root,paths=self.fixture();self.addCleanup(tmp.cleanup);(root/'runs/release-used').mkdir()
  with self.assertRaisesRegex(RuntimeError,'consumed'):run(SimpleNamespace(root=root,gpu_4090=paths[0],gpu_5090=paths[1],out=root/'x'))
if __name__=='__main__':unittest.main()
