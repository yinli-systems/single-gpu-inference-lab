import hashlib,importlib.util,json,sys,tempfile,types,unittest
sys.modules.setdefault('numpy',types.SimpleNamespace())
from pathlib import Path
spec=importlib.util.spec_from_file_location('release_analysis',Path(__file__).with_name('analyze.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class RecoveryTests(unittest.TestCase):
 def test_safe_relative(self):
  with tempfile.TemporaryDirectory() as d:self.assertEqual(m.safe_rel(Path(d),'runs/x'),Path(d)/'runs/x')
 def test_reject_absolute(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):m.safe_rel(Path(d),'/tmp/x')
 def test_reject_parent(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):m.safe_rel(Path(d),'../x')
 def test_schema2_retains_partial_complete_and_unscored(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'receipts').mkdir();runs=root/'runs';runs.mkdir()
   def env(path,mode):
    path.mkdir();(path/'environment.json').write_text(json.dumps({'shard':3,'rep':2,'mode':mode,'hardware':{'out':'gpu'}}))
   def complete(path):
    (path/'launcher-complete.txt').write_text('x');(path/'complete.json').write_text(json.dumps({'complete':True,'files':{}}))
   oldp=runs/'old-partial';env(oldp,'pristine');(oldp/'progress.json').write_text(json.dumps({'rows':1}))
   oldc=runs/'old-cap';env(oldc,'cap');complete(oldc)
   attempt=runs/'attempt-pristine';env(attempt,'pristine');complete(attempt)
   newp=runs/'new-pristine';env(newp,'pristine');complete(newp)
   newc=runs/'new-cap';env(newc,'cap');complete(newc)
   prior=root/'receipts'/'recovery-map.json';prior.write_text(json.dumps({'schema':1}))
   sh=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
   final={'schema':2,'stage':'release','gpu':'gpu_4090','measurement_source_commit':'x','prior_map_sha256':sh(prior),'entries':[
    {'superseded':'runs/old-partial','replacement':'runs/new-pristine','key':{'shard':3,'rep':2,'mode':'pristine'},'superseded_kind':'partial','reason':'timeout','old_job':1,'old_state':'TIMEOUT','old_environment_sha256':sh(oldp/'environment.json'),'old_progress_sha256':sh(oldp/'progress.json'),'replacement_complete_sha256':sh(newp/'complete.json')},
    {'superseded':'runs/old-cap','replacement':'runs/new-cap','key':{'shard':3,'rep':2,'mode':'cap'},'superseded_kind':'complete','reason':'paired allocation replacement','old_job':1,'old_complete_sha256':sh(oldc/'complete.json'),'replacement_complete_sha256':sh(newc/'complete.json')}],
    'unscored_attempts':[{'path':'runs/attempt-pristine','job':2,'reason':'hardware mismatch','complete_sha256':sh(attempt/'complete.json')}]}
   (root/'receipts'/'recovery-map-final.json').write_text(json.dumps(final))
   skipped,info,replacements=m.recovery_map(root,'release','gpu_4090')
   self.assertEqual(len(skipped),3);self.assertEqual(len(info['retained_incomplete_runs']),1)
   self.assertEqual(len(info['superseded_complete_runs']),1);self.assertEqual(len(info['unscored_recovery_attempts']),1)
   self.assertEqual(replacements,{'runs/new-pristine','runs/new-cap'})

if __name__=='__main__':unittest.main()
