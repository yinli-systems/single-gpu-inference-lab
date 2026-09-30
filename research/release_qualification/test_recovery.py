import importlib.util,sys,tempfile,types,unittest
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
if __name__=='__main__':unittest.main()
