import ast,unittest
from pathlib import Path
class Probe(unittest.TestCase):
 def test_entry_guard_and_probe_sources(self):
  self.assertTrue(any(isinstance(x,ast.If) for x in ast.parse(Path(__file__).with_name('server_entry.py').read_text()).body));ast.parse(Path(__file__).with_name('probe_http.py').read_text());ast.parse(Path(__file__).with_name('probe_processor.py').read_text())
 def test_cuda_home_is_bound_before_expansion(self):
  lines=Path(__file__).with_name('probe.sbatch').read_text().splitlines()
  define=next(i for i,x in enumerate(lines) if x.startswith('export CUDA_HOME='))
  path=next(i for i,x in enumerate(lines) if x.startswith('export PATH='))
  library=next(i for i,x in enumerate(lines) if x.startswith('export LD_LIBRARY_PATH='))
  self.assertLess(define,path);self.assertLess(define,library)
if __name__=='__main__':unittest.main()
