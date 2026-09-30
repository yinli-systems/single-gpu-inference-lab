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
 def test_position_probe_does_not_install_tensor_observer(self):
  entry=Path(__file__).with_name('probe_server_entry.py').read_text()
  self.assertNotIn('from observer import install',entry)
  self.assertIn('sglang.launch_server',entry)
  http=Path(__file__).with_name('probe_http.py').read_text()
  self.assertIn("probe_server_entry.py",http)
 def test_cccl_and_private_jit_cache_are_bound(self):
  text=Path(__file__).with_name('probe.sbatch').read_text()
  for needle in ['flashinfer/data/cccl','NVCC_PREPEND_FLAGS','SGLANG_JIT_CACHE_DIR','SGLANG_CACHE_DIR','SGLANG_CUTE_AOT_CACHE_DIR']:
   self.assertIn(needle,text)
if __name__=='__main__':unittest.main()
