import ast,unittest
from pathlib import Path
class Probe(unittest.TestCase):
 def test_entry_guard_and_probe_sources(self):
  self.assertTrue(any(isinstance(x,ast.If) for x in ast.parse(Path(__file__).with_name('server_entry.py').read_text()).body));ast.parse(Path(__file__).with_name('probe_http.py').read_text());ast.parse(Path(__file__).with_name('probe_processor.py').read_text())
if __name__=='__main__':unittest.main()
