import ast,unittest
from pathlib import Path
class Entry(unittest.TestCase):
 def test_main_guard(self):
  tree=ast.parse(Path(__file__).with_name('server_entry.py').read_text());self.assertTrue(any(isinstance(x,ast.If) for x in tree.body))
if __name__=='__main__':unittest.main()
