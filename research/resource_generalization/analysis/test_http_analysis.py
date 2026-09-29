import hashlib,json,tempfile,unittest
from pathlib import Path
from analyze_http import read_mode
class AnalysisContracts(unittest.TestCase):
 def fixture(self,root,mutate=None):
  reqs=[dict(id=f'r{i}',tokens=[1,2],token_times=[.1,.2],ttft=.1,tpot=.1,latency=.21) for i in range(4)]
  data=dict(errors=[],requests=reqs,output_tokens=8,elapsed=1.,output_tokens_per_second=8.)
  if mutate:mutate(data)
  f=root/'smoke-b0.json';f.write_text(json.dumps(data));h=hashlib.sha256(f.read_bytes()).hexdigest()
  (root/'complete.json').write_text(json.dumps(dict(complete=True,stage='smoke',HTTP=True,full_model=True,files={f.name:h})))
 def test_valid(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);self.fixture(p);self.assertEqual(set(read_mode(p,'smoke')[1]),{'smoke-b0.json'})
 def test_duplicate(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);self.fixture(p,lambda x:x['requests'][1].update(id='r0'))
   with self.assertRaises(ValueError):read_mode(p,'smoke')
 def test_nonfinite(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);self.fixture(p,lambda x:x['requests'][0].update(ttft=float('nan')))
   with self.assertRaises(ValueError):read_mode(p,'smoke')
 def test_wrong_numerator(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);self.fixture(p,lambda x:x.update(output_tokens=100))
   with self.assertRaises(ValueError):read_mode(p,'smoke')
 def test_checksum(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);self.fixture(p);(p/'smoke-b0.json').write_text('{}')
   with self.assertRaises(ValueError):read_mode(p,'smoke')
 def test_missing_completed_workload(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);self.fixture(p);(p/'smoke-b0.json').rename(p/'wrong-b0.json')
   with self.assertRaises((ValueError,FileNotFoundError)):read_mode(p,'smoke')
 def test_failed_server(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);self.fixture(p);(p/'failure.json').write_text('{}')
   with self.assertRaises(ValueError):read_mode(p,'smoke')
if __name__=='__main__':unittest.main()
