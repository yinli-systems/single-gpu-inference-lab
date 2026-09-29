"""CPU-only fabricated fixtures validate auditing, never GPU performance."""
import copy,itertools,tempfile,unittest
from pathlib import Path
from campaign import CONFIGS,ARMS,MODES,states,geometry
from analyze import validate_rows,spearman,interval,analyze

def fixture():
 rows=[]
 for d,n in itertools.product(('float16','bfloat16'),CONFIGS):
  for b in range(12):
   for label,p in states(n):
    for a,m in itertools.product(ARMS,MODES):
     rows.append(dict(geometry(n,label,p),dtype=d,rep=0,block=b,arm=a,mode=m,device_us=1.,wall_us=2.,plan_us=3.,timed_calls=12 if m=='eager' else 48,trial_id=f'0/{d}/{n}/{b}/{label}/{a}/{m}'))
 return rows
class AuditContracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.rows=fixture()
 def test_complete_matrix(self):validate_rows(self.rows)
 def test_missing_row(self):
  with self.assertRaises(ValueError):validate_rows(self.rows[:-1])
 def test_duplicate_row(self):
  with self.assertRaises(ValueError):validate_rows(self.rows+[self.rows[0]])
 def test_trial_mislabel(self):
  x=copy.deepcopy(self.rows);x[0]['trial_id']='wrong'
  with self.assertRaises(ValueError):validate_rows(x)
 def test_geometry_tamper(self):
  x=copy.deepcopy(self.rows);x[0]['cached'][0]+=1
  with self.assertRaises(ValueError):validate_rows(x)
 def test_changed_inner_count(self):
  x=copy.deepcopy(self.rows);x[0]['timed_calls']+=1
  with self.assertRaises(ValueError):validate_rows(x)
 def test_no_processes_no_claim(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):analyze(Path(d),Path(d)/'out')
   self.assertFalse((Path(d)/'out').exists())
 def test_zero_effect_interval(self):self.assertEqual(interval([[0.]*12 for _ in range(3)])['CI'],[0.,0.])
 def test_missing_repeat_interval(self):
  with self.assertRaises(ValueError):interval([[0.]*12 for _ in range(2)])
 def test_tie_aware_rank(self):
  self.assertEqual(spearman([1,2,2,4],[1,2,2,4]),1.)
  self.assertEqual(spearman([1,2,2,4],[-1,-2,-2,-4]),-1.)
 def test_cached_bootstrap_matches_literal_protocol(self):
  import random,statistics
  from analyze import quantile
  values=[[((i+1)*(j-5))/7 for j in range(12)] for i in range(3)]
  rng=random.Random(20260929);draws=[]
  for _ in range(5000):
   sample=[values[rng.randrange(3)] for _ in range(3)]
   draws.append(statistics.mean(statistics.mean(v[rng.randrange(12)] for _ in range(12)) for v in sample))
  result=interval(values)
  self.assertAlmostEqual(result['CI'][0],quantile(draws,.025),places=11)
  self.assertAlmostEqual(result['CI'][1],quantile(draws,.975),places=11)
 def test_failure_marker_not_ignored(self):
  with tempfile.TemporaryDirectory() as d:
   for i in range(6):
    p=Path(d)/('test-'+str(i));p.mkdir();(p/'failure.json').write_text('{}')
   with self.assertRaisesRegex(ValueError,'failed job'):analyze(Path(d),Path(d)/'out')

if __name__=='__main__':unittest.main()
