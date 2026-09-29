"""CPU contract tests only; synthetic receipts are not performance evidence."""
import copy
import itertools
import unittest
from validate_gpu import check_rows
class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.cases=[dict(id="fixture",kind="test")]
        self.policies=["identity","identity_repeat"]
        self.rows=[dict(case="fixture",kind="test",dtype="float16",split=s,block=b,policy=p,mode=m,cache="warm",rep=0,calls=1,device_us=1.,wall_us=2.,setup_us=3.) for s,b,p,m in itertools.product(("auto","unsplit"),range(6),self.policies,("eager","graph"))]
    def run_check(self,rows):return check_rows(rows,self.cases,["float16"],["warm"],self.policies,0)
    def test_complete(self):self.assertEqual(len(self.run_check(self.rows)),48)
    def test_missing(self):
        with self.assertRaises(ValueError):self.run_check(self.rows[:-1])
    def test_duplicate(self):
        with self.assertRaises(ValueError):self.run_check(self.rows+self.rows[:1])
    def test_invalid_fields(self):
        for field,value in [("device_us",float("nan")),("wall_us",float("inf")),("setup_us",0.),("rep",1),("block",True),("calls",2),("kind","fresh")]:
            rows=copy.deepcopy(self.rows);rows[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.run_check(rows)
if __name__=="__main__":unittest.main()
