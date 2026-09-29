import copy
import itertools
import math
import unittest
from analyze import validate_rows,interval
from plan_contract import POLICIES

def fixture():
    c=dict(case='holdout-00',state='same',holdout=True)
    env=dict(case_manifest=[c],dtype=['float16'],blocks=1,rep=0)
    rows=[]
    for split,policy,mode in itertools.product(('auto','unsplit'),POLICIES,('eager','graph')):
        rows.append(dict(c,dtype='float16',split=split,block=0,policy=policy,mode=mode,rep=0,
            device_us=10.,wall_us=12.,apply_us=15.,plan_us=20.,trial_id='/'.join((split,policy,mode))))
    return rows,env

class AnalysisContracts(unittest.TestCase):
    def test_complete(self):
        rows,env=fixture();self.assertEqual(len(validate_rows(rows,env)),20)
    def test_missing(self):
        rows,env=fixture();rows.pop()
        with self.assertRaises(ValueError):validate_rows(rows,env)
    def test_duplicate(self):
        rows,env=fixture();rows.append(rows[0])
        with self.assertRaises(ValueError):validate_rows(rows,env)
    def test_zero(self):
        rows,env=fixture();rows[0]['device_us']=0
        with self.assertRaises(ValueError):validate_rows(rows,env)
    def test_nan(self):
        rows,env=fixture();rows[0]['device_us']=float('nan')
        with self.assertRaises(ValueError):validate_rows(rows,env)
    def test_boolean_timing(self):
        rows,env=fixture();rows[0]['device_us']=True
        with self.assertRaises(ValueError):validate_rows(rows,env)
    def test_wrong_holdout(self):
        rows,env=fixture();rows[0]['holdout']=False
        with self.assertRaises(ValueError):validate_rows(rows,env)
    def test_wrong_repeat(self):
        rows,env=fixture();rows[0]['rep']=1
        with self.assertRaises(ValueError):validate_rows(rows,env)
    def test_duplicate_trial(self):
        rows,env=fixture();rows[1]['trial_id']=rows[0]['trial_id']
        with self.assertRaises(ValueError):validate_rows(rows,env)
    def test_unknown_policy(self):
        rows,env=fixture();rows[0]['policy']='oracle-winner'
        with self.assertRaises(ValueError):validate_rows(rows,env)
    def test_incomplete_repeats(self):
        with self.assertRaises(ValueError):interval([[0.]*6]*2)
    def test_equal_latency(self):
        self.assertEqual(interval([[0.]*6]*3,log=True)['ratio'],1.)
    def test_direction(self):
        self.assertAlmostEqual(interval([[math.log(2)]*6]*3,log=True)['ratio'],2.)
    def test_reproducible(self):
        x=[[float(i+j) for i in range(6)] for j in range(3)]
        self.assertEqual(interval(x),interval(x))
if __name__=='__main__':unittest.main()
