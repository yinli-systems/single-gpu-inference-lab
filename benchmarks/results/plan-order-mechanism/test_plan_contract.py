import unittest
import random
from plan_contract import *

class Contracts(unittest.TestCase):
    def test_abi(self):
        with self.assertRaises(ValueError): decode_info([0]*14)
    def test_boolean_abi(self):
        with self.assertRaises(ValueError): decode_info([True]*15)
    def test_dynamic_graph_rejected(self):
        v=[0]*15;v[0]=v[1]=v[3]=1;v[13]=1
        with self.assertRaises(ValueError):decode_info(v)
    def test_expected_merge(self):
        d,o,m=expected_descriptors([2,3],[5,10],4,8,4,True)
        self.assertEqual(o,[0,4,13]);self.assertEqual(m,[0,2,4,7,10,13])
        self.assertEqual(len(d),8)
    def test_length(self):
        with self.assertRaises(ValueError):expected_descriptors([3],[2],4,128,0,False)
    def test_empty(self):
        with self.assertRaises(ValueError):expected_descriptors([],[],4,128,1,False)
    def test_negative(self):
        with self.assertRaises(ValueError):expected_descriptors([-1],[5],4,128,1,False)
    def test_zero_chunk(self):
        with self.assertRaises(ValueError):expected_descriptors([1],[5],4,128,0,True)
    def test_unknown_policy(self):
        with self.assertRaises(ValueError):order_indices([],[],[],128,1,False,4,'best')
    def test_duplicate_order(self):
        with self.assertRaises(ValueError):validate_permutation([0,0],2)
    def test_missing_order(self):
        with self.assertRaises(ValueError):validate_permutation([0],2)
    def test_boolean_order(self):
        with self.assertRaises(ValueError):validate_permutation([False,True],2)
    def test_out_of_bounds_order(self):
        with self.assertRaises(ValueError):validate_permutation([0,2],2)
    def test_randomized_bijections(self):
        rng=random.Random(731)
        for _ in range(1000):
            n=rng.randint(1,16);q=[rng.randint(1,1024) for _ in range(n)]
            L=[qi+rng.randint(0,16000) for qi in q];tile=rng.choice([16,64,128]);chunk=rng.choice([128,512,2048,8192]);split=rng.choice([False,True])
            d,o,m=expected_descriptors(q,L,4,tile,chunk,split)
            info=dict(total_num_rows=sum(q),padded_batch_size=len(d),cta_tile_q=tile,split_kv=int(split))
            for policy in POLICIES:
                ids=order_indices(d,q,L,tile,chunk,split,4,policy);dd=[d[i] for i in ids]
                validate_descriptors(dd,q,L,info,chunk,4,o,m)
    def test_corrupt_descriptor(self):
        d,o,m=expected_descriptors([32],[1024],4,128,1,False)
        info=dict(total_num_rows=32,padded_batch_size=1,cta_tile_q=128,split_kv=0)
        with self.assertRaises(ValueError):validate_descriptors([(2,0,0)],[32],[1024],info,1,4,o,m)
    def test_wrong_output(self):
        d,o,m=expected_descriptors([32],[1024],4,128,1,False)
        info=dict(total_num_rows=32,padded_batch_size=1,cta_tile_q=128,split_kv=0)
        with self.assertRaises(ValueError):validate_descriptors(d,[32],[1024],info,1,4,[0,31],m)
    def test_wrong_merge(self):
        d,o,m=expected_descriptors([32],[1024],4,128,512,True)
        info=dict(total_num_rows=32,padded_batch_size=2,cta_tile_q=128,split_kv=1)
        with self.assertRaises(ValueError):validate_descriptors(d,[32],[1024],info,512,4,o,[0]*33)
    def test_wrong_capacity(self):
        d,o,m=expected_descriptors([32],[1024],4,128,1,False)
        info=dict(total_num_rows=32,padded_batch_size=2,cta_tile_q=128,split_kv=0)
        with self.assertRaises(ValueError):validate_descriptors(d,[32],[1024],info,1,4,o,m)
if __name__=='__main__':unittest.main()
