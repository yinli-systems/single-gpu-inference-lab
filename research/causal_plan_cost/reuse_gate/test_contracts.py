"""CPU tests are not GPU-performance evidence."""
import dataclasses
import unittest
from contracts import (PlanKey,PolicyCache,profitable_interval,scratch_bytes,
                       bounded_policy,fresh_corpus,corpus_hash,POLICIES)


class ReuseContracts(unittest.TestCase):
    def key(self,**changes):
        return dataclasses.replace(PlanKey((128,384),(1024,2048),32,8,128,environment='GPU-A/FI0618/CUDA13/hash'),**changes)
    def test_break_even_strict(self):self.assertEqual(profitable_interval(1.,.1),[11,None])
    def test_break_even_immediate(self):self.assertEqual(profitable_interval(-1.,.1),[1,None])
    def test_no_eventual_payback(self):self.assertIsNone(profitable_interval(1.,-.1))
    def test_finite_profitable_window(self):self.assertEqual(profitable_interval(-.5,-.1),[1,4])
    def test_equal_running_cost(self):
        self.assertEqual(profitable_interval(-1.,0.),[1,None])
        self.assertIsNone(profitable_interval(0.,0.))
        self.assertIsNone(profitable_interval(1.,0.))
    def test_invalid_cost(self):
        for x in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):profitable_interval(x,1.)
    def test_against_integer_enumeration(self):
        for a in (-1.,-.5,0.,.5,1.):
            for b in (-.2,-.1,0.,.1,.2):
                interval=profitable_interval(a,b)
                for r in range(1,25):
                    expected=a-r*b < -1e-12
                    actual=interval is not None and r>=interval[0] and (interval[1] is None or r<=interval[1])
                    self.assertEqual(expected,actual,(a,b,r,interval))
    def test_cache_hit(self):
        c=PolicyCache(2);calls=[]
        choose=lambda:(calls.append(1) or 'auto')
        c.resolve(self.key(),choose);c.resolve(self.key(),choose)
        self.assertEqual((len(calls),c.hits,c.misses),(1,1,1))
    def test_lru_eviction(self):
        c=PolicyCache(2)
        for e in ('A','B','C','A'):c.resolve(self.key(environment=e),lambda:'none')
        self.assertEqual((c.hits,c.misses,c.evictions,len(c.entries)),(0,4,2,2))
    def test_disabled_cache(self):
        c=PolicyCache(0)
        for _ in range(3):c.resolve(self.key(),lambda:'auto')
        self.assertEqual((c.hits,c.misses,len(c.entries)),(0,3,0))
    def test_validity_fields_invalidate(self):
        k=self.key()
        for patch in ({'q':(129,383)},{'k':(1025,2048)},{'hq':16},{'hkv':4},
                      {'sms':170},{'environment':'other-runtime'},{'scratch_ceiling':512*1024**2}):
            c=PolicyCache(2);c.resolve(k,lambda:'auto');c.resolve(dataclasses.replace(k,**patch),lambda:'none')
            self.assertEqual(c.misses,2)
    def test_order_is_not_discarded(self):self.assertNotEqual(self.key(),self.key(q=(384,128)))
    def test_unsupported_modes_fail_closed(self):
        for patch in ({'execution_mode':'graph'},{'head_dim':64},{'dtype':'bfloat16'},
                      {'layout':'HND'},{'causal':False},{'window_left':1024},
                      {'position_mode':'ROPE_LLAMA'},{'environment':''}):
            with self.assertRaises(ValueError):self.key(**patch)
    def test_bad_choice_not_cached(self):
        c=PolicyCache(2)
        with self.assertRaises(ValueError):c.resolve(self.key(),lambda:'unregistered')
        self.assertEqual(len(c.entries),0)
    def test_failed_choice_not_cached(self):
        c=PolicyCache(2)
        def fail():raise RuntimeError('failure')
        with self.assertRaises(RuntimeError):c.resolve(self.key(),fail)
        self.assertEqual(len(c.entries),0)
    def test_scratch_reference(self):
        from geometry import corpus,workspace_bytes
        for s in corpus():
            for hq,hkv in ((16,4),(32,8)):
                for sms in (128,170):
                    for p in POLICIES:
                        self.assertEqual(scratch_bytes(s.q,s.k,hq,hkv,sms,p),workspace_bytes(s,hq,hkv,sms,p))
    def test_resource_fallback(self):
        q,k=(255,769),(8192,7935)
        m=bounded_policy(q,k,32,8,128,'s512',128*1024**2)
        self.assertNotEqual(m,'s512')
        self.assertLessEqual(scratch_bytes(q,k,32,8,128,m),128*1024**2)
    def test_all_fresh_bounded(self):
        for s in fresh_corpus():
            for sms in (128,170):
                for p in POLICIES:
                    m=bounded_policy(tuple(s['q']),tuple(s['k']),32,8,sms,p,128*1024**2)
                    self.assertLessEqual(scratch_bytes(tuple(s['q']),tuple(s['k']),32,8,sms,m),128*1024**2)
    def test_corpus_identity(self):
        self.assertEqual(len(fresh_corpus()),54)
        self.assertEqual(len({s['family'] for s in fresh_corpus()}),9)
        self.assertEqual(corpus_hash(),corpus_hash())
    def test_fresh_geometries_not_old(self):
        from geometry import corpus
        old={(s.q,s.k) for s in corpus()}
        self.assertFalse(old & {(tuple(s['q']),tuple(s['k'])) for s in fresh_corpus()})


if __name__=='__main__':unittest.main()
