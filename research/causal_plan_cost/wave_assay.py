"""Prospective hardware-threshold diagnostic; not training/test-model data."""
import hashlib
import json
import sys
from pathlib import Path
import measure
from geometry import Shape

QUERY_BUDGETS=(960,1024,1088,1280,1344,1408,2656,2720,2784)

def shapes():
    result=[]
    for total in QUERY_BUDGETS:
        b=(total//64)*32
        a=b//2-1
        depths=(8192,8192-(total-a-b))
        for state,cut in (('A',a),('B',b)):
            result.append(Shape(f'wave-Q{total}-{state}',f'wave-Q{total}',
                                'diagnostic',(cut,total-cut),depths))
        assert result[-1].work==result[-2].work
    return result

if __name__=='__main__':
    measure.corpus=shapes
    measure.corpus_json=lambda:json.dumps([s.record() for s in shapes()],sort_keys=True)
    measure.POLICIES=('auto','none','s512','s1024')
    measure.main()
