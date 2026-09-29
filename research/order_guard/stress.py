"""Large deterministic CPU property campaign. No GPU/performance evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from order_guard import Geometry, PROPOSALS, causal_pairs, propose, transitions


def run(count=10000):
    rng = random.Random(20260929132921)
    total_descriptors = 0; brute_checks = 0; changed = Counter()
    out_hash = hashlib.sha256()
    for _ in range(count):
        n = rng.randint(1, 8)
        qs = tuple(rng.randint(1, 129) for _ in range(n))
        ls = tuple(q+rng.randint(0, 2048) for q in qs)
        g = Geometry(qs, ls, rng.choice((1,2,3,4,8)), rng.choice((32,64,128)),
                     rng.choice((128,512,2048)), rng.choice((False,True)))
        desc = g.descriptors();total_descriptors += len(desc)
        score = sum(causal_pairs(g,d) for d in desc)
        assert score == g.group*sum(q*(k-q)+q*(q+1)//2 for q,k in zip(qs,ls))
        for d in rng.sample(desc, min(5,len(desc))):
            r,t,s=d;lo=s*g.chunk if g.split else 0;hi=min(ls[r],(s+1)*g.chunk) if g.split else ls[r]
            brute=sum(max(0,min(hi,ls[r]-qs[r]+u//g.group+1)-lo)
                      for u in range(t*g.tile,min((t+1)*g.tile,qs[r]*g.group)))
            assert causal_pairs(g,d)==brute;brute_checks+=1
        for policy in PROPOSALS:
            order=propose(g,desc,policy)
            assert sorted(order)==list(range(len(desc)))
            assert sum(causal_pairs(g,desc[i]) for i in order)==score
            if tuple(order)!=tuple(range(len(desc))):changed[policy]+=1
            if policy=='locality_packet8':
                assert tuple(order[:8])==tuple(range(min(8,len(desc))))
                assert transitions(desc,order)<=transitions(desc,range(len(desc)))
                assert all(abs(i-j)<=31 for i,j in enumerate(order))
            out_hash.update(json.dumps(order,separators=(',',':')).encode())
    return dict(complete=True,GPU_used=False,performance_claim=False,geometries=count,
                descriptor_checks=total_descriptors,independent_bruteforce_checks=brute_checks,
                proposal_bijections=count*len(PROPOSALS),changed_geometries=dict(changed),
                deterministic_order_hash=out_hash.hexdigest(),seed=20260929132921)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--count',type=int,default=10000)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('preserve previous stress receipt')
    if not 1<=a.count<=10000:raise ValueError('bounded count required')
    result=run(a.count);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
