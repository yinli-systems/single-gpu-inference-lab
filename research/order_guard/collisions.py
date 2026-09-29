"""Construct stronger logical-feature collisions. Never invent GPU timings."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from order_guard import Geometry, causal_pairs


def representation(g):
    cached=tuple(k-q for q,k in zip(g.query,g.total_kv))
    return dict(query_marginal=sorted(g.query),cached_marginal=sorted(cached),
                total_length_marginal=sorted(g.total_kv),
                logical_work=sum(q*(k-q)+q*(q+1)//2 for q,k in zip(g.query,g.total_kv)))


def family(a,h,b,group=4,tile=128,chunk=128):
    if any(type(x) is not int for x in (a,h,b)) or a<=0 or h<=0 or b<0:raise ValueError('invalid family')
    q=(a,a+h,a+2*h)
    cached_a=(b+h,b+2*h,b)
    cached_b=(b+2*h,b,b+h)
    ga=Geometry(q,tuple(x+y for x,y in zip(q,cached_a)),group,tile,chunk,True)
    gb=Geometry(q,tuple(x+y for x,y in zip(q,cached_b)),group,tile,chunk,True)
    if representation(ga)!=representation(gb):raise RuntimeError('feature collision violated')
    return ga,gb


def audit():
    witnesses=[];checked=0;unequal=0
    for a in (16,32,48,64,96,128):
        for h in (8,16,32,64,128):
            for b in (0,32,64,128,512,1024):
                for tile in (64,128):
                    for chunk in (128,256,512):
                        ga,gb=family(a,h,b,tile=tile,chunk=chunk);checked+=1
                        da,db=ga.descriptors(),gb.descriptors()
                        wa=sum(causal_pairs(ga,d) for d in da);wb=sum(causal_pairs(gb,d) for d in db)
                        assert wa==wb==4*representation(ga)['logical_work']
                        if len(da)!=len(db):
                            unequal+=1
                            if len(witnesses)<12:
                                witnesses.append(dict(a=asdict(ga),b=asdict(gb),common=representation(ga),
                                    descriptor_counts=[len(da),len(db)]))
    ga,gb=family(64,64,0)
    return dict(complete=True,GPU_used=False,measured_latency=False,source_planner_executed=False,
                checked_pairs=checked,different_descriptor_counts=unequal,
                minimal_example=dict(a=asdict(ga),b=asdict(gb),common=representation(ga),
                                     descriptor_counts=[len(ga.descriptors()),len(gb.descriptors())]),
                witnesses=witnesses,
                scope='Fixed abstract tile/chunk contract; equal full q/cached/total-length marginals AND logical work. Actual auto-planner and latency require GPU validation.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('preserve receipt')
    r=audit();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='witnesses'},indent=2))
