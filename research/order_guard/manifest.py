"""Deterministic unmeasured GPU suite, frozen before new GPU observations."""
import argparse
import json
from pathlib import Path
import random
from collisions import family
from order_guard import digest


def build():
    rows=[];rng=random.Random(20260929134631)
    for j,n in enumerate((1,2,3,4,5,8,12,16)*3):
        q=[rng.choice((1,31,32,33,63,64,65,127,128,129,256,512,1024)) for _ in range(n)]
        k=[rng.choice((0,63,128,512,2048,8192,16384)) for _ in range(n)]
        for state,qs in [('same',sorted(q)),('opposite',sorted(q,reverse=True))]:
            rows.append(dict(id=f'fresh-{j:02d}-{state}',query=qs,cached=sorted(k),exposed=False,kind='fresh'))
    # Explicitly exposed regression sentinel: never relabel as unseen.
    rows.append(dict(id='regression-5090-42pct',query=[672,176,96],cached=[8192,16896,24064],
                     exposed=True,kind='old-regression'))
    # These constructions fix three complete marginal distributions and logical W.
    for j,(a,h,b) in enumerate(((64,64,0),(32,32,0),(48,64,128),(96,128,1024))):
        for side,g in zip(('a','b'),family(a,h,b)):
            rows.append(dict(id=f'collision-{j}-{side}',query=list(g.query),
                cached=[k-q for q,k in zip(g.query,g.total_kv)],exposed=False,kind='constructed-collision',
                abstract_tile=g.tile,abstract_chunk=g.chunk))
    if len({r['id'] for r in rows})!=len(rows):raise RuntimeError('duplicate case ID')
    return dict(schema=1,seed=20260929134631,GPU_observations=0,cases=rows,cases_sha256=digest(rows),
                gpu_families=['RTX 4090','RTX 5090'],dtypes=['float16','bfloat16'],
                split_modes=['auto','unsplit'],cache_regimes=['warm','flush128MiB'],
                policies=['identity','identity_repeat','heavy_first','request_reverse','causal_heavy','locality_packet8'],
                blocks=6,notes='Frozen fresh-shape suite plus explicitly exposed sentinel. Auto planner may not select the abstract collision tile/chunk.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('preserve frozen manifest')
    r=build();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(cases=len(r['cases']),hash=r['cases_sha256'],GPU_observations=0)))
