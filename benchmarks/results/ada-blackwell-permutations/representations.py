"""Exact information audit; feature equality is not a runtime accuracy guarantee."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from campaign import CONFIGS,states,geometry

def reconstruct_work(query_squares:int,cached_squares:int,total_kv_squares:int,query_sum:int):
 values=(query_squares,cached_squares,total_kv_squares,query_sum)
 if any(type(x) is not int or x<0 for x in values): raise ValueError('nonnegative integer moments required')
 twice_c=total_kv_squares-query_squares-cached_squares
 if twice_c<0 or twice_c%2 or (query_squares+query_sum)%2: raise ValueError('inconsistent integer moments')
 return twice_c//2+(query_squares+query_sum)//2

def features(g):
 q,k=g['query'],g['cached'];l=[a+b for a,b in zip(q,k)]
 sums=(len(q),sum(q),sum(k));squares=(sum(x*x for x in q),sum(x*x for x in k),sum(x*x for x in l))
 return dict(sums=sums,query_cached_marginals=(tuple(sorted(q)),tuple(sorted(k))),
             augmented_marginal_moments=(*sums,*squares),analytical_work=(g['W'],),
             paired_set=tuple(sorted(zip(q,k))),ordered_pairs=tuple(zip(q,k)))

def audit():
 output=[]
 for name in CONFIGS:
  rows=[geometry(name,l,p) for l,p in states(name)]
  for g in rows:
   q,k=g['query'],g['cached'];l=[a+b for a,b in zip(q,k)]
   computed=reconstruct_work(sum(x*x for x in q),sum(x*x for x in k),sum(x*x for x in l),sum(q))
   if computed!=g['W']:raise AssertionError('moment identity broken')
  maps={g['state']:features(g) for g in rows}
  for fname in maps['same']:
   classes={}
   for g in rows:classes.setdefault(maps[g['state']][fname],[]).append(g)
   output.append(dict(config=name,feature=fname,distinct_feature_vectors=len(classes),
      state_count=len(rows),same_opposite_alias=maps['same'][fname]==maps['opposite'][fname],
      row_order_invariant=maps['same'][fname]==maps['order-null'][fname],
      distinct_work_alias_classes=sum(len({g['W'] for g in members})>1 for members in classes.values()),
      equal_work_pair_alias=(maps['eqC-a'][fname]==maps['eqC-b'][fname]) if 'eqC-a' in maps else None))
 return dict(kind='exact arithmetic information audit, no measured latency',
   identity='2*C = sum((q+k)^2) - sum(q^2) - sum(k^2)',
   warning='The full paired set is not uniquely necessary for analytical W. Feature equality for W does not imply equal GPU latency.',
   rows=output)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('do not overwrite prior audit')
 a.out.write_text(json.dumps(audit(),indent=2)+'\n')
 print(a.out)
