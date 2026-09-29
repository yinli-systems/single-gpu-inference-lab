"""Complete frozen-suite analysis. Real archived timings only; no serving claim."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import numpy as np
from validate_gpu import validate,require

def bootstrap_weights():
    rng=np.random.default_rng(202609291451)
    weights=np.zeros((10000,18),dtype=np.float64)
    for draw in range(10000):
        for process in rng.integers(0,3,size=3):
            for block in rng.integers(0,6,size=6):weights[draw,process*6+block]+=1
    return weights/18

def interval(values,weights,confidence=.95,log=False):
    a=np.asarray(values,dtype=float)
    require(a.shape==(3,6) and np.isfinite(a).all(),"invalid process/block grid")
    draws=weights@a.reshape(18)
    tail=(1-confidence)/2
    low,high=np.quantile(draws,[tail,1-tail]);mean=float(a.mean())
    if log:return dict(ratio=math.exp(mean),CI=[math.exp(float(low)),math.exp(float(high))])
    return dict(mean=mean,CI=[float(low),float(high)])

def analyze(root,out):
    require(not out.exists(),"preserve analysis results")
    receipt,runs=validate(root,"formal");weights=bootstrap_weights()
    from order_guard import Geometry,propose,digest
    from manifest import build
    manifest=build();policies=manifest["policies"][1:]
    data=defaultdict(dict);envs=defaultdict(list);plan_records=defaultdict(list);changed={}
    for run in runs:
        gpu=run["env"]["gpu"];rep=run["env"]["rep"];envs[gpu].append(run["env"])
        for r in run["rows"]:
            key=(r["case"],r["dtype"],r["split"],r["mode"],r["cache"],rep,r["block"],r["policy"])
            require(key not in data[gpu],"duplicate aggregate row");data[gpu][key]=r
        for pr in run["plans"]:
            pp=pr["plan"];g=Geometry(tuple(pp["query"]),tuple(pp["total_kv"]),4,pp["info"]["cta_tile_q"],pp["chunk"],bool(pp["info"]["split_kv"]))
            order=propose(g,pp["descriptors"],"locality_packet8")
            changed[(gpu,pr["case"],pr["dtype"],pr["split"],rep)]=list(order)!=list(range(len(order)))
            plan_records[gpu].append(dict(case=pr["case"],kind=pr["kind"],dtype=pr["dtype"],requested_split=pr["split"],rep=rep,
              actual_split=pp["info"]["split_kv"],tile=pp["info"]["cta_tile_q"],chunk=pp["chunk"],descriptors=len(pp["descriptors"]),
              locality_changed=changed[(gpu,pr["case"],pr["dtype"],pr["split"],rep)]))
    report=dict(receipt,analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       scope="0.6.18 fixed-shape ragged FA2 run-only; candidate-generation and full lifecycle unmeasured",
       statistical_scope="10000 hierarchical process/block bootstrap draws conditional on named frozen entries; not familywise adjusted or hardware-population inference",
       primary_policy="locality_packet8",default_promotion=False,serving_promotion=False,GPUs={})
    md=["# Frozen order-guard GPU result","","All 57 named entries and 24 tasks retained. Run-only ratios exclude diagnostic metadata setup and candidate generation.",
        "Process/block intervals are conditional on the suite, not simultaneous no-regression guarantees.","",
        "|GPU|Set|Mode|Cache|Policy|Native/candidate [95% interval]|","|---|---|---|---|---|---|"]
    for gpu,index in sorted(data.items()):
        cells=[];aa_failures=[];grids={};kinds={c["id"]:c["kind"] for c in manifest["cases"]}
        keys=itertools.product(kinds,manifest["dtypes"],manifest["split_modes"],("eager","graph"),manifest["cache_regimes"])
        for key in keys:
            case,dtype,split,mode,cache=key
            base=np.array([[index[key+(rep,b,"identity")]["device_us"] for b in range(6)] for rep in range(3)])
            aa=np.array([[index[key+(rep,b,"identity_repeat")]["device_us"] for b in range(6)] for rep in range(3)])
            ast=interval(aa-base,weights,.9);eps=max(3.,.02*float(np.median(base)))
            ast.update(epsilon_us=eps,equivalent=ast["CI"][0]>=-eps and ast["CI"][1]<=eps)
            cell=dict(case=case,kind=kinds[case],dtype=dtype,split=split,mode=mode,cache=cache,
              native_median_us=float(np.median(base)),AA=ast,policies={},
              locality_changed_repeats=sum(changed[(gpu,case,dtype,split,r)] for r in range(3)))
            if not ast["equivalent"]:aa_failures.append({k:cell[k] for k in ("case","kind","dtype","split","mode","cache","AA")})
            for policy in policies:
                alt=np.array([[index[key+(rep,b,policy)]["device_us"] for b in range(6)] for rep in range(3)])
                lg=np.log(base/alt);st=interval(lg,weights,log=True)
                st["point_slowdown_gt1pct"]=st["ratio"]<1/1.01
                st["nominal_CI_slowdown_gt1pct"]=st["CI"][1]<1/1.01
                st["metadata_apply_median_us"]=statistics.median(index[key+(r,b,policy)]["setup_us"] for r in range(3) for b in range(6))
                wall=np.array([[math.log(index[key+(r,b,"identity")]["wall_us"]/index[key+(r,b,policy)]["wall_us"]) for b in range(6)] for r in range(3)])
                st["wall_ratio"]=math.exp(float(wall.mean()));cell["policies"][policy]=st;grids[(key,policy)]=lg
            cells.append(cell)
        summaries=[]
        for kind,mode,cache,policy in itertools.product(("fresh","constructed-collision","old-regression"),("eager","graph"),manifest["cache_regimes"],policies):
            selected=[c for c in cells if (c["kind"],c["mode"],c["cache"])==(kind,mode,cache)]
            lg=np.mean([grids[((c["case"],c["dtype"],c["split"],mode,cache),policy)] for c in selected],axis=0)
            st=interval(lg,weights,log=True)
            st.update(kind=kind,mode=mode,cache=cache,policy=policy,cells=len(selected),
              worst_ratio=min(c["policies"][policy]["ratio"] for c in selected),
              point_regressions=sum(c["policies"][policy]["point_slowdown_gt1pct"] for c in selected),
              nominal_CI_regressions=sum(c["policies"][policy]["nominal_CI_slowdown_gt1pct"] for c in selected),
              AA_failures=sum(not c["AA"]["equivalent"] for c in selected))
            summaries.append(st)
            if policy=="locality_packet8":md.append("|%s|%s|%s|%s|%s|%.6f [%.6f, %.6f]|"%(gpu,kind,mode,cache,policy,st["ratio"],*st["CI"]))
        bad=[c for c in cells if c["kind"]=="fresh" and c["policies"]["locality_packet8"]["nominal_CI_slowdown_gt1pct"]]
        report["GPUs"][gpu]=dict(summaries=summaries,cells=cells,AA_failures=aa_failures,
          primary_fresh_nominal_regressions=bad,plans=plan_records[gpu],environments=envs[gpu],
          locality_changed_plan_count=sum(p["locality_changed"] for p in plan_records[gpu]),
          plan_count=len(plan_records[gpu]),clean_control_gate=not aa_failures,
          no_confirmed_fresh_slowdown_gate=not bad)
        md += ["","%s: %d A/A equivalence failures; %d fresh primary cells with nominal upper ratio below 1/1.01. No default or serving promotion."%(gpu,len(aa_failures),len(bad)),""]
    out.mkdir(parents=True)
    (out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    (out/"RESULTS.md").write_text("\n".join(md).rstrip()+"\n")
    print("\n".join(md),flush=True)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();analyze(a.root,a.out)
