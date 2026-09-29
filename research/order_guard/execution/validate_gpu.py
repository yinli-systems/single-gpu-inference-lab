"""Read-only qualification audit of real GPU receipts; never fabricates timings."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

MANIFEST="fb6c14b78cc03f8d02b7b16ff8efa9f382ce313b13e46781f5506a78be7cfa37"
MEASUREMENT_COMMIT="9e09364d13ce06f4dd391c42855e40f8fbbaa261"

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path): return json.loads(path.read_text())

def check_rows(rows,cases,dtypes,caches,policies,rep):
    ids={c["id"]:c for c in cases}
    expected=set(itertools.product(ids,dtypes,("auto","unsplit"),range(6),policies,("eager","graph"),caches))
    seen={}
    for r in rows:
        key=tuple(r[k] for k in ("case","dtype","split","block","policy","mode","cache"))
        require(key in expected and key not in seen,"unexpected/duplicate timing cell")
        require(type(r["block"]) is int and type(r["rep"]) is int and r["rep"]==rep,"wrong repeat/block")
        require(r["kind"]==ids[r["case"]]["kind"],"changed case category")
        require(type(r["calls"]) is int and r["calls"]==1,"not one actual invocation")
        for name in ("device_us","wall_us","setup_us"):
            require(type(r[name]) in (int,float) and math.isfinite(r[name]) and r[name]>0,"invalid "+name)
        seen[key]=r
    require(set(seen)==expected,"incomplete timing matrix")
    return seen

def validate(root,stage):
    source=root/"source"
    sys.path.insert(0,str(source/"research/order_guard"))
    sys.path.insert(0,str(source/"benchmarks/results/plan-order-mechanism"))
    from manifest import build
    from order_guard import Geometry,propose,digest
    from plan_contract import validate_descriptors,order_indices
    manifest=build()
    require(manifest["cases_sha256"]==MANIFEST and digest(manifest["cases"])==MANIFEST,"wrong frozen manifest")
    require((source/"SOURCE_COMMIT.txt").read_text().strip()==MEASUREMENT_COMMIT,"wrong measurement commit")
    for line in (source/"source.sha256").read_text().splitlines():
        digest_,name=line.split("  ",1)
        require(not Path(name).is_absolute() and ".." not in Path(name).parts,"unsafe source path")
        require(sha(source/name)==digest_,"changed source "+name)
    paths=sorted((root/"runs").glob(stage+"-*"))
    require(len(paths)==(2 if stage=="canary" else 24),"wrong run count")
    families={};slots=set();all_loaded=[];counts=Counter()
    required={"environment.json","measurements.json","qualification.json","plans.json","progress.json"}
    for path in paths:
        require(not (path/"failure.json").exists(),"failed run "+path.name)
        require((path/"completed_utc.txt").exists(),"launcher not complete "+path.name)
        complete=load(path/"complete.json");env=load(path/"environment.json")
        require(complete["complete"] is True and complete["GPU_executed"] is True,"GPU run incomplete")
        require(complete["source_manifest"]==env["manifest_sha256"]==MANIFEST,"manifest binding failed")
        require(env["stage"]==stage and env["flashinfer"]=="0.6.18","wrong experiment")
        require(env["full_model"] is False and env["serving_promotion"] is False,"scope changed")
        require(required.issubset(complete["files"]),"missing hashed files")
        for name,digest_ in complete["files"].items():
            require(Path(name).name==name and sha(path/name)==digest_,"evidence hash mismatch")
        require((path/"source_commit.txt").read_text().strip()==MEASUREMENT_COMMIT,"source receipt mismatch")
        wanted_sources={f.name:sha(f) for f in (source/"research/order_guard").glob("*.py")}
        require(env["source_sha256"]==wanted_sources,"unqualified measurement source")
        gpu=env["gpu"];require(any(g in gpu for g in ("RTX 4090","RTX 5090")),"unexpected device")
        family="4090" if "4090" in gpu else "5090"
        rep,shard,shards=env["rep"],env["shard"],env["shards"]
        require(type(rep) is int and type(shard) is int,"bad allocation coordinates")
        require((shards==1 and rep==shard==0) if stage=="canary" else (shards==4 and 0<=rep<3 and 0<=shard<4),"wrong sharding")
        slot=(family,rep,shard);require(slot not in slots,"duplicate family/repeat/shard")
        slots.add(slot);families[family]=gpu
        selected=[c for i,c in enumerate(manifest["cases"]) if i%shards==shard]
        if stage=="canary":selected=[manifest["cases"][0],next(c for c in manifest["cases"] if c["kind"]=="old-regression")]
        require(env["cases"]==selected,"changed shard geometry")
        dtypes=["float16"] if stage=="canary" else manifest["dtypes"]
        caches=["warm"] if stage=="canary" else manifest["cache_regimes"]
        policies=manifest["policies"]
        rows=load(path/"measurements.json");index=check_rows(rows,selected,dtypes,caches,policies,rep)
        require(complete["rows"]==complete["expected_rows"]==len(rows),"completion row count mismatch")
        cm={c["id"]:c for c in selected};base=set(itertools.product(cm,dtypes,("auto","unsplit")))
        plans=load(path/"plans.json");seen=set();hashes={}
        for plan in plans:
            key=tuple(plan[k] for k in ("case","dtype","split"))
            require(key in base and key not in seen,"bad plan coverage")
            seen.add(key);case=cm[key[0]];pp=plan["plan"];qs=case["query"]
            ls=[q+k for q,k in zip(qs,case["cached"])]
            require(pp["query"]==qs and pp["total_kv"]==ls,"plan geometry changed")
            validate_descriptors(pp["descriptors"],qs,ls,pp["info"],pp["chunk"],4,pp["output_indptr"],pp["merge_indptr"])
            ref=plan["FP32_reference"]
            require(ref["selected_fp32_vectors"]==sum(len({0,q//2,q-1})*32 for q in qs),"reference coverage gap")
            require(all(math.isfinite(ref[k]) and ref[k]>=0 for k in ("max_abs","lse_max_abs")),"invalid reference errors")
            g=Geometry(tuple(qs),tuple(ls),4,pp["info"]["cta_tile_q"],pp["chunk"],bool(pp["info"]["split_kv"]))
            for policy in policies:
                order=propose(g,pp["descriptors"],policy) if policy in ("causal_heavy","locality_packet8") else order_indices(pp["descriptors"],qs,ls,g.tile,g.chunk,g.split,g.group,policy)
                hashes[key+(policy,)]=digest([pp["descriptors"][i] for i in order])
            counts["selected_FP32_vectors"]+=ref["selected_fp32_vectors"]
        require(seen==base,"incomplete plans")
        quals=load(path/"qualification.json");seen=set()
        for q in quals:
            key=tuple(q[k] for k in ("case","dtype","split","policy"))
            require(key in hashes and key not in seen,"bad qualification coverage")
            require(q["descriptor_sha256"]==hashes[key],"qualified wrong descriptors")
            require(all(q[k] is True for k in ("exact_output","exact_lse","graph_exact")),"numerical failure")
            seen.add(key)
        require(seen==set(hashes),"qualification coverage gap")
        for r in rows:
            key=tuple(r[k] for k in ("case","dtype","split","policy"))
            require(r["descriptor_sha256"]==hashes[key],"timing/qualification mismatch")
        counts["runs"]+=1;counts["timing_rows"]+=len(rows);counts["qualifications"]+=len(quals);counts["plans"]+=len(plans)
        all_loaded.append(dict(name=path.name,env=env,rows=rows,index=index,plans=plans,complete_sha256=sha(path/"complete.json")))
    wanted=set(itertools.product(("4090","5090"),range(1 if stage=="canary" else 3),range(1 if stage=="canary" else 4)))
    require(slots==wanted,"missing run slots")
    return dict(complete=True,stage=stage,GPU_executed=True,counts=dict(counts),measurement_commit=MEASUREMENT_COMMIT,
        manifest_sha256=MANIFEST,run_receipts={r["name"]:r["complete_sha256"] for r in all_loaded},
        numerical_contracts_pass=True,performance_promotion=False,serving_promotion=False),all_loaded

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--stage",choices=("canary","formal"),required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();require(not a.out.exists(),"preserve audit receipt")
    report,_=validate(a.root,a.stage);a.out.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report),flush=True)
