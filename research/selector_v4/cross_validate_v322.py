"""Development-only leave-one-process-out validation on exposed v3.2.2 data."""
from __future__ import annotations
import argparse, collections, json, math
from pathlib import Path
import numpy as np
from .crossfit import RepeatEvidence, crossfit_cell
from .eligibility import evaluate_eligibility
from .identity import TacticIdentity
from .schema import DEFAULT_THRESHOLDS

def geo(values): return math.exp(sum(math.log(float(x)) for x in values)/len(values))
def run(root: Path, out: Path):
    manifest_path=root/"source/research/selector_v32/manifest.json"
    if not manifest_path.exists(): manifest_path=root/"source/manifest.json"
    manifest=json.loads(manifest_path.read_text());cases={c["id"]:c for c in manifest["cases"]}
    result={"scope":"EXPOSED DEVELOPMENT DATA ONLY","thresholds":DEFAULT_THRESHOLDS.to_dict(),"gpus":{}}
    for gpu in ("gpu_4090","gpu_5090"):
        cells=collections.defaultdict(lambda:collections.defaultdict(list));quals={};identities={}
        for shard in range(8):
            for rep in range(3):
                path=next((root/"runs").glob(f"release-{gpu}-s{shard}-r{rep}-paired-*"))
                rows=json.loads((path/"measurements.json").read_text());qq=json.loads((path/"qualification.json").read_text())
                for q in qq: quals[(q["case"],q["dtype"],q["layout"],q["split"])]=q
                grouped=collections.defaultdict(lambda:{"off":[],"cap":[]})
                for row in rows:
                    if row["comparison_group"]=="cap":grouped[(row["case"],row["dtype"],row["layout"],row["split"],row["execution_mode"],row["block"])][row["arm"]].append((row["position"],row["wall_us"]))
                for key,value in grouped.items():
                    off=[x[1] for x in sorted(value["off"])];cap=[x[1] for x in sorted(value["cap"])];basic=key[:4];q=quals[basic]
                    ratio=geo(off)/geo(cap);native=(off[0]/off[1],);candidate=(cap[0]/cap[1],)
                    cells[key[:5]][rep].append((ratio,native[0],candidate[0]))
        folds=[]
        for key,reps in cells.items():
            q=quals[key[:4]];case=cases[key[0]];payload=q["identity_payloads"]["cap"][key[4]]
            op=dict(payload["operation"]);op["plan_signature"]=op["plan_signature"][:-1]
            env=dict(payload["environment"]);env["backend_source_sha256"]=env.pop("backend_sha256");env.setdefault("max_smem_per_sm",102400);env.setdefault("max_smem_per_block_optin",101376)
            policy={"execution_mode":key[4],"timer":"deployment_wall","qualification_revision":"4.0.0"}
            identity=TacticIdentity(env,op,policy);eligibility=evaluate_eligibility(identity)
            evidence={rep:RepeatEvidence(tuple(x[0] for x in values),tuple(x[1] for x in values),tuple(x[2] for x in values)) for rep,values in reps.items()}
            for fold in crossfit_cell(identity,eligibility,evidence,exact_outputs=True):
                folds.append({"case":key[0],"dtype":key[1],"layout":key[2],"split":key[3],"execution":key[4],"selected":fold.receipt.tactic=="resource_cap","test_geomean":fold.held_out_geomean,"test_min":fold.held_out_minimum,"policy_geomean":fold.policy_geomean,"receipt":fold.receipt.to_dict()})
        selected=[x for x in folds if x["selected"]];policy=[x["policy_geomean"] for x in folds]
        result["gpus"][gpu]={"records":len(folds),"eligible":sum(x["receipt"]["eligible"] for x in folds),"selected":len(selected),
            "selected_coverage_of_eligible":len(selected)/sum(x["receipt"]["eligible"] for x in folds),
            "selected_geomean":geo([x["test_geomean"] for x in selected]),"selected_worst":min(x["test_geomean"] for x in selected),
            "selected_block_worst":min(x["test_min"] for x in selected),"policy_geomean":geo(policy),"policy_worst":min(policy)}
    out.mkdir(parents=True);(out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# V4 threshold cross-validation on exposed v3.2.2 data","","Development evidence only; not fresh validation.","",
        "| GPU | Selected / eligible folds | Held-out selected geomean | Held-out selected worst | Held-out block worst | Whole-policy geomean | Policy worst |","|---|---:|---:|---:|---:|---:|---:|"]
    for gpu,x in result["gpus"].items():lines.append(f"| {gpu} | {x['selected']} / {x['eligible']} | {x['selected_geomean']:.6f}x | {x['selected_worst']:.6f}x | {x['selected_block_worst']:.6f}x | {x['policy_geomean']:.6f}x | {x['policy_worst']:.6f}x |")
    (out/"RESULTS.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args();run(a.root,a.out)
