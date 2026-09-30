from __future__ import annotations
import argparse, csv, hashlib, json, math, tempfile
from pathlib import Path
from typing import Any
import numpy as np
from .cache import SafeTacticCache
from .crossfit import RepeatEvidence, crossfit_cell, simultaneous_min_lcb
from .eligibility import EligibilityDecision
from .evidence_v4 import EXECUTIONS, validate_run
from .identity import TacticIdentity
from .manifest_v4 import load
from .schema import QUALIFICATION_REVISION, TACTIC_CAP


def need(value: bool, message: str) -> None:
    if not value: raise RuntimeError(message)

def geo(values):
    return math.exp(sum(math.log(float(x)) for x in values) / len(values))

def ratio_text(value):
    return "n/a" if value is None else f"{value:.6f}x"

def ci90(values, seed):
    logs=np.log(np.asarray(values,dtype=np.float64));rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(logs),size=(10000,len(logs)));return [float(x) for x in np.exp(np.quantile(logs[idx].mean(1),(.05,.95)))]

def block_evidence(rows, key, execution):
    selected=[r for r in rows if (r["case"],r["dtype"],r["layout"],r["split"])==key and r["execution_mode"]==execution]
    treatment=[];native=[];cap=[]
    blocks=sorted({r["block"] for r in selected})
    for block in blocks:
        rr=sorted([r for r in selected if r["block"]==block],key=lambda r:r["position"])
        off=[r["wall_us"] for r in rr if r["arm"]=="off"];candidate=[r["wall_us"] for r in rr if r["arm"]=="cap"]
        need(len(off)==len(candidate)==2,"block multiplicity")
        treatment.append(geo(off)/geo(candidate));native.append(off[0]/off[1]);cap.append(candidate[0]/candidate[1])
    return RepeatEvidence(tuple(treatment),tuple(native),tuple(cap))

def absolute_native_block_evidence(pristine_rows, paired_rows, key, execution):
    """Compare official pristine against candidate-native without training the tactic.

    The two arms live in independent processes.  Process order is rotated by the
    launcher; this receipt is qualification-only and never enters cap selection.
    """
    pristine=[r for r in pristine_rows if (r["case"],r["dtype"],r["layout"],r["split"])==key and r["execution_mode"]==execution]
    paired=[r for r in paired_rows if (r["case"],r["dtype"],r["layout"],r["split"])==key and r["execution_mode"]==execution]
    blocks=sorted({r["block"] for r in pristine}|{r["block"] for r in paired})
    ratios=[];pristine_controls=[];off_controls=[]
    for block in blocks:
        pp=sorted([r for r in pristine if r["block"]==block],key=lambda r:r["position"])
        qq=sorted([r for r in paired if r["block"]==block],key=lambda r:r["position"])
        need(len(pp)==len(qq)==4,"absolute native block multiplicity")
        need(all(r["arm"]=="pristine" for r in pp),"absolute pristine arm")
        off=[r for r in qq if r["arm"]=="off"]
        need(len(off)==2,"absolute off multiplicity")
        a=[r["wall_us"] for r in pp if r["role"]=="a"]
        b=[r["wall_us"] for r in pp if r["role"]=="b"]
        need(len(a)==len(b)==2,"pristine position-control multiplicity")
        ratios.append(geo([r["wall_us"] for r in pp])/geo([r["wall_us"] for r in off]))
        pristine_controls.append(geo(a)/geo(b))
        off_controls.append(off[0]["wall_us"]/off[1]["wall_us"])
    return {"ratios":tuple(ratios),"pristine_controls":tuple(pristine_controls),"off_controls":tuple(off_controls)}


def hierarchical_ratio_summary(repeats, *, seed, draws=20000):
    """Three-process / within-process block bootstrap for an absolute ratio."""
    if sorted(repeats)!=[0,1,2]:raise ValueError("absolute ratio requires repeats 0,1,2")
    arrays=[np.asarray(repeats[i],dtype=np.float64) for i in range(3)]
    if any(x.ndim!=1 or len(x)==0 or not np.isfinite(x).all() or np.any(x<=0) for x in arrays):
        raise ValueError("invalid absolute-ratio evidence")
    if len({len(x) for x in arrays})!=1:raise ValueError("absolute-ratio block mismatch")
    logs=np.stack([np.log(x) for x in arrays]);rng=np.random.default_rng(seed)
    samples=np.empty(draws,dtype=np.float64)
    for d in range(draws):
        proc=rng.integers(0,3,3);values=[]
        for i in proc:
            idx=rng.integers(0,logs.shape[1],logs.shape[1]);values.append(logs[i,idx])
        samples[d]=np.concatenate(values).mean()
    point=float(np.exp(logs.mean()));ci=[float(x) for x in np.exp(np.quantile(samples,(.025,.975)))]
    return {"ratio":point,"CI95":ci,"block_worst":float(np.exp(logs.min())),"draws":samples}


def simultaneous_ratio_lcb(summaries, quantile=.025):
    if not summaries:return 1.0
    matrix=np.stack([x["draws"] for x in summaries])
    return float(np.exp(np.quantile(matrix.min(axis=0),quantile)))


def telemetry_receipt(path: Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        rows=list(csv.DictReader(stream))
    clocks=[];active=0
    for row in rows:
        raw=next((v for k,v in row.items() if k and 'clocks' in k and ('sm' in k or 'graphics' in k)),None)
        util=next((v for k,v in row.items() if k and 'utilization.gpu' in k),None)
        try:
            utilization=float(str(util).split()[0]);clock=float(str(raw).split()[0])
        except (TypeError,ValueError,AttributeError):
            continue
        if utilization>=90 and clock>0:
            active+=1;clocks.append(clock)
    if len(clocks)<10:
        return {'stable':False,'active_samples':active,'reason':'insufficient-active-telemetry'}
    ordered=sorted(clocks);lo=ordered[max(0,int(.05*len(ordered))-1)];hi=ordered[min(len(ordered)-1,int(.95*len(ordered)))]
    return {'stable':hi/lo<=1.05,'active_samples':active,'p05_mhz':lo,'p95_mhz':hi,'ratio':hi/lo,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}

def validate_binary_audit(path: Path) -> dict[str, Any]:
    value=json.loads(Path(path).read_text())
    need(value.get('kernel_symbol_isolation_compiled') is True,'compiled symbol isolation')
    need(value.get('same_module_pairs')=={'ragged':True,'paged':True},'native/resource co-residence')
    need(bool(value.get('binaries')),'missing compiled tactic binaries')
    return {'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'binaries':[{'sha256':x['sha256'],'bytes':x['bytes']} for x in value['binaries']], 'same_module_pairs':value['same_module_pairs'], 'native_sass_identity':value.get('native_sass_identity',False)}

def run(args):
    measurement_manifest_path=args.root/"source/research/selector_v4/manifest.json"
    manifest=load(measurement_manifest_path);cases=[c for c in manifest["cases"] if c["family"]==args.stage];lookup={};hardware={};source=set();overlay=set();binary_audits={};telemetry={}
    for shard in range(args.shards):
        subset=[c for i,c in enumerate(cases) if i%args.shards==shard]
        need(bool(subset),"empty shard")
        for rep in range(3):
            for mode in ("pristine","paired"):
                paths=list((args.root/"runs").glob(f"{args.stage}-{args.gpu}-s{shard}-r{rep}-{mode}-*"));need(len(paths)==1,f"run discovery {shard} {rep} {mode}")
                lookup[shard,rep,mode]=validate_run(paths[0],mode=mode,stage=args.stage,rep=rep,shard=shard,shards=args.shards,manifest=manifest)
        envs=[lookup[shard,rep,mode]["environment"] for rep in range(3) for mode in ("pristine","paired")]
        uuids={x["gpu_uuid"] for x in envs};drivers={x["driver"] for x in envs};need(len(uuids)==len(drivers)==1,"shard hardware drift")
        hardware[str(shard)]={"uuid":next(iter(uuids)),"driver":next(iter(drivers)),"gpu_name":envs[0]["gpu_name"]}
        source|={x["source_archive_sha256"] for x in envs};overlay|={x["official_overlay_sha256"] for x in envs}
        paired_path=Path(lookup[shard,0,"paired"]["path"]);job_id=paired_path.name.rsplit("-",1)[-1]
        binary_audits[str(shard)]=validate_binary_audit(args.root/f"receipts/binary-audit-{job_id}.json")
        telemetry[str(shard)]=telemetry_receipt(args.root/f"logs/telemetry-{job_id}.csv")
    need(len(source)==len(overlay)==1,"campaign source drift")
    all_folds=[];cell_receipts=[];native_overlay_records=[];native_overlay_summaries=[];numerics=0;cache_ok=True;native_after_cap=True
    for shard in range(args.shards):
        base=lookup[shard,0,"paired"]
        for key,q0 in base["qualifications"].items():
            for rep in range(3):
                pristine=lookup[shard,rep,"pristine"]["qualifications"][key];paired=lookup[shard,rep,"paired"]["qualifications"][key]
                need(pristine["reference_sha256"]==paired["reference_sha256"],"reference mismatch")
                native_after_cap &= paired.get("native_after_cap_exact") is True
                for arm in ("off","cap"):
                    need(paired["arms"][arm]["out_sha256"]==pristine["arms"]["pristine"]["out_sha256"],"output hash mismatch")
                    need(paired["arms"][arm]["lse_sha256"]==pristine["arms"]["pristine"]["lse_sha256"],"lse hash mismatch")
                numerics+=1
            for execution in EXECUTIONS:
                raw=q0["identities"]["cap"][execution]["payload"]
                identity=TacticIdentity(raw["environment"],raw["operation"],raw["measurement_policy"])
                eligibility=EligibilityDecision(bool(q0["eligibility"][execution]["eligible"]),tuple(q0["eligibility"][execution]["reasons"]))
                identity_keys=[];eligibility_receipts=[];cap_support=[]
                for rep in range(3):
                    qrep=lookup[shard,rep,"paired"]["qualifications"][key]
                    identity_keys.append(qrep["identities"]["cap"][execution]["key"])
                    eligibility_receipts.append(qrep["eligibility"][execution])
                    cap_support.append(qrep["cap_supported"])
                need(identity_keys==[identity.key]*3,"cross-repeat tactic identity drift")
                need(all(x==eligibility_receipts[0] for x in eligibility_receipts),"cross-repeat eligibility drift")
                need(all(x==cap_support[0] for x in cap_support),"cross-repeat cap-support drift")
                repeats={rep:block_evidence(lookup[shard,rep,"paired"]["rows"],key,execution) for rep in range(3)}
                absolute={rep:absolute_native_block_evidence(lookup[shard,rep,"pristine"]["rows"],lookup[shard,rep,"paired"]["rows"],key,execution) for rep in range(3)}
                absolute_summary=hierarchical_ratio_summary({rep:absolute[rep]["ratios"] for rep in range(3)},seed=int(identity.key[:16],16)^0xA850)
                pristine_control_values=[x for rep in range(3) for x in absolute[rep]["pristine_controls"]]
                off_control_values=[x for rep in range(3) for x in absolute[rep]["off_controls"]]
                pristine_ci=ci90(pristine_control_values,int(identity.key[16:24],16)^0x5151)
                off_ci=ci90(off_control_values,int(identity.key[24:32],16)^0x0FF0)
                controls_resolve=(pristine_ci[0]>=1/1.01 and pristine_ci[1]<=1.01 and off_ci[0]>=1/1.01 and off_ci[1]<=1.01)
                native_record={"case":key[0],"dtype":key[1],"layout":key[2],"split":key[3],"execution_mode":execution,
                    "shard":shard,"ratio":absolute_summary["ratio"],"CI95":absolute_summary["CI95"],
                    "block_worst":absolute_summary["block_worst"],"pristine_control_CI90":pristine_ci,
                    "candidate_native_control_CI90":off_ci,"controls_resolve_one_percent":controls_resolve}
                native_overlay_records.append(native_record);native_overlay_summaries.append(absolute_summary)
                folds=crossfit_cell(identity,eligibility,repeats,exact_outputs=True)
                for fold in folds:
                    held=repeats[fold.held_out_repeat]
                    nci=ci90(held.native_controls,int(identity.key[:8],16)^fold.held_out_repeat)
                    cci=ci90(held.cap_controls,int(identity.key[8:16],16)^fold.held_out_repeat)
                    held_control=(nci[0]>=1/1.01 and nci[1]<=1.01 and cci[0]>=1/1.01 and cci[1]<=1.01)
                    record={"case":key[0],"dtype":key[1],"layout":key[2],"split":key[3],"execution_mode":execution,
                        "shard":shard,"held_out_repeat":fold.held_out_repeat,"eligible":eligibility.eligible,
                        "selected":fold.receipt.tactic==TACTIC_CAP,"receipt":fold.receipt.to_dict(),
                        "held_out_geomean":fold.held_out_geomean,"held_out_minimum":fold.held_out_minimum,
                        "policy_geomean":fold.policy_geomean,"policy_minimum":fold.policy_minimum,
                        "held_out_native_control_CI90":nci,"held_out_cap_control_CI90":cci,
                        "held_out_controls_resolve_one_percent":held_control}
                    all_folds.append((fold,record));cell_receipts.append(record)
                    with tempfile.TemporaryDirectory(prefix="v4-cache-check-") as tmp:
                        cache=SafeTacticCache(tmp,identity);cache.publish(identity,fold.receipt.to_dict(),provenance={"stage":args.stage,"fold":fold.held_out_repeat});cache.reload()
                        cache_ok&=cache.lookup(identity) is not None
    # Held-out oracle uses measurements solely for evaluation, never selection.
    regrets=[];actual_policy=[]
    for fold,record in all_folds:
        oracle=max(1.0,record['held_out_geomean'])
        chosen=record['policy_geomean']
        regrets.append(max(0.0,oracle/chosen-1.0))
        key=(record['case'],record['dtype'],record['layout'],record['split'])
        shard=record['shard'];rep=record['held_out_repeat'];execution=record['execution_mode']
        native=absolute_native_block_evidence(lookup[shard,rep,'pristine']['rows'],lookup[shard,rep,'paired']['rows'],key,execution)
        actual_policy.append(geo(native['ratios'])*chosen)
    selected=[(fold,record) for fold,record in all_folds if record["selected"]]
    policy=[record["policy_geomean"] for _,record in all_folds]
    selected_geos=[record["held_out_geomean"] for _,record in selected]
    selected_mins=[record["held_out_minimum"] for _,record in selected]
    selected_controls=all(record["held_out_controls_resolve_one_percent"] for _,record in selected)
    modes={record["execution_mode"] for _,record in selected};layouts={record["layout"] for _,record in selected};dtypes={record["dtype"] for _,record in selected}
    joint=simultaneous_min_lcb([fold for fold,_ in selected],draws=10000,seed=917331) if selected else 1.0
    by_mode={}
    for mode in EXECUTIONS:
        values=[record["held_out_geomean"] for _,record in selected if record["execution_mode"]==mode]
        by_mode[mode]={"count":len(values),"geomean":geo(values) if values else None,"worst":min(values) if values else None}
    native_overlay_joint=simultaneous_ratio_lcb(native_overlay_summaries)
    native_overlay_controls=all(x["controls_resolve_one_percent"] for x in native_overlay_records)
    metrics={"fold_records":len(all_folds),"selected_records":len(selected),"coverage":len(selected)/len(all_folds),
        "selected_geomean":geo(selected_geos) if selected else None,"selected_worst":min(selected_geos) if selected else None,
        "selected_block_worst":min(selected_mins) if selected else None,"selected_joint_min_lcb95":joint,
        "policy_geomean":geo(policy),"policy_worst":min(policy),"by_execution_mode":by_mode,
        "selected_layouts":sorted(layouts),"selected_dtypes":sorted(dtypes),
        "native_overlay_geomean":geo([x["ratio"] for x in native_overlay_records]),
        "native_overlay_worst":min(x["ratio"] for x in native_overlay_records),
        "native_overlay_block_worst":min(x["block_worst"] for x in native_overlay_records),
        "native_overlay_joint_min_lcb95":native_overlay_joint}
    metrics['regret']={'definition':'held-out oracle latency / chosen latency - 1; native=1 in same-module tactic comparison',
        'p50':float(np.quantile(regrets,.5)),'p90':float(np.quantile(regrets,.9)),
        'p99':float(np.quantile(regrets,.99)),'worst':max(regrets),
        'above_one_percent_count':sum(x>.01 for x in regrets),'records':len(regrets)}
    metrics['actual_policy_vs_pristine_geomean']=geo(actual_policy)
    metrics['actual_policy_vs_pristine_worst']=min(actual_policy)
    expected_numeric=len(cases)*len(manifest["dtypes"])*len(manifest["layouts"])*len(manifest["requested_splits"])*3
    requirements={"numerical_exact":numerics==expected_numeric,"native_after_cap_exact":native_after_cap,
        "compiled_kernel_symbol_isolation":len(binary_audits)==args.shards,
        "native_sass_identity":all(x.get("native_sass_identity") is True for x in binary_audits.values()),
        "native_source_identity":json.loads((args.root/"overlays/candidate/RESOURCE_BINDING.json").read_text()).get("native_source_audit",{}).get("native_source_identity") is True,
        "telemetry_stable":len(telemetry)==args.shards and all(x["stable"] for x in telemetry.values()),
        "cache_roundtrip":cache_ok,"selected_nonempty":bool(selected),
        "selected_geomean_at_least_1_05":bool(selected) and metrics["selected_geomean"]>=1.05,
        "selected_worst_at_least_0_99":bool(selected) and metrics["selected_worst"]>=.99,
        "selected_block_worst_at_least_0_99":bool(selected) and metrics["selected_block_worst"]>=.99,
        "selected_joint_min_lcb_at_least_0_99":bool(selected) and joint>=.99,
        "policy_worst_at_least_0_99":metrics["policy_worst"]>=.99,
        "actual_policy_vs_pristine_worst_at_least_0_99":min(actual_policy)>=.99,
        "native_overlay_worst_at_least_0_99":metrics["native_overlay_worst"]>=.99,
        "native_overlay_joint_min_lcb_at_least_0_99":native_overlay_joint>=.99,
        "native_overlay_controls_resolve_one_percent":native_overlay_controls,
        "held_out_controls_resolve":selected_controls,
        "all_execution_modes_selected":modes==set(EXECUTIONS),"both_layouts_selected":layouts=={"ragged","paged"},
        "both_dtypes_selected":dtypes=={"float16","bfloat16"}}
    result={"qualification_revision":QUALIFICATION_REVISION,"stage":args.stage,"gpu":args.gpu,
        "pass":all(requirements.values()),"requirements":requirements,"metrics":metrics,"hardware":hardware,
        "source_archive_sha256":next(iter(source)),"official_overlay_sha256":next(iter(overlay)),
        "case_hash":manifest["case_hash"],"stage_hash":manifest["stage_hashes"][args.stage],"measurement_manifest_sha256":hashlib.sha256(measurement_manifest_path.read_bytes()).hexdigest(),
        "campaign_source_commit":(args.root/"source/SOURCE_COMMIT.txt").read_text().strip(),
        "analysis_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "binary_audits":binary_audits,"telemetry":telemetry,
        "records":cell_receipts,"native_overlay_records":native_overlay_records,
        "default_promotion":False,"serving_promotion":False}
    args.out.mkdir(parents=True);(args.out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    lines=["# Selector v4 safe-autotune qualification","",f"**{args.gpu}: {'PASS' if result['pass'] else 'HOLD'}**","",
        f"- Cross-fit records: {metrics['fold_records']}; selected {metrics['selected_records']} ({metrics['coverage']:.1%}).",
        f"- Held-out selected geomean {ratio_text(metrics['selected_geomean'])}; point worst {ratio_text(metrics['selected_worst'])}; block worst {ratio_text(metrics['selected_block_worst'])}; joint-min LCB {ratio_text(joint)}.",
        f"- Whole-policy geomean {ratio_text(metrics['policy_geomean'])}; worst {ratio_text(metrics['policy_worst'])}.",
        f"- Candidate-native / pristine geomean {ratio_text(metrics['native_overlay_geomean'])}; worst {ratio_text(metrics['native_overlay_worst'])}; joint-min LCB {ratio_text(metrics['native_overlay_joint_min_lcb95'])}.",
        f"- Requirements: `{json.dumps(requirements,sort_keys=True)}`"]
    (args.out/"RESULTS.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--stage",choices=["dev","canary","release","stress"],required=True);p.add_argument("--gpu",choices=["gpu_4090","gpu_5090"],required=True);p.add_argument("--shards",type=int,required=True);run(p.parse_args())
