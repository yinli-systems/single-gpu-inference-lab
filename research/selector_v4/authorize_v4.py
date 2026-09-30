from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from .manifest_v4 import load
from .schema import QUALIFICATION_REVISION

NATIVE_OVERLAY_REQUIREMENTS = (
    "native_overlay_worst_at_least_0_99",
    "native_overlay_joint_min_lcb_at_least_0_99",
    "native_overlay_controls_resolve_one_percent",
)

def need(value: bool, message: str) -> None:
    if not value: raise RuntimeError(message)

def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_summary(root: Path, stage: str, gpu: str, manifest: dict, source: str, overlay: str) -> dict:
    path=root/f"analysis/{stage}-{gpu}/summary.json"
    need(path.exists(),f"missing {stage} {gpu}")
    value=json.loads(path.read_text())
    need(value.get("pass") is True,f"{stage} HOLD {gpu}")
    need(value.get("stage")==stage and value.get("gpu")==gpu,f"{stage} route identity {gpu}")
    need(value.get("qualification_revision")==QUALIFICATION_REVISION,f"{stage} qualification revision {gpu}")
    need(value.get("case_hash")==manifest["case_hash"] and value.get("stage_hash")==manifest["stage_hashes"][stage],f"{stage} shape identity {gpu}")
    need(value.get("source_archive_sha256")==source and value.get("official_overlay_sha256")==overlay,f"{stage} provenance {gpu}")
    manifest_path=root/"source/research/selector_v4/manifest.json"
    analyzer_path=root/"source/research/selector_v4/analyze_v4.py"
    commit_path=root/"source/SOURCE_COMMIT.txt"
    need(value.get("measurement_manifest_sha256")==sha(manifest_path),f"{stage} measurement manifest {gpu}")
    need(value.get("analysis_source_sha256")==sha(analyzer_path),f"{stage} analyzer source {gpu}")
    need(value.get("campaign_source_commit")==commit_path.read_text().strip(),f"{stage} campaign source {gpu}")
    requirements=value.get("requirements")
    need(isinstance(requirements,dict),f"{stage} requirements missing {gpu}")
    need(all(requirements.get(key) is True for key in NATIVE_OVERLAY_REQUIREMENTS),f"{stage} native-overlay gate {gpu}")
    return value

def authorize(root: Path, stage: str) -> dict:
    root=Path(root);manifest=load(root/"source/research/selector_v4/manifest.json")
    source=(root/"receipts/source-archive.sha256").read_text().strip();overlay=(root/"receipts/official-overlay.sha256").read_text().strip()
    canaries={}
    for gpu in ("gpu_4090","gpu_5090"):
        value=validate_summary(root,"canary",gpu,manifest,source,overlay)
        canaries[gpu]={"summary_sha256":sha(root/f"analysis/canary-{gpu}/summary.json"),"metrics":value["metrics"],"hardware":value["hardware"]}
    out={"authorized_stage":stage,"qualification_revision":QUALIFICATION_REVISION,"case_hash":manifest["case_hash"],
        "source_archive_sha256":source,"official_overlay_sha256":overlay,"canaries":canaries,
        "default_promotion":False,"serving_promotion":False,"historical_token_divergence_resolved":False}
    if stage=="stress":
        releases={}
        for gpu in ("gpu_4090","gpu_5090"):
            value=validate_summary(root,"release",gpu,manifest,source,overlay)
            releases[gpu]={"summary_sha256":sha(root/f"analysis/release-{gpu}/summary.json"),"metrics":value["metrics"],"hardware":value["hardware"]}
        out["release_passed_both_gpus"]=True;out["releases"]=releases
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--stage",choices=["release","stress"],required=True)
    args=p.parse_args();print(json.dumps(authorize(args.root,args.stage),indent=2))
