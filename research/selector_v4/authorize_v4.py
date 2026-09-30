from __future__ import annotations
import argparse, json
from pathlib import Path
from .manifest_v4 import load
from .schema import QUALIFICATION_REVISION

def need(value: bool, message: str) -> None:
    if not value: raise RuntimeError(message)

def authorize(root: Path, stage: str) -> dict:
    root=Path(root);manifest=load(root/"source/research/selector_v4/manifest.json")
    source=(root/"receipts/source-archive.sha256").read_text().strip();overlay=(root/"receipts/official-overlay.sha256").read_text().strip()
    canaries={}
    for gpu in ("gpu_4090","gpu_5090"):
        path=root/f"analysis/canary-{gpu}/summary.json";need(path.exists(),"missing canary "+gpu)
        value=json.loads(path.read_text());need(value.get("pass") is True,"canary HOLD "+gpu)
        need(value.get("qualification_revision")==QUALIFICATION_REVISION and value.get("case_hash")==manifest["case_hash"],"canary identity "+gpu)
        need(value.get("source_archive_sha256")==source and value.get("official_overlay_sha256")==overlay,"canary provenance "+gpu)
        canaries[gpu]={"metrics":value["metrics"],"hardware":value["hardware"]}
    out={"authorized_stage":stage,"qualification_revision":QUALIFICATION_REVISION,"case_hash":manifest["case_hash"],
        "source_archive_sha256":source,"official_overlay_sha256":overlay,"canaries":canaries,
        "default_promotion":False,"serving_promotion":False,"historical_token_divergence_resolved":False}
    if stage=="stress":
        for gpu in ("gpu_4090","gpu_5090"):
            path=root/f"analysis/release-{gpu}/summary.json";need(path.exists(),"missing release "+gpu)
            value=json.loads(path.read_text());need(value.get("pass") is True,"release HOLD "+gpu)
        out["release_passed_both_gpus"]=True
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--stage",choices=["release","stress"],required=True)
    args=p.parse_args();print(json.dumps(authorize(args.root,args.stage),indent=2))
