"""Create an immutable v4.1 campaign root. This tool never submits a job."""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, os, shutil, tarfile
from .manifest_v4 import load
from .prepare_overlay import EXPECTED, prepare


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

def safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            resolved=(destination/member.name).resolve()
            if destination.resolve() not in resolved.parents and resolved != destination.resolve():
                raise RuntimeError("unsafe archive path")
            if member.issym() or member.islnk(): raise RuntimeError("links forbidden in source archive")
        stream.extractall(destination, filter="data")

def prepare_campaign(*, root: Path, pristine_source: Path, source_archive: Path,
                     source_commit: str, official_overlay_sha256: str, stage: str, shards: int) -> dict:
    root=Path(root);pristine_source=Path(pristine_source);source_archive=Path(source_archive)
    if root.exists(): raise FileExistsError(root)
    manifest=load();cases=[x for x in manifest["cases"] if x["family"]==stage]
    if not cases or not 1<=shards<=12 or any(not [c for i,c in enumerate(cases) if i%shards==s] for s in range(shards)):
        raise ValueError("invalid nonempty shard contract")
    for rel,expected in EXPECTED.items():
        if sha(pristine_source/rel)!=expected: raise RuntimeError("unreviewed pristine source "+rel)
    for name in ("source","overlays","receipts","runs","refs","cache","logs","analysis"):(root/name).mkdir(parents=True)
    shutil.copyfile(source_archive,root/"source-archive.tar.gz");safe_extract(source_archive,root/"source")
    commit=(root/"source/SOURCE_COMMIT.txt").read_text().strip()
    if commit!=source_commit: raise RuntimeError("source commit mismatch")
    ledger=root/"source/source.sha256"
    if not ledger.is_file(): raise RuntimeError("missing source hash ledger")
    for line in ledger.read_text().splitlines():
        expected,rel=line.split(None,1);rel=rel.strip().removeprefix("./")
        target=root/"source"/rel
        if not target.is_file() or sha(target)!=expected: raise RuntimeError("source ledger mismatch "+rel)
    # Copy pristine with independent metadata while hard-linking immutable package files.
    shutil.copytree(pristine_source/"flashinfer",root/"overlays/pristine/flashinfer",copy_function=os.link,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    for metadata in pristine_source.glob("*.dist-info"):
        shutil.copytree(metadata,root/"overlays/pristine"/metadata.name,copy_function=os.link)
    binding=prepare(pristine_source,root/"overlays/candidate")
    if len(official_overlay_sha256)!=64 or any(c not in "0123456789abcdef" for c in official_overlay_sha256):
        raise ValueError("invalid official overlay identity")
    official=official_overlay_sha256;archive_sha=sha(source_archive);binding_sha=sha(root/"overlays/candidate/RESOURCE_BINDING.json")
    (root/"receipts/source-archive.sha256").write_text(archive_sha+"\n")
    (root/"receipts/official-overlay.sha256").write_text(official+"\n")
    (root/"receipts/resource-binding.sha256").write_text(binding_sha+"\n")
    campaign={"schema":4,"qualification_revision":manifest["qualification_revision"],"root":str(root),"stage":stage,
        "source_commit":source_commit,"source_archive_sha256":archive_sha,"official_overlay_sha256":official,
        "resource_binding_sha256":binding_sha,"case_hash":manifest["case_hash"],"stage_hash":manifest["stage_hashes"][stage],
        "shards":shards,"blocks":manifest["blocks"],"release_cases_consumed":0,"default_promotion":False,
        "serving_promotion":False,"historical_token_divergence_resolved":False,"binding":binding,"jobs":{}}
    (root/"receipts/campaign.json").write_text(json.dumps(campaign,indent=2)+"\n")
    return campaign

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--pristine-source",type=Path,required=True)
    p.add_argument("--source-archive",type=Path,required=True);p.add_argument("--source-commit",required=True);p.add_argument("--official-overlay-sha256",required=True)
    p.add_argument("--stage",choices=["dev","canary","release","stress"],required=True);p.add_argument("--shards",type=int,required=True)
    print(json.dumps(prepare_campaign(**vars(p.parse_args())),indent=2))
