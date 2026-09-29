"""CPU-only artifact entry point. Never allocates a GPU or launches a job."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from representations import audit
from audit_vidur import audit as audit_upstream

BASE=Path(__file__).resolve().parent
MEASUREMENT_SHA256='a5c9995b3d3ad442977ac1fdd0609741384124d7eb6c1561395a77f51710287a'

def extract_data(archive:Path,destination:Path):
 if destination.exists():raise FileExistsError('new extraction directory required')
 with tarfile.open(archive,'r:gz') as t:
  members=t.getmembers()
  if sum(m.size for m in members)>256*1024**2:raise ValueError('archive exceeds the registered artifact size bound')
  if len(members)!=len({m.name for m in members}):raise ValueError('duplicate archive entry')
  for m in members:
   p=Path(m.name)
   if p.is_absolute() or '..' in p.parts or not(m.isfile() or m.isdir()):raise ValueError('unsafe archive member')
   if not p.parts or p.parts[0] not in ('runs','logs','receipts'):raise ValueError('unexpected artifact root')
  destination.mkdir(parents=True)
  for m in members:
   target=destination/m.name
   if m.isdir():target.mkdir(parents=True,exist_ok=True);continue
   target.parent.mkdir(parents=True,exist_ok=True)
   source=t.extractfile(m)
   if source is None:raise ValueError('missing archive payload')
   with source,target.open('xb') as out:shutil.copyfileobj(source,out)
 return destination/'runs'

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--archive',type=Path);p.add_argument('--extract-to',type=Path)
 p.add_argument('--runs',type=Path);p.add_argument('--out',type=Path)
 a=p.parse_args()
 if a.archive and a.runs:p.error('use archive or existing runs, not both')
 if a.archive and not a.extract_to:p.error('--archive requires --extract-to')
 if (a.archive or a.runs) and not a.out:p.error('data analysis requires a new --out directory')
 sha=hashlib.sha256((BASE/'campaign.py').read_bytes()).hexdigest()
 if sha!=MEASUREMENT_SHA256:raise ValueError('measurement source differs from registered experiment')
 subprocess.run([sys.executable,'-m','unittest','-v','test_campaign','test_analysis','test_representations','test_upstream'],cwd=BASE,check=True)
 if audit()!=json.loads((BASE/'representation-audit.json').read_text()):raise ValueError('representation golden-data mismatch')
 if audit_upstream(BASE/'reference_sources/vidur/sklearn_execution_time_predictor.py')!=json.loads((BASE/'vidur-source-audit.json').read_text()):raise ValueError('upstream feature-map golden-data mismatch')
 print('CPU_ARTIFACT_PASS: source identity,31contracts,exact representation audit,108upstream feature-map calls',flush=True)
 if a.archive:
  manifest=BASE/'data-manifest.json'
  if not manifest.exists():raise ValueError('no published data manifest')
  m=json.loads(manifest.read_text())
  if hashlib.sha256(a.archive.read_bytes()).hexdigest()!=m['archive_sha256']:raise ValueError('data archive checksum mismatch')
  a.runs=extract_data(a.archive,a.extract_to)
 if a.runs:
  from analyze import analyze
  analyze(a.runs,a.out)
  print('COMPLETE_MATRIX_REPRODUCED',a.out,flush=True)

if __name__=='__main__':main()
