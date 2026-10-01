"""Preserve all public-path development evidence, including terminal HOLD/failures."""
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
b=Path('/ssd/scxi253');p=b/'sgi-public-path-prequal-c6879fb-75544a17-20261001'
c=b/'sgi-public-path-prequal-completion-c6879fb'
out=b/'sgi-public-path-prequal-c6879fb-complete-archive'
def hash_stream(stream):
 h=hashlib.sha256()
 while chunk:=stream.read(8<<20):h.update(chunk)
 return h.hexdigest()
def sha(path):
 with path.open('rb') as stream:return hash_stream(stream)
controller=json.loads((c/'receipt.json').read_text())
assert controller.get('terminal') is True,'Wait for finite controller termination'
jobs=json.loads((p/'receipts/jobs.json').read_text());statuses={}
for gpu,group in jobs.items():
 for case,job in group.items():
  status=subprocess.check_output(['sacct','-j',job,'--format=JobID,State,ExitCode,Elapsed','-n','-X'],text=True)
  assert status.strip() and not any(s in status for s in ('RUNNING','PENDING','COMPLETING','CONFIGURING','SUSPENDED','REQUEUED')),status
  statuses[job]=status
files={};sources={}
for prefix,root in (('campaign',p),('completion',c)):
 for parent,dirs,names in os.walk(root):
  dirs[:]=[d for d in dirs if d not in ('cache','sdk-libraries','pristine-overlay','__pycache__','.pytest_cache') and not d.startswith('xdg-')]
  for n in names:
   f=Path(parent)/n;name=prefix+'/'+str(f.relative_to(root));files[name]=sha(f);sources[name]=f
helper=Path(__file__);files['archive_public_path_prequal.py']=sha(helper);sources['archive_public_path_prequal.py']=helper
analyses={}
for gpu in jobs:
 summary=p/f'analysis/{gpu}/summary.json'
 if summary.exists():
  result=json.loads(summary.read_text())
  for name,expected in result['files'].items():assert sha(p/name)==expected,name
  assert result['binding_sha256']==sha(p/'binding.json')
  analyses[gpu]={'pass':result['pass'],'summary_sha256':sha(summary),'metrics':result['metrics'],'failed_requirements':[k for k,v in result['requirements'].items() if not v]}
all_complete=len(analyses)==2 and all('COMPLETED' in s and '0:0' in s for s in statuses.values())
passed=all_complete and all(a['pass'] for a in analyses.values())
out.mkdir()
a=out/'raw-public-path-development.tar.gz'
with tarfile.open(a,'w:gz') as t:
 for name in sorted(files):t.add(sources[name],arcname=name,recursive=False)
with tarfile.open(a) as t:
 assert set(t.getnames())==set(files)
 for name,expected in files.items():assert hash_stream(t.extractfile(name))==expected,name
receipt=dict(pass_public_path_development=passed,all_nine_phases_complete_on_both_cases_and_cards=all_complete,binding=json.loads((p/'binding.json').read_text()),controller=controller,jobs=jobs,statuses=statuses,analyses=analyses,files=files,archive_sha256=sha(a),archive_bytes=a.stat().st_size,raw_references_included=True,trimmed_windows=0,fresh_cases_consumed=0,qualification_authority=False,canary_authority=False,full_http_qualified=False,default_promotion=False,serving_promotion=False,historical_token_divergence_resolved=False)
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ('pass_public_path_development','all_nine_phases_complete_on_both_cases_and_cards','archive_sha256','archive_bytes')}))
