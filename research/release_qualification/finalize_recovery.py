"""Build a hash-bound final recovery map after one paired shard rerun.

Never edits, deletes, or scores the superseded evidence itself.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
MODES=('pristine','off','cap','guarded')
REPS=range(3)
def require(x,m):
 if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_env(p):return json.loads((p/'environment.json').read_text())
def verify_complete(p):
 c=json.loads((p/'complete.json').read_text());require(c.get('complete') is True,'completion flag '+str(p))
 require((p/'launcher-complete.txt').exists(),'launcher receipt '+str(p))
 for f,h in c.get('files',{}).items():require(Path(f).name==f and (p/f).is_file() and sha(p/f)==h,'hash '+str(p/f))
 return c
def find_one(root,job,rep,mode):
 xs=list((root/'runs').glob(f'release-gpu_4090-s3-r{rep}-{mode}-{job}'))
 require(len(xs)==1,f'run cardinality {job} r{rep} {mode}: {xs}');return xs[0]
def run(a):
 require(not a.out.exists(),'preserve final map');prior=a.root/'receipts'/'recovery-map.json';require(prior.exists(),'prior map missing')
 prior_data=json.loads(prior.read_text());require(prior_data.get('schema')==1,'wrong prior map')
 entries=[];replacement_hw=set();source=set()
 for rep in REPS:
  for mode in MODES:
   old=find_one(a.root,a.old_job,rep,mode);new=find_one(a.root,a.replacement_job,rep,mode)
   oe=load_env(old);ne=load_env(new);key=('release',3,6,rep,mode)
   require((oe['stage'],oe['shard'],oe['shards'],oe['rep'],oe['mode'])==key,'old identity')
   require((ne['stage'],ne['shard'],ne['shards'],ne['rep'],ne['mode'])==key,'new identity')
   require((old/'source_commit.txt').is_file() and (new/'source_commit.txt').is_file(),'source commit receipt missing')
   source|={ (old/'source_commit.txt').read_text().strip(), (new/'source_commit.txt').read_text().strip() };replacement_hw.add(ne['hardware']['out'].strip())
   base=dict(superseded=old.relative_to(a.root).as_posix(),replacement=new.relative_to(a.root).as_posix(),
    key=dict(shard=3,rep=rep,mode=mode),old_job=a.old_job,replacement_job=a.replacement_job,
    replacement_complete_sha256=sha(new/'complete.json'))
   verify_complete(new)
   if rep==2 and mode=='pristine':
    require(not (old/'complete.json').exists() and not (old/'failure.json').exists(),'expected retained partial')
    prog=json.loads((old/'progress.json').read_text());require(prog['rows']==3456 and prog['qualifications']==54,'partial progress changed')
    base.update(superseded_kind='partial',reason='Slurm time limit after 3456/4096 rows and 54/64 qualifications; retained and never scored',
     old_state='TIMEOUT',old_environment_sha256=sha(old/'environment.json'),old_progress_sha256=sha(old/'progress.json'))
   else:
    verify_complete(old);base.update(superseded_kind='complete',reason='Entire shard 3 allocation replaced to preserve within-shard paired hardware/driver identity',old_complete_sha256=sha(old/'complete.json'))
   entries.append(base)
 require(len(entries)==12 and len(replacement_hw)==1 and len(source)==1,'replacement matrix/source/hardware')
 attempt=find_one(a.root,a.unscored_job,2,'pristine');verify_complete(attempt);ae=load_env(attempt)
 require((ae['stage'],ae['shard'],ae['shards'],ae['rep'],ae['mode'])==('release',3,6,2,'pristine'),'unscored identity')
 old_hw=load_env(find_one(a.root,a.old_job,2,'cap'))['hardware']['out'].strip()
 require(ae['hardware']['out'].strip()!=old_hw,'unscored attempt unexpectedly paired')
 out=dict(schema=2,stage='release',gpu='gpu_4090',measurement_source_commit=next(iter(source)),
  prior_map=prior.relative_to(a.root).as_posix(),prior_map_sha256=sha(prior),entries=entries,
  replacement_allocation=dict(job=a.replacement_job,hardware=next(iter(replacement_hw)),complete_runs=12),
  unscored_attempts=[dict(path=attempt.relative_to(a.root).as_posix(),job=a.unscored_job,
    reason='Complete one-arm recovery ran on a different GPU UUID/driver and is retained but excluded from paired analysis',complete_sha256=sha(attempt/'complete.json'))])
 a.out.parent.mkdir(parents=True,exist_ok=True);tmp=a.out.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(a.out);print(json.dumps(out,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--old-job',type=int,required=True);p.add_argument('--unscored-job',type=int,required=True);p.add_argument('--replacement-job',type=int,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
