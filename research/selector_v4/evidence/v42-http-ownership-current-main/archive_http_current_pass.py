from pathlib import Path
import hashlib,json,tarfile,subprocess
p=Path('/ssd/scxi253/sgi-http-ownership-current-main-3825c5a-post1');out=Path('/ssd/scxi253/sgi-http-current-pass-archive-20261001');out.mkdir(exist_ok=True)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();files=[]
for d in ['receipts','logs','runs']:
 if (p/d).exists():files.extend(f for f in (p/d).rglob('*') if f.is_file())
for n in ['source.sha256','source-binding.json','http_ownership_probe.py','http_ownership_server.py','http_ownership.sbatch']:files.append(p/n)
a=out/'raw-http-pass.tar.gz';manifest={str(f.relative_to(p)):sha(f) for f in sorted(files)}
with tarfile.open(a,'w:gz') as t:
 for n in manifest:t.add(p/n,arcname=n)
with tarfile.open(a) as t:
 for n,h in manifest.items():assert hashlib.sha256(t.extractfile(n).read()).hexdigest()==h
sacct=subprocess.run(['sacct','-j','1644258','--format=JobID,State,ExitCode,Elapsed','-n'],text=True,capture_output=True,check=True).stdout
original=json.loads((p/'runs/original-1644258/complete.json').read_text());candidate=json.loads((p/'runs/candidate-1644258/complete.json').read_text())
assert candidate['candidate_pass'] and original['checked_requests']==candidate['checked_requests']==10
assert not original['candidate_pass'] and len(original['stale_reused_rid_aborts'])==2
assert candidate['intentional_disconnects']==original['intentional_disconnects']==1
assert not candidate['stale_reused_rid_aborts'] and not candidate['remaining_owned_at_cleanup']
r={'candidate_commit':'3825c5a4b5e8a1a73ae62f0c0ba6eb35beb1a2ca','upstream_base':'baae019','job':'1644258','sacct':sacct,'source_binding':json.loads((p/'source-binding.json').read_text()),'campaign_binding':json.loads((p/'receipts/campaign.json').read_text()),'stage':'complete full-model production HTTP ownership diagnostic','cpu_registered_tests_passed':49,'model':'full36-layer BF16 Qwen3-4B-Instruct-2507','original':original,'candidate':candidate,'http_lifecycle_verdict':True,'holdout_cases_consumed':0,'archive_sha256':sha(a),'archive_bytes':a.stat().st_size,'files':manifest,'resource_cap_enabled':False,'performance_qualified':False,'historical_token_divergence_resolved':False}
(out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(a.stat().st_size,len(manifest),sha(a))
