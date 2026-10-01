"""Archive exact-source dual GPU functional and prepared managed-v2 results."""
import hashlib,json,subprocess,tarfile,xml.etree.ElementTree as ET
from pathlib import Path

p=Path('/ssd/scxi253/sgi-flashinfer-managed-v2-d7683dd-20261001')
out=Path('/ssd/scxi253/sgi-main-managed-d768-success-archive')
out.mkdir()
jobs=json.loads((p/'receipts/jobs.json').read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
binding=json.loads((p/'binding.json').read_text())
s=Path(binding['source'])
verification=subprocess.run(['sha256sum','-c','source.sha256'],cwd=s,text=True,capture_output=True,check=True)
(out/'source-final-verification.txt').write_text(verification.stdout)
results={}
for partition,job in jobs.items():
    status=subprocess.check_output(['sacct','-j',job,'--format=JobID,State,ExitCode,Elapsed','-n','-X'],text=True)
    assert 'COMPLETED' in status and '0:0' in status,status
    xml=ET.parse(p/'receipts'/f'tests-{job}.xml')
    assert len(xml.findall('.//testcase'))==49
    assert not any(xml.findall('.//'+key) for key in ('failure','error','skipped'))
    modes=json.loads((p/f'modes-{job}/summary.json').read_text())
    assert modes['pass'] and modes['three_distinct_execution_identities']
    for name,m in modes['modes'].items():
        assert m['managed_exact'] and m['disk_reload_replay_exact']
        assert m['managed_output_sha256']==m['native_output_sha256']
        assert m['managed_tactic']==m['reloaded_tactic']
        assert m['persisted_winner']['tactic']==m['managed_tactic']
        assert m['identity'] in m['persisted_winner']['key']
        if name.startswith('graph'):assert m['captured_managed_exact']
        if not all(m['checks'].values()):assert m['managed_tactic'] in (-1,0)
    results[partition]={'job':job,'sacct':status,'cpu_pass':37,'gpu_pass':12,'modes':modes}
files=[]
for d in ('logs','receipts'):
    files.extend(f for f in (p/d).rglob('*') if f.is_file())
files.extend(f for f in p.iterdir() if f.is_file())
for job in jobs.values():
    files.extend(f for f in (p/f'modes-{job}').rglob('*') if f.is_file())
    # Retain the example's separately managed store too.
    files.extend(f for f in (p/'cache'/job).rglob('*.json') if '/autotune/' in str(f))
manifest={str(f.relative_to(p)):sha(f) for f in sorted(files)}
a=out/'raw-managed-validation.tar.gz'
with tarfile.open(a,'w:gz') as t:
    for name in manifest:t.add(p/name,arcname=name)
with tarfile.open(a) as t:
    for name,h in manifest.items():assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==h
receipt={'pass':True,'binding':binding,'results':results,'archive_sha256':sha(a),'archive_bytes':a.stat().st_size,'files':manifest,'source_final_verification_sha256':sha(out/'source-final-verification.txt'),'scope':'exposed geometry; prepared eager and pure Graph1/16; no metadata updates/full serving','independent_process_release_qualified':False,'fresh_cases_consumed':0,'default_promotion':False,'serving_promotion':False,'historical_token_divergence_resolved':False}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'pass':True,'jobs':jobs,'files':len(manifest),'archive_sha256':sha(a),'archive_bytes':a.stat().st_size}))
