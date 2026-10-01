"""Archive the full frozen canary and its technical recovery, retaining failures."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import tarfile

b = Path('/ssd/scxi253')
p = b / 'single-gpu-inference-selector-v42-20260930T230508Z'
r = b / 'sgi-v42-canary-rpc-recovery-20261001T0505Z-single-gpu'
d = b / 'sgi-v42-canary-recovery-audit-repair-20261001T0720Z'
out = b / 'sgi-v42-canary4090-complete-hold-archive'
out.mkdir()
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
pre = json.loads((r / 'recovery-preregistration.json').read_text())
retained = {}
for info in pre['failures'].values():
    for n, h in info['retained_files'].items():
        assert sha(p / n) == h, n
        retained[n] = h
verdicts = {}
for gpu, f in (
    ('gpu_4090', d / 'analysis/canary-gpu_4090-complete-inputs/summary.json'),
    ('gpu_5090', p / 'analysis/canary-gpu_5090/summary.json'),
):
    x = json.loads(f.read_text())
    assert x['campaign_source_commit'] == 'b769e7c18e93c9d6cfdcd79b58edf76955313859'
    assert not x['pass']
    verdicts[gpu] = dict(pass_=False, summary=str(f), summary_sha256=sha(f),
                         failed_requirements=[k for k, v in x['requirements'].items() if not v],
                         metrics=x['metrics'])
complete = []
for shard in range(10):
    for rep in range(3):
        for mode in ('pristine', 'paired'):
            roots = list((d / 'runs').glob(f'canary-gpu_4090-s{shard}-r{rep}-{mode}-*'))
            assert len(roots) == 1 and (roots[0] / 'complete.json').exists()
            complete.append(roots[0])
assert len(complete) == 60
status = subprocess.check_output(['sacct', '-j', '1644913,1644914,1644929',
                                 '--format=JobID,State,ExitCode,Elapsed', '-n', '-X'], text=True)
assert status.count('COMPLETED') == 3 and status.count('0:0') == 3, status
terminal = {
    'state': 'CANARY_QUALIFICATION_HOLD_COMPLETENESS_RECOVERED',
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_commit': 'b769e7c18e93c9d6cfdcd79b58edf76955313859',
    'original_campaign': str(p), 'technical_recovery': str(r), 'cpu_audit_repair': str(d),
    'complete_4090_processes': 60, 'resumed_missing_processes': 5,
    'original_to_resume_jobs': json.loads((r / 'receipts/resume-jobs.json').read_text()),
    'resume_sacct': status, 'retained_original_files_reverified': retained,
    'verdicts': verdicts, 'fresh_cases_consumed_by_recovery': 0,
    'canary_cases_consumed': 10, 'release_cases_consumed': 0, 'stress_cases_consumed': 0,
    'source_changed': False, 'thresholds_changed': False, 'gpu_measurements_repeated': False,
    'original_terminal_controller_unmodified': True,
    'failure_history': {
        'original_pre_process_rpc': sha(p / 'receipts/kernel-pipeline.json'),
        'recovery_cpu_cache_path_discovery': sha(r / 'receipts/recovery-controller.json'),
        'first_analysis_missing_late_audit_links': sha(d / 'receipts/repair-result.json'),
        'late_original_audit_discovery': sha(d / 'receipts/late-original-audit-discovery.json'),
    },
    'default_promotion': False, 'serving_promotion': False,
    'historical_token_divergence_resolved': False, 'subsequent_stages_authorized': False,
    'regret_definition_annotation': 'Numeric implementation is chosen latency / oracle latency - 1. Frozen text describes its inverse and is preserved unchanged.',
}
final = d / 'receipts/complete-input-terminal.json'
assert not final.exists()
final.write_text(json.dumps(terminal, indent=2) + '\n')
files = {}
def add(f, name):
    assert f.is_file(), str(f)
    files[name] = f
def tree(root, prefix):
    if root.exists():
        for f in root.rglob('*'):
            if f.is_file(): add(f, prefix + '/' + str(f.relative_to(root)))
for root in complete:
    tree(root, 'complete-4090/runs/' + root.name)
for folder in ('receipts', 'logs'):
    tree(d / folder, 'cpu-repair/' + folder)
    tree(r / folder, 'technical-recovery/' + folder)
tree(d / 'analysis/canary-gpu_4090-complete-inputs', 'complete-4090/analysis')
add(p / 'analysis/canary-gpu_5090/summary.json', 'reference-5090/summary.json')
for name in ('recovery-preregistration.json', 'resume_measure_single.py',
             'canary_resume_single.sbatch', 'complete_canary_recovery_single.py'):
    if (r / name).exists(): add(r / name, 'technical-recovery/' + name)
for name in ('repair_canary_cpu_audit.py', 'archive_canary4090_recovered.py'):
    add(b / name, 'helpers/' + name)
for name in ('receipts/kernel-pipeline.json', 'receipts/canary-dispatch.json',
             'source/source.sha256', 'source/SOURCE_COMMIT.txt',
             'source/research/selector_v4/analyze_v4.py',
             'source/research/selector_v4/manifest.json'):
    add(p / name, 'original/' + name)
for j in range(1644188, 1644198):
    for name in (f'logs/canary-{j}.out', f'logs/telemetry-{j}.csv'):
        if (p / name).exists(): add(p / name, 'original/' + name)
    for f in (p / 'receipts').glob(f'*{j}*'):
        if f.is_file(): add(f, 'original/receipts/' + f.name)
ledger = {n: sha(f) for n, f in sorted(files.items())}
a = out / 'raw-canary4090-complete.tar.gz'
with tarfile.open(a, 'w:gz', dereference=True) as t:
    for n, f in sorted(files.items()): t.add(f, arcname=n, recursive=False)
with tarfile.open(a) as t:
    assert all(x.isfile() for x in t.getmembers())
    for n, h in ledger.items(): assert hashlib.sha256(t.extractfile(n).read()).hexdigest() == h
receipt = dict(terminal, files=ledger, archive_sha256=sha(a), archive_bytes=a.stat().st_size,
               all_archive_members_regular_files=True)
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k: receipt[k] for k in ('state', 'archive_sha256', 'archive_bytes', 'verdicts')}))
