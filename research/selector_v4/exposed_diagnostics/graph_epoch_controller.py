"""Finite two-card diagnostic submission, validation and raw archival; no retry."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    target = path.with_suffix(path.suffix+'.tmp')
    target.write_text(json.dumps(value, indent=2)+'\n')
    target.replace(path)


def validate(root, job, gpu):
    failures = []
    tests = root/'receipts'/f'tests-{job}.xml'
    cells = sorted((root/'runs'/f'{gpu}-{job}').glob('*/result.json'))
    if not tests.exists():
        failures.append('missing_pytest_xml')
    else:
        testcases = ET.parse(tests).findall('.//testcase')
        if len(testcases) != 24 or any(c.find(tag) is not None for c in testcases for tag in ('failure','error','skipped')):
            failures.append('incomplete_or_failed_24_case_population')
    expected = {f'{dtype}-{layout}-{packing}-graph{count}' for dtype in ('float16','bfloat16')
                for layout in ('NHD','HND') for packing in ('packed','tuple') for count in (1,16)}
    if {p.parent.name for p in cells} != expected:
        failures.append('incomplete_16_gpu_cells')
    for path in cells:
        result = json.loads(path.read_text())
        if result.get('cell') != path.parent.name or gpu[4:] not in result.get('gpu_name',''):
            failures.append(path.parent.name+':population_or_gpu_binding')
        if not all(result.get(k) is True for k in ('pass_functional','stale_graph_never_replayed',
                   'unchanged_capture_across_three_metadata_epochs','unannounced_inference_tensor_write_detected',
                   'changed_ordered_geometry_native')):
            failures.append(path.parent.name+':functional_contract')
        if len(result.get('epochs', [])) != 3:
            failures.append(path.parent.name+':incomplete_epochs')
        count = result.get('graph_replays_per_call')
        if count not in (1,16):
            failures.append(path.parent.name+':graph_count')
            continue
        for index,row in enumerate(result.get('epochs', []),1):
            trace_path = path.parent/f'epoch{index}-trace.json'
            if not trace_path.exists() or sha(trace_path) != row['trace_sha256']:
                failures.append(path.parent.name+':trace_hash')
                continue
            events = json.loads(trace_path.read_text())['traceEvents']
            kernels = [e for e in events if e.get('cat') == 'kernel' and 'BatchPrefillWithPagedKV' in e.get('name','')]
            if (len(kernels) != 4*count or any(e.get('args',{}).get('shared memory') != 65536 for e in kernels)
                    or row.get('output_lse_exact') is not True or row.get('native_after_resource_exact') is not True):
                failures.append(path.parent.name+':actual_launch_or_parity')
        if result.get('probe',{}).get('resource_graph_serving_qualified') is not False:
            failures.append(path.parent.name+':authority_widening')
    sass = root/f'sass-{job}'/'receipt.json'
    if not sass.exists() or json.loads(sass.read_text()).get('pass') is not True:
        failures.append('independent_sass_missing_or_failed')
    for stage in ('helper','native','sdk'):
        for phase in ('pre','post'):
            p = root/'receipts'/f'{stage}-{phase}-{job}.txt'
            if not p.exists() or not p.read_text().strip() or any(not line.endswith(': OK') for line in p.read_text().splitlines()):
                failures.append(stage+'_'+phase+'_source_check')
    return {'pass':not failures, 'job':job, 'gpu':gpu, 'cells':len(cells), 'failures':failures,
            'scope':'Uncertified fixed-geometry Graph metadata minimal driver only',
            'qualification_authority':False, 'default_promotion':False, 'serving_promotion':False,
            'historical_token_divergence_resolved':False}


def run(root, source):
    root = root.resolve()
    receipt_path = root/'receipts/controller.json'
    if receipt_path.exists():
        raise FileExistsError('Keep the first finite controller; never duplicate submissions')
    binding = json.loads((root/'binding.json').read_text())
    state = {'state':'SUBMITTING', 'terminal':False, 'jobs':{}, 'qualification_authority':False,
             'default_promotion':False, 'serving_promotion':False, 'historical_token_divergence_resolved':False}
    save(receipt_path,state)
    for gpu in ('gpu_4090','gpu_5090'):
        command = ['sbatch','--parsable','--partition='+gpu,
                   '--job-name=sgi-ge-'+binding['harness_commit'][:7]+'-'+gpu[4:],
                   '--output='+str(root/'logs/slurm-%j.log'),
                   str(root/'harness/research/selector_v4/exposed_diagnostics/graph_epoch.sbatch'),str(root),str(source)]
        intent = root/'receipts'/('submit-intent-'+gpu+'.json')
        save(intent,{'argv':command,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     'harness_commit':binding['harness_commit']})
        try:
            result = subprocess.run(command,capture_output=True,text=True,timeout=90)
            job = result.stdout.strip().split(';')[0]
            if result.returncode or not job.isdigit():
                raise RuntimeError('Unknown submission outcome: '+result.stdout+result.stderr)
        except Exception as error:
            state.update(state='UNKNOWN_SUBMISSION_OUTCOME_NO_RETRY',error=str(error))
            save(receipt_path,state)
            raise
        save(root/'receipts'/('submit-confirmed-'+gpu+'.json'),{'job':job,'stdout':result.stdout,'stderr':result.stderr})
        state['jobs'][gpu] = job
        save(receipt_path,state)
    state['state'] = 'WAIT_ALL_TWO_JOBS_TERMINAL'
    save(receipt_path,state)
    deadline = time.monotonic()+6*3600
    records = {}
    while True:
        for job in state['jobs'].values():
            result = subprocess.run(['sacct','-j',job,'--parsable2','--noheader','--format=JobIDRaw,State,ExitCode'],
                                    capture_output=True,text=True,timeout=30,check=True)
            rows = [line.split('|') for line in result.stdout.splitlines() if line.startswith(job+'|')]
            if rows:
                records[job] = {'state':rows[0][1],'exit':rows[0][2]}
        if len(records)==2 and all(r['state'].split()[0] not in ('PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED') for r in records.values()):
            break
        if time.monotonic()>deadline and not state.get('deadline_owned_jobs_cancelled'):
            for job in state['jobs'].values():
                subprocess.run(['scancel',job],capture_output=True,text=True,check=False)
            state['deadline_owned_jobs_cancelled'] = True
        if time.monotonic()>deadline+1200:
            state.update(state='TERMINAL_STATE_UNKNOWN_ARCHIVE_BLOCKED',terminal=True,slurm=records)
            save(receipt_path,state)
            return
        state['slurm'] = records
        save(receipt_path,state)
        time.sleep(30)
    state['slurm'] = records
    analyses = {}
    for gpu,job in state['jobs'].items():
        analyses[gpu] = validate(root,job,gpu)
        if records[job] != {'state':'COMPLETED','exit':'0:0'}:
            analyses[gpu]['pass'] = False
            analyses[gpu]['failures'].append('slurm_not_completed_zero')
        exit_path = root/'receipts'/f'exit-{job}.txt'
        if not exit_path.exists() or exit_path.read_text().strip() != '0':
            analyses[gpu]['pass'] = False
            analyses[gpu]['failures'].append('shell_exit_not_zero')
    state['analyses'] = analyses
    state['state'] = 'ARCHIVING_COMPLETE_DIAGNOSTIC'
    save(receipt_path,state)
    files = [p for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
             and p.name not in ('raw-complete.tar.gz','archive-members.json','verification-receipt.json')
             and (not p.is_relative_to(root/'cache') or p.suffix in ('.so','.cu','.cuh','.h','.hpp','.inc','.json','.txt'))]
    member_index = {str(p.relative_to(root)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}
    save(root/'archive-members.json',member_index)
    archive = root/'raw-complete.tar.gz'
    with tarfile.open(archive,'x:gz') as t:
        for p in files:t.add(p,arcname=str(p.relative_to(root)),recursive=False)
    with tarfile.open(archive,'r|gz') as t:
        seen=set()
        for member in t:
            with t.extractfile(member) as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            assert member_index[member.name]=={'sha256':actual,'bytes':member.size}
            seen.add(member.name)
        assert seen==set(member_index)
    proof={'pass':True,'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'verified_members':len(seen)}
    save(root/'verification-receipt.json',proof)
    state.update(terminal=True,state='DUAL_GRAPH_EPOCH_DIAGNOSTIC_PASS' if all(a['pass'] for a in analyses.values()) else 'GRAPH_EPOCH_DIAGNOSTIC_HOLD',archive=proof)
    save(receipt_path,state)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('source',type=Path)
    args=parser.parse_args()
    run(args.root,args.source)
