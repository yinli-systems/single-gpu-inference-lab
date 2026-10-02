"""Verify persisted CPU/source/dispatch bytes; does not grant GPU authority."""
import hashlib
import json
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path


def sha(value): return hashlib.sha256(value).hexdigest()
def verify(root):
    index=json.loads((root/'cpu-archive-members.json').read_text())
    with tarfile.open(root/'raw-cpu-preflight.tar.gz') as archive:
        members=archive.getmembers()
        assert all(m.isfile() for m in members) and len(members)==len(index)
        raw={m.name:archive.extractfile(m).read() for m in members}
    assert set(raw)==set(index)
    for name,value in raw.items(): assert index[name]=={'sha256':sha(value),'bytes':len(value)},name
    cpu=json.loads(raw['cpu-receipt.json'])
    for name,digest in cpu['files'].items():assert sha(raw[name])==digest,name
    for name in ('cpu-receipt.json','cpu-tests.log','cpu-tests.xml','cpu-environment.json','cpu-exit.txt'):
        assert (root/name).read_bytes()==raw[name],name
    cases=ET.fromstring(raw['cpu-tests.xml']).findall('.//testcase')
    assert len(cases)==29 and sum(c.find('skipped') is not None for c in cases)==16
    assert all(c.find('failure') is None and c.find('error') is None for c in cases)
    assert cpu['cpu_passed']==13 and cpu['gpu_skipped']==16 and cpu['gpu_qualification'] is False
    packet=json.loads((root/'source-packet.json').read_text())
    assert sha((root/'source.tar.gz').read_bytes())==packet['archive_sha256']==cpu['harness_archive_sha256']
    with tarfile.open(root/'source.tar.gz') as archive:
        binding=json.load(archive.extractfile('harness-binding.json'))
        assert len(archive.getmembers())==11
        for name,digest in binding['source_files'].items():assert sha(archive.extractfile(name).read())==digest,name
    assert binding['harness_commit']==cpu['harness_commit']==packet['harness_commit']
    intent=json.loads((root/'dispatch-intent.json').read_text())
    assert sha((root/'dispatch-driver.py').read_bytes())==intent['dispatch_driver_sha256']
    assert sha(raw['cpu-receipt.json'])==intent['cpu_receipt_sha256']
    for gpu in ('gpu_4090','gpu_5090'):
        submit=json.loads((root/f'submit-confirmed-{gpu}.json').read_text())
        assert submit['job'].isdigit() and not submit['stderr']
        assert submit['job']==json.loads((root/'dispatch-snapshot.json').read_text())['diagnostic_controller']['jobs'][gpu]
    with tarfile.open(root/'preparation-failure-35be3b3.tar.gz') as archive:
        failure=json.load(archive.extractfile('preparation-failure-35be3b3/receipt.json'))
        for name,digest in failure['files'].items():assert sha(archive.extractfile('preparation-failure-35be3b3/'+name).read())==digest
        assert failure['failed_before_gpu_submission'] and failure['original_files_preserved']
    return {'pass_persisted_cpu_source_dispatch_integrity':True,'cpu_passed':13,'gpu_skipped':16,
            'qualification_authority':False,'gpu_diagnostic_pass':None,'default_promotion':False,
            'serving_promotion':False,'historical_token_divergence_resolved':False}

if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parent),indent=2))
