"""Source-bound dual-GPU smoke gate, before consuming any holdout geometry."""
import argparse
import hashlib
import json
from pathlib import Path
from .schema import QUALIFICATION_REVISION


def qualify(root, verify_only=False):
    root=Path(root);receipt=root/'receipts/smoke-verdict.json'
    campaign=json.loads((root/'receipts/campaign.json').read_text())
    if verify_only:
        result=json.loads(receipt.read_text())
        if result.get('pass') is not True or result.get('qualification_revision') != QUALIFICATION_REVISION or result.get('source_archive_sha256') != campaign['source_archive_sha256']:
            raise RuntimeError('smoke qualification missing or source drift')
        for rel,expected in result['artifacts'].items():
            if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=expected:raise RuntimeError('smoke artifact drift '+rel)
        return result
    jobs=json.loads((root/'receipts/smoke-submit.json').read_text());artifacts={}
    def read(path):
        artifacts[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    for gpu,job in jobs.items():
        pristine=read(root/f'runs/{gpu}/pristine-{job}/results.json')
        candidate=read(root/f'runs/{gpu}/candidate-{job}/results.json')
        binary=read(root/f'receipts/binary-audit-{job}.json')
        if not pristine.get('complete') or not candidate.get('complete') or not binary.get('native_sass_identity') or not binary.get('kernel_symbol_isolation_compiled'):
            raise RuntimeError('smoke HOLD '+gpu)
        refs={(x['layout'],x['dtype']):x for x in pristine['results']}
        checked=set()
        for item in candidate['results']:
            if item['label'] not in ('native_before','cap','native_after'):continue
            key=item['layout'],item['dtype'];ref=refs[key]
            if item['out_sha256'] != ref['out_sha256'] or item['lse_sha256'] != ref['lse_sha256'] or item['plan_info'] != ref['plan_info']:
                raise RuntimeError('smoke numerical/plan identity failed')
            checked.add((*key,item['label']))
        if len(checked)!=12:raise RuntimeError('incomplete dtype/layout/sequence smoke')
    if set(jobs)!={'gpu_4090','gpu_5090'}:raise RuntimeError('dual GPU required')
    result={'pass':True,'qualification_revision':QUALIFICATION_REVISION,'source_archive_sha256':campaign['source_archive_sha256'],
            'jobs':jobs,'artifacts':artifacts,'authorized_stage':'dev','canary_cases_consumed':0,
            'release_cases_consumed':0,'stress_cases_consumed':0,'default_promotion':False,'serving_promotion':False}
    receipt.write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--verify-only',action='store_true')
    print(json.dumps(qualify(**vars(p.parse_args())),indent=2))
