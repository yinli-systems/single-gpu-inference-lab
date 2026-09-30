"""Authorize v4 release qualification only after dual-GPU canary PASS."""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(value,message):
    if not value:raise RuntimeError(message)
def run(a):
    root=Path(a.root);reports={}
    for gpu,path in [('gpu_4090',a.gpu_4090),('gpu_5090',a.gpu_5090)]:
        data=json.loads(Path(path).read_text())
        need(data.get('gpu')==gpu and data.get('stage')=='canary' and data.get('pass') is True,'canary HOLD '+gpu)
        need(data.get('requirements',{}).get('numerical_exact') is True,'numerics '+gpu)
        reports[gpu]={'sha256':sha(path),'provenance':data['provenance'],'chosen_worst':data['chosen_worst'],'chosen_joint_lcb':data['chosen_joint_lcb'],'max_regret':data['max_regret'],'p95_regret':data['p95_regret'],'cap_geomean':data['cap_geomean'],'cap_ci95':data['cap_ci95']}
    p0=reports['gpu_4090']['provenance'];p1=reports['gpu_5090']['provenance']
    for key in ('case_hash','release_hash','source_archive_sha256','overlay_sha256','measurement_revision'):
        need(p0[key]==p1[key], 'dual-GPU provenance '+key)
    audits=list((root/'receipts').glob('binary-audit-*.json'))
    need(len(audits)>=4,'missing per-shard binary audits')
    audit_receipts=[]
    for path in audits:
        data=json.loads(path.read_text());need(data.get('kernel_symbol_isolation_compiled') is True,'binary audit '+path.name);audit_receipts.append({'path':path.name,'sha256':sha(path)})
    release_dirs=list((root/'runs').glob('release-*'));need(not release_dirs,'release cases already consumed')
    result={'schema':1,'release_qualification_authorized':True,'canary_reports':reports,'binary_audits':audit_receipts,'case_hash':p0['case_hash'],'release_hash':p0['release_hash'],'source_archive_sha256':p0['source_archive_sha256'],'overlay_sha256':p0['overlay_sha256'],'release_cases_consumed_at_authorization':0,'default_promotion':False,'serving_promotion':False,'historical_token_divergence_resolved':False}
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu-4090',type=Path,required=True);p.add_argument('--gpu-5090',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
