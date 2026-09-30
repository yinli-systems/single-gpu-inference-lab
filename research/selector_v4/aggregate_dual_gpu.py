"""Fail-closed dual-GPU release authorization; serving remains separately blocked."""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(a):
    reports={}
    for gpu,path in [('gpu_4090',a.gpu_4090),('gpu_5090',a.gpu_5090)]:
        data=json.loads(Path(path).read_text())
        if data.get('gpu')!=gpu or data.get('stage')!='release' or data.get('pass') is not True:raise RuntimeError('GPU release HOLD '+gpu)
        reports[gpu]={'sha256':sha(path),'chosen_worst':data['chosen_worst'],'chosen_joint_lcb':data['chosen_joint_lcb'],'max_regret':data['max_regret'],'p95_regret':data['p95_regret'],'cap_geomean':data['cap_geomean'],'cap_ci95':data['cap_ci95'],'provenance':data['provenance']}
    if reports['gpu_4090']['provenance']['case_hash']!=reports['gpu_5090']['provenance']['case_hash'] or reports['gpu_4090']['provenance']['release_hash']!=reports['gpu_5090']['provenance']['release_hash']:raise RuntimeError('dual-GPU manifest drift')
    result={'schema':1,'performance_release_pass':True,'reports':reports,'default_promotion':False,'serving_promotion':False,'historical_token_divergence_resolved':False,'next_gate':'paired full HTTP with complete token parity'}
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu-4090',type=Path,required=True);p.add_argument('--gpu-5090',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
