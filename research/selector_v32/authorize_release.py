"""Fail-closed release preflight. This tool never submits a job."""
from __future__ import annotations
import argparse,contextlib,io,json,tempfile
from pathlib import Path
from types import SimpleNamespace
from manifest import load
from measurement_contract import REVISION

def authorize(root: Path) -> dict:
    from analyze import run
    root=Path(root)
    archive=(root/'receipts'/'source-archive.sha256').read_text().strip()
    overlay=(root/'receipts'/'official-overlay.sha256').read_text().strip()
    evidence={}
    for gpu in ('gpu_4090','gpu_5090'):
        for mode in ('pristine','paired'):
            paths=list((root/'runs').glob(f'canary-{gpu}-s0-r0-{mode}-*'))
            if len(paths)!=1 or not (paths[0]/'complete.json').exists():
                raise RuntimeError('Missing unique completed current-revision canary: '+gpu+' '+mode)
        # Recompute from immutable raw evidence, not a hand-edited PASS JSON.
        with tempfile.TemporaryDirectory(prefix='sgi-release-preflight-') as tmp:
            out=Path(tmp)/'analysis'
            with contextlib.redirect_stdout(io.StringIO()):
                run(SimpleNamespace(root=root,out=out,stage='canary',gpu=gpu,shards=1))
            summary=json.loads((out/'summary.json').read_text())
        if summary.get('measurement_contract_revision')!=REVISION:
            raise RuntimeError('Canary measurement revision mismatch')
        if summary.get('source_archive_sha256')!=archive or summary.get('official_overlay_sha256')!=overlay:
            raise RuntimeError('Canary campaign identity mismatch')
        if summary.get('case_hash')!=load()['case_hash'] or summary.get('canary_gate',{}).get('pass') is not True:
            raise RuntimeError('Canary HOLD: '+gpu)
        evidence[gpu]=dict(hardware=summary['hardware'],requirements=summary['canary_gate']['requirements'])
    return dict(release_qualification_authorized=True,measurement_contract_revision=REVISION,case_hash=load()['case_hash'],source_archive_sha256=archive,official_overlay_sha256=overlay,canaries=evidence,default_promotion=False,serving_promotion=False,historical_token_divergence_resolved=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    print(json.dumps(authorize(p.parse_args().root),indent=2))
