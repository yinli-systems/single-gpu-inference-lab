"""Evidence integrity and honest promotion decisions. Pure CPU, no runtime mutation."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path

def first_difference(a,b):
    for xs in (a,b):
        if not isinstance(xs,list) or any(type(v) is not int or v<0 for v in xs):
            raise ValueError('invalid token IDs')
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)) if len(a)!=len(b) else None)

def file_hash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_verified(path):
    path=Path(path);complete=json.loads((path.parent/'complete.json').read_text())
    if complete.get('complete') is not True:raise ValueError('incomplete artifact')
    files=complete.get('files',{})
    if path.name not in files or files[path.name]!=file_hash(path):raise ValueError('unbound/corrupt file')
    for name,h in files.items():
        if Path(name).name!=name or name in ('.','..'):raise ValueError('unsafe artifact member')
        if file_hash(path.parent/name)!=h:raise ValueError('artifact member hash mismatch')
    return json.loads(path.read_text())

def validate_requests(batch):
    rows=batch['requests']
    if batch.get('errors') or not rows:raise ValueError('failed/empty request set')
    if len({x['id'] for x in rows})!=len(rows):raise ValueError('duplicate request ID')
    for r in rows:
        first_difference(r['tokens'],r['tokens'])
        if not r['tokens']:raise ValueError('empty output')
    elapsed=batch['elapsed']
    if not math.isfinite(elapsed) or elapsed<=0:raise ValueError('invalid elapsed seconds')
    count=sum(len(r['tokens']) for r in rows)
    if count!=batch['output_tokens']:raise ValueError('token numerator mismatch')
    if not math.isclose(count/elapsed,batch['output_tokens_per_second'],rel_tol=1e-10):
        raise ValueError('wrong throughput units or numerator')
    return {r['id']:r['tokens'] for r in rows}

def compare_batches(base,candidate):
    if base['workload_sha256']!=candidate['workload_sha256']:raise ValueError('workload changed')
    b=validate_requests(base);c=validate_requests(candidate)
    if set(b)!=set(c):raise ValueError('request identity mismatch')
    mismatches=[]
    for rid in sorted(b):
        pos=first_difference(b[rid],c[rid])
        if pos is not None:mismatches.append(dict(id=rid,first_difference_zero_based=pos,
            lengths=[len(b[rid]),len(c[rid])]))
    return dict(candidate_requests=len(c),mismatches=mismatches,exact_parity=not mismatches)

def release_gate(*,token_parity,all_controls_resolve,all_supported_paths_qualified,
                 worst_ratio_lower_bound,simultaneous_bound,net_gain_lower_bound,
                 independent_validation,old_failure_attributed):
    values=(worst_ratio_lower_bound,net_gain_lower_bound)
    if any(type(v) not in (float,int) or not math.isfinite(v) or v<=0 for v in values):
        raise ValueError('nonfinite/invalid bound')
    checks=dict(token_parity=token_parity,all_controls_resolve=all_controls_resolve,
        supported_paths=all_supported_paths_qualified,simultaneous_bound=simultaneous_bound,
        independent_validation=independent_validation,old_failure_attributed=old_failure_attributed,
        bounded_regression=worst_ratio_lower_bound>=1/1.01,net_gain=net_gain_lower_bound>1)
    if any(type(v) is not bool for v in checks.values()):raise ValueError('unknown gate state')
    failures=[k for k,v in checks.items() if not v]
    return dict(promote=not failures,failed_gates=failures,decision='HOLD' if failures else 'ELIGIBLE_FOR_REVIEW')
