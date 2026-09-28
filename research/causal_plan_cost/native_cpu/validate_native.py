"""Validate frozen decisions and measure CPU selection overhead, not LLM serving.

The CPU models/decisions are unchanged from the frozen evaluation. GPU timings
are reused only in explicitly additive estimates, never as new GPU measurements.
"""
from __future__ import annotations
import argparse, ctypes, hashlib, importlib, json, math, pathlib, pickle, platform, socket, sys, time
import numpy as np
from export_native import Native, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection-dir',type=pathlib.Path,required=True)
    p.add_argument('--native-dir',type=pathlib.Path,required=True)
    p.add_argument('--geometry-root',type=pathlib.Path,required=True)
    p.add_argument('--out',type=pathlib.Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Results are create-only')
    sys.path.insert(0,str(a.geometry_root.resolve()))
    from geometry import POLICIES,corpus,features
    from analyze import policy_metrics
    selection=json.loads((a.selection_dir/'model_selection.json').read_text())
    freeze=json.loads((a.selection_dir/'freeze.json').read_text())
    if sha(a.selection_dir/'model_selection.json')!=freeze['model_selection_sha256']:
        raise ValueError('selection changed')
    manifest=json.loads((a.native_dir/'export_manifest.json').read_text())
    if manifest['selection_sha256']!=freeze['model_selection_sha256']:
        raise ValueError('model export selection mismatch')
    decisions=json.loads((a.selection_dir/'test_decisions.json').read_text())
    ss=corpus();byname={s.name:s for s in ss}
    report={'scope':'CPU decision-equivalence and overhead; additive GPU-cycle estimates ONLY',
        'gpu_jobs_submitted':0,'refitting_performed':False,'policy_changed':False,
        'selection_sha256':freeze['model_selection_sha256'],
        'cpu_host':socket.gethostname(),'cpu_platform':platform.platform(),
        'validation_script_sha256':sha(__file__),
        'export_manifest_sha256':sha(a.native_dir/'export_manifest.json'),
        'model_results':[],'native_timing_rows':[]}
    for exported in manifest['models']:
        key=exported['key'];gpu,target,kind=key.split('|')
        original=selection['models'][key];path=a.selection_dir/original['file']
        lib=a.native_dir/exported['library']
        if sha(path)!=original['sha256'] or sha(lib)!=exported['library_sha256']:
            raise ValueError('frozen model or compiled library changed')
        model=pickle.loads(path.read_bytes());native=Native(lib)
        checked=0;maxfeature=0.;maxpred=0.;matches=0;checks=[]
        for s in ss:
            for hq,hkv in ((16,4),(32,8)):
                for sms in (128,170):
                    xb=[];qb,kb,out=native.buffers(s.q,s.k)
                    for policy_index,policy in enumerate(POLICIES):
                        b=(ctypes.c_double*40)()
                        length=native.feat(qb,kb,len(s.q),hq,hkv,sms,policy_index,b)
                        if length<0:raise ValueError('feature call rejected checked geometry')
                        actual=np.array(b)[:length];expected=np.array(features(s,hq,hkv,sms,policy,kind))
                        np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-13)
                        maxfeature=max(maxfeature,float(np.max(np.abs(actual-expected))));xb.append(expected)
                    ref=model.predict(np.array(xb));choice,pred=native.choose(s.q,s.k,hq,hkv,sms)
                    np.testing.assert_allclose(pred,ref,rtol=1e-11,atol=1e-11)
                    maxpred=max(maxpred,float(np.max(np.abs(pred-ref))));checked+=1
                    if choice!=int(np.argmin(ref)):
                        raise AssertionError(f'Decision mismatch: {key} {s.name} {hq} {sms}')
                    matches+=1
        relevant=[d for d in decisions if d['gpu']==gpu and d['target']==target and d['kind']==kind]
        inclusive=[];cached=[];cost=[];reference=[];oracle=[];families=[]
        sms=128 if '4090' in gpu else 170
        for row in relevant:
            s=byname[row['shape']];hq=row['hq'];hkv=hq//4
            qb,kb,out=native.buffers(s.q,s.k)
            for _ in range(5):native.choose(s.q,s.k,hq,hkv,sms)
            inc=[];cache=[]
            for block in range(21):
                tick=time.perf_counter_ns()
                for _ in range(5):idx,_=native.choose(s.q,s.k,hq,hkv,sms)
                inc.append((time.perf_counter_ns()-tick)/5e6)
                tick=time.perf_counter_ns()
                for _ in range(5):native.fn(qb,kb,len(s.q),hq,hkv,sms,out)
                cache.append((time.perf_counter_ns()-tick)/5e6)
            if POLICIES[idx]!=row['policy']:
                raise AssertionError('Frozen test decision changed')
            med=float(np.median(inc));p95=float(np.quantile(inc,.95))
            inclusive.append(med);cached.append(float(np.median(cache)))
            cost.append(row['cost_ms']);reference.append(row['fixed_cost_ms'])
            oracle.append(row['oracle_ms']);families.append(row['family'])
            report['native_timing_rows'].append(dict(key=key,shape=s.name,hq=hq,policy=row['policy'],
                inclusive_blocks_ms=inc,cached_buffer_blocks_ms=cache,native_median_ms=med,
                native_p95_ms=p95,cost_ms=row['cost_ms'],fixed_ms=row['fixed_cost_ms'],
                family=row['family'],oracle_ms=row['oracle_ms']))
        result={'key':key,'metadata_cells_compared':checked,'feature_values_compared':checked*6*len(xb[0]),
            'decisions_equal':matches,'max_abs_feature_difference':maxfeature,
            'max_abs_prediction_difference':maxpred,'test_cells':len(relevant),
            'selection_ms_median_inclusive':float(np.median(inclusive)),
            'selection_ms_p95_across_cells':float(np.quantile(inclusive,.95)),
            'selection_ms_median_existing_buffers':float(np.median(cached)),
            'raw_decision_performance':policy_metrics(cost,reference,families,oracle),
            'additive_inclusive_median':policy_metrics(np.array(cost)+inclusive,reference,families,oracle),
            'additive_per_case_p95':policy_metrics(np.array(cost)+[
                r['native_p95_ms'] for r in report['native_timing_rows'] if r['key']==key],reference,families,oracle)}
        report['model_results'].append(result)
        print('NATIVE_VALIDATED',json.dumps(result),flush=True)
    with a.out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print('RESULT_SHA256',sha(a.out),flush=True)

if __name__=='__main__':main()
