"""Metadata-only parity on fresh geometry. No test timings or refitting."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from contracts import fresh_corpus,corpus_hash,POLICIES
from geometry import Shape,features
from native import Native,sha

p=argparse.ArgumentParser()
p.add_argument('--analysis',type=Path,required=True);p.add_argument('--native',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.out.exists():raise FileExistsError('preserve prior qualification')
selection=a.analysis/'model_selection.json';meta=json.loads(selection.read_text())
if sha(selection)!=json.loads((a.analysis/'freeze.json').read_text())['model_selection_sha256']:raise ValueError('changed frozen selection')
native=Native(a.native)
if native.info['model_selection_sha256']!=sha(selection):raise ValueError('wrong native model')
report=dict(kind='fresh_metadata_parity_only',corpus_sha256=corpus_hash(),timing_labels_read=False,
            model_selection_sha256=sha(selection),native_manifest_sha256=sha(a.native/'manifest.json'),models={})
for key in native.info['models']:
    info=meta['models'][key];f=a.analysis/info['file']
    if sha(f)!=info['sha256']:raise ValueError('untrusted/changed local fitted model')
    model=pickle.loads(f.read_bytes());sms=128 if '4090' in key else 170
    maxf=maxp=0.;same=0
    for r in fresh_corpus():
        s=Shape(r['name'],r['family'],r['split'],tuple(r['q']),tuple(r['k']))
        X=np.asarray([features(s,32,8,sms,p,'causal') for p in POLICIES])
        for i,mode in enumerate(POLICIES):
            actual=native.feature_vector(s,32,8,sms,mode)
            np.testing.assert_allclose(actual,X[i],atol=1e-11,rtol=1e-11)
            maxf=max(maxf,float(np.max(np.abs(actual-X[i]))))
        expected=model.predict(X);mode,actual=native.choose(key,s,32,8,sms,True)
        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10)
        maxp=max(maxp,float(np.max(np.abs(actual-expected))))
        if mode!=POLICIES[int(np.argmin(expected))] or native.choose(key,s,32,8,sms)!=mode:raise AssertionError('changed fresh decision')
        same+=1
    report['models'][key]=dict(choices_equal=same,max_feature_error=maxf,max_prediction_error=maxp)
report['passed']=True
report['source_sha256']=sha(__file__)
a.out.write_text(json.dumps(report,indent=2)+'\n')
print('FRESH_NATIVE_PARITY',json.dumps(report))
