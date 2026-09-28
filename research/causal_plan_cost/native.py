"""Export/validate native lowering without using timing labels or refitting.

The pickle inputs must be our own hash-checked training artifacts. Never load
untrusted pickle files. Native overhead includes input-array conversion.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import pickle
import random
import statistics
import subprocess
import time
import sysconfig
import importlib.util
import numpy as np
from geometry import Shape, POLICIES, corpus, features


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def number(x):return format(float(x),'.17g')


def export(analysis,out):
    out.mkdir(parents=True,exist_ok=False)
    selection=analysis/'model_selection.json'
    freeze=json.loads((analysis/'freeze.json').read_text())
    if sha(selection)!=freeze['model_selection_sha256']:raise ValueError('Modified freeze')
    data=json.loads(selection.read_text());code=[];models=[];mapping={}
    for key,info in sorted(data['models'].items()):
        if not key.endswith('|causal'):continue
        p=analysis/info['file']
        if sha(p)!=info['sha256']:raise ValueError('Modified training model')
        model=pickle.loads(p.read_bytes())
        index=len(models);name=f'M{index}';mapping[key]=index
        if info['choice']['spec']['class']=='ridge':
            scaler,ridge=model.steps[0][1],model.steps[1][1]
            weights=ridge.coef_/scaler.scale_
            bias=ridge.intercept_-np.dot(weights,scaler.mean_)
            if len(weights)!=35:raise ValueError('Unexpected feature schema')
            code.append(f'static const double {name}W[]={{'+','.join(map(number,weights))+'};')
            models.append(f'{{0,{name}W,{number(bias)},nullptr,nullptr,0}}')
        else:
            nodes=[];roots=[]
            for estimator in model.estimators_:
                tree=estimator.tree_;offset=len(nodes);roots.append(offset)
                for i in range(tree.node_count):
                    left=int(tree.children_left[i]);right=int(tree.children_right[i])
                    nodes.append('{'+','.join((str(left+offset if left>=0 else -1),
                        str(right+offset if right>=0 else -1),str(int(tree.feature[i])),
                        number(tree.threshold[i]),number(tree.value[i,0,0])))+'}')
            code.append(f'static const Node {name}N[]={{'+','.join(nodes)+'};')
            code.append(f'static const int {name}R[]={{'+','.join(map(str,roots))+'};')
            models.append(f'{{1,nullptr,0,{name}N,{name}R,{len(roots)}}}')
    code.append('static const Model MODELS[]={'+','.join(models)+'};')
    code.append(f'constexpr int MODEL_COUNT={len(models)};')
    (out/'models.inc').write_text('\n'.join(code)+'\n')
    source=Path(__file__).with_name('native_selector.cpp')
    command=['g++','-O3','-std=c++17','-shared','-fPIC','-I',str(out),str(source),'-o',str(out/'libselector.so')]
    completed=subprocess.run(command,capture_output=True,text=True,check=True)
    extension=out/('geometry_native'+sysconfig.get_config_var('EXT_SUFFIX'))
    bridge_command=['g++','-O3','-std=c++17','-shared','-fPIC','-DBUILD_PYTHON_BRIDGE',
        '-I',sysconfig.get_paths()['include'],'-I',str(out),str(source),'-o',str(extension)]
    subprocess.run(bridge_command,capture_output=True,text=True,check=True)
    manifest={'extension':extension.name,'extension_sha256':sha(extension),
              'bridge_compile_command':bridge_command,'models':mapping,'model_selection_sha256':sha(selection),
              'source_sha256':sha(source),'exporter_sha256':sha(__file__),
              'library_sha256':sha(out/'libselector.so'),'header_sha256':sha(out/'models.inc'),
              'compile_command':command,'compiler_stderr':completed.stderr,
              'changed_weights':False,'training_performed':False,'test_timings_used':False}
    save(out/'manifest.json',manifest);print('NATIVE_EXPORTED',manifest,flush=True)


class Native:
    def __init__(self,path):
        self.path=Path(path);self.info=json.loads((self.path/'manifest.json').read_text())
        libpath=self.path/'libselector.so'
        if sha(libpath)!=self.info['library_sha256']:raise ValueError('Modified native library')
        extension=self.path/self.info['extension']
        if sha(extension)!=self.info['extension_sha256']:raise ValueError('Modified bridge')
        spec=importlib.util.spec_from_file_location('geometry_native',extension)
        self.bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.bridge)
        self.lib=ct.CDLL(str(libpath.resolve()))
        ptr=ct.POINTER(ct.c_int64);out=ct.POINTER(ct.c_double)
        self.lib.select_policy.argtypes=[ct.c_int,ct.c_int,ptr,ptr,ct.c_int,ct.c_int,ct.c_int,out]
        self.lib.select_policy.restype=ct.c_int
        self.lib.causal_features.argtypes=[ct.c_int,ptr,ptr,ct.c_int,ct.c_int,ct.c_int,ct.c_int,out]
        self.lib.causal_features.restype=ct.c_int
    def choose(self,key,shape,hq,hkv,sms,return_scores=False):
        if not return_scores:
            i=self.bridge.choose(self.info['models'][key],shape.q,shape.k,hq,hkv,sms)
            return POLICIES[i]
        q=np.ascontiguousarray(shape.q,dtype=np.int64);k=np.ascontiguousarray(shape.k,dtype=np.int64)
        scores=np.empty(6,dtype=np.float64)
        i=self.lib.select_policy(self.info['models'][key],len(q),
            q.ctypes.data_as(ct.POINTER(ct.c_int64)),k.ctypes.data_as(ct.POINTER(ct.c_int64)),
            hq,hkv,sms,scores.ctypes.data_as(ct.POINTER(ct.c_double)))
        if i<0:raise ValueError(f'Native input rejected: {i}')
        return (POLICIES[i],scores) if return_scores else POLICIES[i]
    def feature_vector(self,shape,hq,hkv,sms,policy):
        q=np.ascontiguousarray(shape.q,dtype=np.int64);k=np.ascontiguousarray(shape.k,dtype=np.int64)
        x=np.empty(35,dtype=np.float64)
        status=self.lib.causal_features(len(q),q.ctypes.data_as(ct.POINTER(ct.c_int64)),
            k.ctypes.data_as(ct.POINTER(ct.c_int64)),hq,hkv,sms,POLICIES.index(policy),
            x.ctypes.data_as(ct.POINTER(ct.c_double)))
        if status:raise ValueError(f'Native feature input rejected: {status}')
        return x


def validate(analysis,out):
    native=Native(out);sel=json.loads((analysis/'model_selection.json').read_text())
    if sha(analysis/'model_selection.json')!=native.info['model_selection_sha256']:
        raise ValueError('Different frozen selection')
    shapes=corpus();rng=random.Random(20260928)
    for i in range(100):
        n=rng.randrange(1,9)
        shapes.append(Shape(f'input-contract-{i}','input-contract','diagnostic',
            tuple(rng.randrange(1,1025) for _ in range(n)),
            tuple(rng.randrange(0,32769) for _ in range(n))))
    result={'input_shapes':len(shapes),'original_corpus_shapes':len(corpus()),
            'random_contract_shapes':100,'test_timing_labels_read':False,'models':{},
            'equivalence_tolerance':1e-10,'feature_vectors_checked':0}
    for key in sorted(native.info['models']):
        path=analysis/sel['models'][key]['file']
        if sha(path)!=sel['models'][key]['sha256']:raise ValueError('Modified model')
        model=pickle.loads(path.read_bytes());sms=128 if '4090' in key else 170
        errors=[];timings=[];mismatches=[];feature_error=0.
        for i,shape in enumerate(shapes):
            hq,hkv=((16,4),(32,8))[i%2]
            x=np.asarray([features(shape,hq,hkv,sms,p,'causal') for p in POLICIES])
            for j,p in enumerate(POLICIES):
                actual=native.feature_vector(shape,hq,hkv,sms,p)
                np.testing.assert_allclose(actual,x[j],atol=1e-11,rtol=1e-11)
                feature_error=max(feature_error,float(abs(actual-x[j]).max()))
                result['feature_vectors_checked']+=1
            scores=model.predict(x);p,actual=native.choose(key,shape,hq,hkv,sms,True)
            error=float(abs(scores-actual).max());errors.append(error)
            np.testing.assert_allclose(actual,scores,atol=1e-10,rtol=1e-10)
            if p!=POLICIES[int(np.argmin(scores))] or native.choose(key,shape,hq,hkv,sms)!=p:mismatches.append(shape.name)
            # Includes metadata conversion and FFI; no timing label access.
            for _ in range(5):
                start=time.perf_counter_ns();native.choose(key,shape,hq,hkv,sms)
                timings.append((time.perf_counter_ns()-start)/1000)
        result['models'][key]={'max_log_prediction_difference':max(errors),
            'max_feature_difference':feature_error,'policy_mismatches':mismatches,
            'selector_microseconds_p50':statistics.median(timings),
            'selector_microseconds_p95':float(np.quantile(timings,.95))}
    result['all_policies_identical']=all(not v['policy_mismatches'] for v in result['models'].values())
    save(out/'validation.json',result);print('NATIVE_VALIDATION',json.dumps(result),flush=True)
    if not result['all_policies_identical']:raise SystemExit(2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['export','validate'])
    p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();{'export':export,'validate':validate}[a.command](a.analysis,a.out)
