"""Compile a frozen sklearn selector without changing its decisions or training.

Only hash-verified local models created by this project's analyze.py are loaded.
Generated native libraries are CPU-only. Never load a downloaded pickle blindly.
"""
from __future__ import annotations
import argparse, ctypes, hashlib, json, pathlib, pickle, subprocess
import numpy as np

HERE=pathlib.Path(__file__).resolve().parent
KINDS=('marginal','joint','plan','causal')

def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def number(x):
    x=float(x)
    if not np.isfinite(x): raise ValueError('nonfinite model coefficient')
    return format(x,'.17g')

def scorer(model):
    if hasattr(model,'steps'):
        scaler,reg=model.steps[0][1],model.steps[-1][1]
        w=np.asarray(reg.coef_,dtype=np.float64)
        return (f'constexpr int NF={len(w)};\n'+
            'static const double W[]={'+','.join(map(number,w))+'};\n'+
            'static const double MU[]={'+','.join(map(number,scaler.mean_))+'};\n'+
            'static const double SD[]={'+','.join(map(number,scaler.scale_))+'};\n'+
            f'double score(const double* x){{ double y={number(reg.intercept_)};'+
            'for(int i=0;i<NF;++i)y+=((x[i]-MU[i])/SD[i])*W[i];return y;}\n')
    roots=[];nodes=[]
    for estimator in model.estimators_:
        tree=estimator.tree_;offset=len(nodes);roots.append(offset)
        for i in range(tree.node_count):
            left=int(tree.children_left[i]);right=int(tree.children_right[i])
            nodes.append('{'+','.join((str(int(tree.feature[i])),str(left+offset if left>=0 else -1),
                str(right+offset if right>=0 else -1),number(tree.threshold[i]),
                number(tree.value[i].reshape(-1)[0])))+'}')
    return (f'constexpr int NF={model.n_features_in_};\n'+
        'struct Node { int feature,left,right;double threshold,value;};\n'+
        'static const Node NODES[]={'+',\n'.join(nodes)+'};\n'+
        'static const int ROOTS[]={'+','.join(map(str,roots))+'};\n'+
        'double score(const double* x){ float xx[NF];for(int i=0;i<NF;++i)xx[i]=float(x[i]);'+
        'double y=0.;for(int root:ROOTS){int j=root;while(NODES[j].left>=0)'+
        'j=double(xx[NODES[j].feature])<=NODES[j].threshold?NODES[j].left:NODES[j].right;'+
        f'y+=NODES[j].value;}}return y/{len(roots)}.;}}\n')

SUFFIX=r'''
extern "C" int choose_plan(const std::int64_t* q,const std::int64_t* k,int n,
                           int hq,int hkv,int sms,double* predictions) noexcept {
    try {
        if(!predictions)return -1;
        sgi::validate(q,k,n,hq,hkv,sms,KIND);
        double x[40];int choice=0;
        for(int p=0;p<6;++p){
            if(sgi::feature(q,k,n,hq,hkv,sms,KIND,p,x)!=NF)return -2;
            predictions[p]=score(x);
            if(!std::isfinite(predictions[p]))return -3;
            if(p && predictions[p]<predictions[choice])choice=p;
        }
        return choice;
    } catch(...) {return -4;}
}
extern "C" int feature_map(const std::int64_t* q,const std::int64_t* k,int n,
                           int hq,int hkv,int sms,int policy,double* output) noexcept {
    try {sgi::validate(q,k,n,hq,hkv,sms,KIND);return sgi::feature(q,k,n,hq,hkv,sms,KIND,policy,output);}
    catch(...){return -4;}
}
'''

def export(selection_dir,out,kinds=('causal',)):
    selection_dir=pathlib.Path(selection_dir);out=pathlib.Path(out)
    frozen=json.loads((selection_dir/'freeze.json').read_text())
    if sha(selection_dir/'model_selection.json')!=frozen['model_selection_sha256']:
        raise ValueError('selection freeze hash mismatch')
    selection=json.loads((selection_dir/'model_selection.json').read_text())
    out.mkdir(parents=True,exist_ok=False);receipt=[]
    for key,info in selection['models'].items():
        kind=key.rsplit('|',1)[1]
        if kind not in kinds:continue
        p=selection_dir/info['file']
        if sha(p)!=info['sha256']:raise ValueError('local model hash mismatch')
        model=pickle.loads(p.read_bytes())
        src=out/(p.stem+'.cpp');lib=out/(p.stem+'.so')
        src.write_text('#include "feature_core.hpp"\n'+f'constexpr int KIND={KINDS.index(kind)};\n'+scorer(model)+SUFFIX)
        cmd=['c++','-O3','-std=c++17','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',
             '-I',str(HERE),str(src),'-o',str(lib)]
        subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=120)
        receipt.append({'key':key,'model_sha256':info['sha256'],'source':src.name,'source_sha256':sha(src),
                        'library':lib.name,'library_sha256':sha(lib),'compile_command':cmd})
        print('NATIVE_COMPILED',key,flush=True)
    (out/'export_manifest.json').write_text(json.dumps({'selection_sha256':frozen['model_selection_sha256'],
        'core_sha256':sha(HERE/'feature_core.hpp'),'exporter_sha256':sha(__file__),'models':receipt},indent=2))
    return receipt

class Native:
    def __init__(self,path):
        self.lib=ctypes.CDLL(str(pathlib.Path(path).resolve()))
        self.fn=self.lib.choose_plan
        self.fn.argtypes=[ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int64),
                          ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
        self.fn.restype=ctypes.c_int
        self.feat=self.lib.feature_map
        self.feat.argtypes=self.fn.argtypes[:-1]+[ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
        self.feat.restype=ctypes.c_int
    def buffers(self,q,k):
        if len(q)!=len(k) or not len(q):raise ValueError('aligned nonempty lengths required')
        if any(type(v) is not int for v in (*q,*k)):raise TypeError('lengths must be integers')
        return ((ctypes.c_int64*len(q))(*q),(ctypes.c_int64*len(k))(*k),(ctypes.c_double*6)())
    def choose(self,q,k,hq,hkv,sms):
        qb,kb,p=self.buffers(q,k)
        ans=self.fn(qb,kb,len(q),hq,hkv,sms,p)
        if ans<0:raise ValueError(f'Native input/model failure: {ans}')
        return ans,np.array(p)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection-dir',required=True,type=pathlib.Path)
    p.add_argument('--out',required=True,type=pathlib.Path)
    a=p.parse_args();export(a.selection_dir,a.out)
