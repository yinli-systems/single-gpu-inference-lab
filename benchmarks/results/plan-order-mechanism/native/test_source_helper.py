"""Host-level conformance of the actual header inserted into FlashInfer.
This test is not a CUDA performance measurement.
"""
import argparse
import ctypes
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from plan_contract import expected_descriptors,order_indices
D=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path);args=parser.parse_args()
if args.out is not None and args.out.exists():raise FileExistsError('new receipt required')
wrapper=r"""
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>
#define FLASHINFER_CHECK(x, message) do {if(!(x))throw std::runtime_error(message);}while(false)
#include "sgi_experimental_order.cuh"
extern "C" int check_order(const int32_t* qp,const int32_t* kp,uint32_t batch,
 uint32_t group,uint32_t tile,int64_t chunk,int split,int graph,uint32_t page,
 int32_t* r,int32_t* q,int32_t* k,uint32_t count) noexcept {
 try {
  std::vector<int32_t> vr(r,r+count),vq(q,q+count),vk(k,k+count);
  flashinfer::SGIExperimentalOrderFA2(qp,kp,batch,group,tile,chunk,split,graph,page,vr,vq,vk);
  std::copy(vr.begin(),vr.end(),r);std::copy(vq.begin(),vq.end(),q);std::copy(vk.begin(),vk.end(),k);return 0;
 } catch(...) {return 1;}
}
"""
with tempfile.TemporaryDirectory(prefix='sgi-source-helper-') as temp:
 temp=Path(temp);(temp/'test.cc').write_text(wrapper)
 lib=temp/('helper.dylib' if sys.platform=='darwin' else 'helper.so')
 compiler=['xcrun','clang++'] if sys.platform=='darwin' else ['g++']
 shared=['-dynamiclib'] if sys.platform=='darwin' else ['-shared','-fPIC']
 subprocess.run(compiler+['-std=c++17','-O2','-Wall','-Wextra','-Werror',*shared,'-I'+str(D),str(temp/'test.cc'),'-o',str(lib)],check=True)
 dll=ctypes.CDLL(str(lib));f=dll.check_order;I=ctypes.c_int32;P=ctypes.POINTER(I)
 f.argtypes=[P,P,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_int64,ctypes.c_int,ctypes.c_int,ctypes.c_uint32,P,P,P,ctypes.c_uint32];f.restype=ctypes.c_int
 arr=lambda xs:(I*len(xs))(*xs)
 def invoke(q,L,desc,group,tile,chunk,split,mode,graph=0,page=1):
  if mode is None:os.environ.pop('FLASHINFER_EXP_WORK_ORDER',None)
  else:os.environ['FLASHINFER_EXP_WORK_ORDER']=mode
  qp=arr([0]+list(itertools.accumulate(q)));kp=arr([0]+list(itertools.accumulate(L)))
  cols=[arr([d[j] for d in desc]) for j in range(3)]
  rc=f(qp,kp,len(q),group,tile,chunk,int(split),graph,page,*cols,len(desc))
  return rc,list(zip(*[list(c) for c in cols]))
 rng=random.Random(7771);comparisons=0;rejects=0
 try:
  for _ in range(300):
   n=rng.randint(1,12);q=[rng.randint(1,512) for _ in range(n)];L=[qi+rng.randint(0,12000) for qi in q]
   group=rng.choice([1,2,4,8]);tile=rng.choice([16,64,128]);chunk=rng.choice([512,2048,8192]);split=rng.choice([False,True])
   desc,_,_=expected_descriptors(q,L,group,tile,chunk,split)
   for mode in [None,'identity','heavy_first']:
    rc,got=invoke(q,L,desc,group,tile,chunk,split,mode)
    order=order_indices(desc,q,L,tile,chunk,split,group,'heavy_first' if mode=='heavy_first' else 'identity')
    assert rc==0 and got==[desc[i] for i in order];comparisons+=1
   for mode,graph,page in [('invalid',0,1),('heavy_first',1,1),('heavy_first',0,16)]:
    rc,_=invoke(q,L,desc,group,tile,chunk,split,mode,graph,page);assert rc!=0;rejects+=1
   rc,got=invoke(q,L,desc,group,tile,chunk,split,None,1,16)
   assert rc==0 and got==desc;comparisons+=1
 finally:os.environ.pop('FLASHINFER_EXP_WORK_ORDER',None)
 result=dict(complete=True,platform=sys.platform,actual_header_sha256=hashlib.sha256((D/'sgi_experimental_order.cuh').read_bytes()).hexdigest(),
             random_geometries=300,exact_descriptor_comparisons=comparisons,unsupported_regime_rejections=rejects,
             actual_CUDA_execution=False,performance_claim=False)
 if args.out is not None:args.out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
