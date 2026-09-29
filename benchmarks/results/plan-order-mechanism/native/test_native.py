"""CPU native/Python differential tests, not GPU performance evidence."""
import ctypes
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from plan_contract import expected_descriptors,order_indices,POLICIES
D=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='sgi-order-native-') as tmp:
    lib=Path(tmp)/('order.dylib' if sys.platform=='darwin' else 'order.so')
    compiler=['xcrun','clang++'] if sys.platform=='darwin' else ['g++']
    shared=['-dynamiclib'] if sys.platform=='darwin' else ['-shared','-fPIC']
    cmd=compiler+['-std=c++17','-O3']+shared+['-Wall','-Wextra','-Werror',str(D/'descriptor_order.cc'),'-o',str(lib)]
    subprocess.run(cmd,check=True)
    dll=ctypes.CDLL(str(lib));f=dll.sgi_descriptor_order;I=ctypes.c_int32;P=ctypes.POINTER(I)
    f.argtypes=[P,P,ctypes.c_int,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
    f.restype=ctypes.c_int
    arr=lambda xs:(I*len(xs))(*xs)
    rng=random.Random(8821);checks=0;rejections=0;start=time.monotonic()
    for _ in range(300):
        n=rng.randint(1,8);q=[rng.randint(1,512) for i in range(n)];L=[qi+rng.randint(0,12000) for qi in q]
        tile=rng.choice([64,128]);chunk=rng.choice([512,2048,8192]);split=rng.choice([False,True])
        desc,_,_=expected_descriptors(q,L,4,tile,chunk,split);rng.shuffle(desc)
        qa,la=arr(q),arr(L);columns=[arr([d[j] for d in desc]) for j in range(3)];out=(I*len(desc))()
        for pi,policy in enumerate(POLICIES):
            rc=f(qa,la,n,*columns,len(desc),4,tile,chunk,int(split),pi,out)
            assert rc==0,(rc,policy)
            assert list(out)==order_indices(desc,q,L,tile,chunk,split,4,policy),policy
            checks+=1
        old=columns[0][0];columns[0][0]=n
        assert f(qa,la,n,*columns,len(desc),4,tile,chunk,int(split),3,out)!=0;columns[0][0]=old;rejections+=1
    # Adversarial integer product must reject, not overflow into an allocation.
    assert f(arr([2147483647]),arr([2147483647]),1,arr([0]),arr([0]),arr([0]),1,256,1,1,1,3,(I*1)())!=0
    rejections+=1
    result=dict(complete=True,platform=sys.platform,compiler_command=cmd[:-1]+['<temporary-library>'],
                differential_checks=checks,invalid_input_rejections=rejections,random_geometries=300,
                source_sha256=hashlib.sha256((D/'descriptor_order.cc').read_bytes()).hexdigest(),
                seconds=time.monotonic()-start,GPU_integration=False,performance_claim=False)
    (D/'cpu-differential.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
