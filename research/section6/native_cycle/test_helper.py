"""Compile the actual native helper and compare exact descriptor permutations."""
import argparse,json,random,subprocess,sys,tempfile,shutil,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from measure import Geometry,propose
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.out.exists():raise FileExistsError('preserve receipt')
compiler=shutil.which('g++') or shutil.which('clang++')
if not compiler:raise RuntimeError('no C++ compiler')
header=Path(__file__).with_name('sgi_causal_order.cuh')
source=r"""#include <stdexcept>
#include <iostream>
#define FLASHINFER_CHECK(x,msg) do {if(!(x)) throw std::runtime_error(msg);} while(0)
#include "sgi_causal_order.cuh"
int main(int argc,char**argv){
 try{
  setenv("SGI_FA2_RESIDENCY_PROBE",argc>1?argv[1]:"1",1);
  int qp[]={0,1024,1087},kp[]={0,1152,17599};
  std::vector<int>r(34),q(34),k(34);
  for(int i=0;i<34;++i)if(!(std::cin>>r[i]>>q[i]>>k[i]))return 3;
  flashinfer::SGICausalOrderWitness(qp,kp,2,4,128,false,false,1,r,q,k);
  for(int i=0;i<34;++i)std::cout<<r[i]<<" "<<q[i]<<" "<<k[i]<<"\n";
 }catch(const std::exception&e){std::cerr<<e.what();return 7;}
}
"""
g=Geometry((1024,63),(1152,16447),4,128,-1,False);native=list(g.descriptors());rng=random.Random(8941)
with tempfile.TemporaryDirectory(prefix='sgi-native-helper-') as tmp:
 tmp=Path(tmp);cc=tmp/'test.cc';cc.write_text(source);exe=tmp/'test'
 subprocess.run([compiler,'-std=c++17','-O2','-I',str(header.parent),str(cc),'-o',str(exe)],check=True)
 def call(desc,flag):
  return subprocess.run([str(exe),flag],input='\n'.join(' '.join(map(str,x)) for x in desc)+'\n',capture_output=True,text=True)
 checked=0
 for _ in range(100):
  desc=native[:];rng.shuffle(desc)
  for flag,policy in [('0','identity'),('1','causal_heavy')]:
   result=call(desc,flag);assert result.returncode==0,result.stderr
   actual=[tuple(map(int,line.split())) for line in result.stdout.splitlines()]
   expected=[desc[i] for i in propose(g,desc,policy)];assert actual==expected;checked+=1
 rejected=0
 for value in [(2,0,0),(-1,0,0),(0,-1,0),(0,32,0),(0,0,1),(0,1,0)]:
  desc=native[:];desc[0]=value;assert call(desc,'1').returncode==7;rejected+=1
 assert call(native,'bad').returncode==7;rejected+=1
 rec=dict(compiler=compiler,comparisons=checked,invalid_rejections=rejected,helper_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),GPU_executed=False)
 a.out.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec))
