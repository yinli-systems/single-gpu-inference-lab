#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <utility>
namespace flashinfer {
template <typename IdType>
inline void SGICausalOrderWitness(const IdType* qp, const IdType* kp,
 uint32_t batch,uint32_t group,uint32_t tile,bool split,bool graph,uint32_t page,
 std::vector<IdType>& requests,std::vector<IdType>& qtiles,std::vector<IdType>& ktiles) {
 const char* flag=std::getenv("SGI_FA2_RESIDENCY_PROBE");
 if(flag==nullptr||std::strcmp(flag,"0")==0)return;
 FLASHINFER_CHECK(std::strcmp(flag,"1")==0,"invalid diagnostic flag");
 FLASHINFER_CHECK(batch==2&&group==4&&tile==128&&!split&&!graph&&page==1,
                  "native witness diagnostic outside qualified shape");
 FLASHINFER_CHECK(qp[0]==0&&qp[1]==1024&&qp[2]==1087&&kp[0]==0&&kp[1]==1152&&kp[2]==17599,
                  "native witness diagnostic geometry mismatch");
 FLASHINFER_CHECK(requests.size()==34&&qtiles.size()==34&&ktiles.size()==34,"incomplete work map");
 std::vector<std::pair<int64_t,size_t>> scores;scores.reserve(34);bool seen[34]={};
 for(size_t i=0;i<34;++i){
  const uint32_t r=requests[i];FLASHINFER_CHECK(r<2&&ktiles[i]==0,"invalid work descriptor");
  const int64_t q=qp[r+1]-qp[r],L=kp[r+1]-kp[r];
  FLASHINFER_CHECK(qtiles[i]>=0,"negative query tile");
  const int64_t left=int64_t(qtiles[i])*tile,right=std::min<int64_t>(left+tile,q*group);
  FLASHINFER_CHECK(left<right,"invalid query tile");
  const size_t slot=(r?32:0)+qtiles[i];FLASHINFER_CHECK(slot<34&&!seen[slot],"duplicate work descriptor");seen[slot]=true;
  const auto prefix=[&](int64_t n){const int64_t full=n/group,rem=n%group;
   return int64_t(group)*((L-q+1)*full+full*(full-1)/2)+rem*(L-q+1+full);};
  scores.emplace_back(prefix(right)-prefix(left),i);
 }
 std::stable_sort(scores.begin(),scores.end(),[](const auto&a,const auto&b){return a.first>b.first;});
 const auto oldr=requests,oldq=qtiles,oldk=ktiles;
 for(size_t i=0;i<34;++i){auto j=scores[i].second;requests[i]=oldr[j];qtiles[i]=oldq[j];ktiles[i]=oldk[j];}
}
}
