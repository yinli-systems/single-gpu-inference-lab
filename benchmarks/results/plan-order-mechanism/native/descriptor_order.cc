// Experimental host-only FA2 descriptor ordering. Not installed in FlashInfer.
// The GPU arithmetic, output slots and merge ordering are unchanged.
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <vector>

extern "C" int sgi_descriptor_order(const int32_t* q,const int32_t* lengths,int n,
    const int32_t* request,const int32_t* qt,const int32_t* kt,int count,
    int group,int tile,int chunk,int split,int policy,int32_t* out) noexcept {
  try {
    if (!q||!lengths||!request||!qt||!kt||!out||n<1||n>4096||count<1||count>(1<<24)||
        group<1||group>256||tile<1||tile>4096||(split!=0&&split!=1)||
        (split&&chunk<1)||policy<0||policy>4) return 1;
    std::vector<int64_t> offsets(n+1,0),splits(n,1),tiles(n,0);
    for(int r=0;r<n;++r){
      if(q[r]<1||lengths[r]<q[r])return 2;
      splits[r]=split?(int64_t(lengths[r])+chunk-1)/chunk:1;
      tiles[r]=(int64_t(q[r])*group+tile-1)/tile;
      if(tiles[r]>(count-offsets[r])/splits[r])return 3; // checked product, no signed overflow
      offsets[r+1]=offsets[r]+tiles[r]*splits[r];
      if(offsets[r+1]>count)return 3;
    }
    if(offsets.back()!=count)return 3;
    std::vector<uint8_t> seen(count,0);
    std::vector<int64_t> cost(count,0);
    for(int i=0;i<count;++i){
      const int r=request[i];
      if(r<0||r>=n||qt[i]<0||qt[i]>=tiles[r]||kt[i]<0||kt[i]>=splits[r])return 4;
      const int64_t slot=offsets[r]+int64_t(qt[i])*splits[r]+kt[i];
      if(seen[slot]++)return 5;
      const int64_t packed=std::min<int64_t>(tile,int64_t(q[r])*group-int64_t(qt[i])*tile);
      const int64_t visible=std::min<int64_t>(lengths[r],int64_t(lengths[r])-q[r]+((int64_t(qt[i])+1)*tile+group-1)/group);
      const int64_t first=split?int64_t(kt[i])*chunk:0;
      const int64_t last=split?std::min<int64_t>(visible,(int64_t(kt[i])+1)*chunk):visible;
      cost[i]=std::max<int64_t>(0,last-first)*packed;
    }
    std::iota(out,out+count,0);
    if(policy==0||policy==1)return 0; // identity and identical A/A control
    std::stable_sort(out,out+count,[&](int a,int b){
      if(policy==2)return std::make_tuple(-request[a],qt[a],kt[a])<std::make_tuple(-request[b],qt[b],kt[b]);
      if(policy==3)return std::make_tuple(-cost[a],request[a],qt[a],kt[a])<std::make_tuple(-cost[b],request[b],qt[b],kt[b]);
      return std::make_tuple(qt[a],kt[a],-int64_t(lengths[request[a]]),request[a])<std::make_tuple(qt[b],kt[b],-int64_t(lengths[request[b]]),request[b]);
    });
    return 0;
  } catch(...) {return 99;}
}
