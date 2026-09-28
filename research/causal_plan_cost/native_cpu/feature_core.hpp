// Exact implementation of the frozen geometry.py feature map, not a new policy.
// Scope: causal FA2, FP16, D=128, no graphs. No GPU calls or benchmarking here.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace sgi {
using I = std::int64_t;
inline I cd(I x, I y) { return (x + y - 1) / y; }
struct Moments {
    I count=0, active=0, sum=0, square=0, maximum=0;
    void add(I value) {
        ++count; active += value > 0; sum += value;
        square += value * value; maximum=std::max(maximum,value);
    }
};
inline void validate(const I* q,const I* k,int n,int hq,int hkv,int sms,int kind) {
    if (!q || !k || n<1 || n>128 || hq<1 || hq>256 || hkv<1 ||
        hq%hkv || sms<1 || sms>1024 || kind<0 || kind>3)
        throw std::invalid_argument("unsupported feature input");
    I total=0;
    for(int i=0;i<n;++i) {
        if(q[i]<1 || q[i]>65536 || k[i]<0 || k[i]>1048576)
            throw std::invalid_argument("length outside checked native scope");
        total+=q[i];
    }
    if(total>65536) throw std::invalid_argument("query budget outside checked native scope");
}
// Policy IDs are auto, none, 512, 1024, 2048, 4096 in the registered order.
inline int feature(const I* q,const I* k,int n,int hq,int hkv,int sms,
                   int kind,int policy,double* out) {
    if(policy<0 || policy>5 || !out) throw std::invalid_argument("invalid policy/output");
    I Q=0,K=0,qmax=0,kmax=0,qsq=0,ksq=0,W=0,tri=0,Lmax=0;
    for(int i=0;i<n;++i) {
        Q+=q[i];K+=k[i];qmax=std::max(qmax,q[i]);kmax=std::max(kmax,k[i]);
        qsq+=q[i]*q[i];ksq+=k[i]*k[i];tri+=q[i]*(q[i]+1)/2;
        W+=q[i]*k[i]+q[i]*(q[i]+1)/2;Lmax=std::max(Lmax,q[i]+k[i]);
    }
    const double vq=static_cast<double>(static_cast<long double>(qsq)/n-
                    (static_cast<long double>(Q)/n)*(static_cast<long double>(Q)/n));
    const double vk=static_cast<double>(static_cast<long double>(ksq)/n-
                    (static_cast<long double>(K)/n)*(static_cast<long double>(K)/n));
    double values[40];int f=0;
    auto add=[&](double x){values[f++]=x;};
    add(n);add(Q);add(K);add(qmax);add(kmax);add(qsq);add(vq);add(vk);
    add(hq);add(hkv);add(sms);
    const double marginal=static_cast<double>(Q)*K/n+tri;
    if(kind==0) add(marginal);else{add(W);add(marginal);}
    if(kind>=2) {
        const I g=hq/hkv,packed=Q*g/n;
        const I tile=packed>64?128:(packed>16?64:16);
        I size=Lmax;bool split=false;
        if(policy==0) {
            I lo=128,hi=Lmax,capacity=2*sms/hkv;
            while(lo<hi) {
                const I mid=(lo+hi)/2;I tasks=0;
                for(int i=0;i<n;++i)tasks+=cd(q[i]*g,tile)*cd(q[i]+k[i],mid);
                if(tasks>capacity)lo=mid+1;else hi=mid;
            }
            size=lo;split=size<Lmax;
        } else if(policy>=2) {
            size=512LL<<(policy-2);split=Lmax>size;
        }
        Moments rect,causal;I merge=0;
        for(int i=0;i<n;++i) {
            const I L=q[i]+k[i],chunks=split?cd(L,size):1;
            merge+=q[i]*chunks;
            for(I qt=0;qt<cd(q[i]*g,tile);++qt) {
                const I qend=std::min(q[i],cd((qt+1)*tile,g));
                for(I c=0;c<chunks;++c) {
                    const I start=c*size,end=std::min(L,start+size);
                    rect.add(cd(end-start,64));
                    causal.add(cd(std::max(I(0),std::min(end,k[i]+qend)-start),64));
                }
            }
        }
        add(tile);add(size);add(rect.count*hkv);add(split?1:0);
        add(rect.sum*hkv);add(rect.maximum);add(rect.square*hkv);
        add(cd(rect.count*hkv,2*sms));add(split?merge*hq:0);
        if(kind==3) {
            add(causal.active*hkv);add((causal.count-causal.active)*hkv);
            add(causal.sum*hkv);add(causal.maximum);add(causal.square*hkv);
            add(cd(causal.active*hkv,2*sms));
            add(std::max(double(causal.maximum),double(causal.sum*hkv)/(2*sms)));
        }
    }
    for(int j=0;j<f;++j)out[j]=std::log1p(values[j]);
    for(int p=0;p<6;++p)out[f++]=(p==policy)?1.:0.;
    return f;
}
}
