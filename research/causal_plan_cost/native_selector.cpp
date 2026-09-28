// Native lowering of geometry.py causal features and frozen sklearn predictors.
// This does not change a learned policy or tune on test timings.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
using I = int64_t;
constexpr int DIM=35;
struct Node { int left,right,feature; double threshold,value; };
struct Model { int kind; const double* weights; double bias;
               const Node* nodes; const int* roots; int trees; };
#include "models.inc"
static I cd(I x,I y) { return (x+y-1)/y; }

static int representation(int n,const I*q,const I*k,int hq,int hkv,int sms,
                          int policy,double*out) {
    if(n<1 || n>64 || hq<1 || hkv<1 || hq%hkv || sms<1 || policy<0 || policy>5)
        return -1;
    I Q=0,K=0,maxq=0,maxk=0,maxL=0; double q2=0,k2=0,W=0,tri=0;
    for(int i=0;i<n;++i) {
        if(q[i]<1 || q[i]>131072 || k[i]<0 || k[i]>1048576) return -2;
        Q+=q[i];K+=k[i];maxq=std::max(maxq,q[i]);maxk=std::max(maxk,k[i]);
        maxL=std::max(maxL,q[i]+k[i]);q2+=double(q[i])*q[i];k2+=double(k[i])*k[i];
        tri+=double(q[i])*(q[i]+1)/2;W+=double(q[i])*k[i]+double(q[i])*(q[i]+1)/2;
    }
    if(Q>131072 || hq>256) return -3;
    double meanq=double(Q)/n,meank=double(K)/n,vq=0,vk=0;
    for(int i=0;i<n;++i) { vq+=(q[i]-meanq)*(q[i]-meanq);vk+=(k[i]-meank)*(k[i]-meank); }
    vq/=n;vk/=n;
    int g=hq/hkv;I packed=Q*g/n,tile=packed>64?128:(packed>16?64:16);
    I size=maxL;bool split=false;
    if(policy==0) {
        I low=128,high=maxL,capacity=2*I(sms)/hkv;
        while(low<high) {
            I mid=(low+high)/2,count=0;
            for(int i=0;i<n;++i) count+=cd(q[i]*g,tile)*cd(q[i]+k[i],mid);
            if(count>capacity)low=mid+1;else high=mid;
        }
        size=low;split=size<maxL;
    } else if(policy>1) {
        size=I(512)<<(policy-2);split=maxL>size;
    }
    I grid=0,active=0,merge=0,rmax=0,tmax=0;
    double rsum=0,rsq=0,tsum=0,tsq=0;
    for(int i=0;i<n;++i) {
        I L=q[i]+k[i],chunks=split?cd(L,size):1;
        merge+=q[i]*chunks;
        for(I qt=0;qt<cd(q[i]*g,tile);++qt) {
            I qe=std::min(q[i],cd((qt+1)*tile,g));
            for(I c=0;c<chunks;++c) {
                I start=c*size,end=std::min(L,start+size);
                I r=cd(end-start,64),t=cd(std::max(I(0),std::min(end,k[i]+qe)-start),64);
                ++grid;active+=t>0;rsum+=r;rsq+=double(r)*r;tsum+=t;tsq+=double(t)*t;
                rmax=std::max(rmax,r);tmax=std::max(tmax,t);
            }
        }
    }
    if(!split) merge=0;
    const double vals[29]={double(n),double(Q),double(K),double(maxq),double(maxk),q2,vq,vk,
        double(hq),double(hkv),double(sms),W,double(Q)*K/n+tri,
        double(tile),double(size),double(grid*hkv),double(split),rsum*hkv,double(rmax),
        rsq*hkv,double(cd(grid*hkv,2*sms)),double(merge*hq),double(active*hkv),
        double((grid-active)*hkv),tsum*hkv,double(tmax),tsq*hkv,
        double(cd(active*hkv,2*sms)),std::max(double(tmax),tsum*hkv/(2*sms))};
    for(int i=0;i<29;++i)out[i]=std::log1p(vals[i]);
    for(int i=0;i<6;++i)out[29+i]=double(i==policy);
    return 0;
}

static double predict(const Model&m,const double*x) {
    if(m.kind==0) {
        double value=m.bias;
        for(int i=0;i<DIM;++i)value+=m.weights[i]*x[i];
        return value;
    }
    double result=0;
    for(int tree=0;tree<m.trees;++tree) {
        int i=m.roots[tree];
        while(m.nodes[i].left>=0) {
            const Node&node=m.nodes[i];
            // sklearn ExtraTrees predicts on float32 feature inputs.
            i=float(x[node.feature])<=node.threshold?node.left:node.right;
        }
        result+=m.nodes[i].value;
    }
    return result/m.trees;
}

extern "C" int causal_features(int n,const I*q,const I*k,int hq,int hkv,int sms,
                               int policy,double*out) {
    if(!q || !k || !out)return -4;
    return representation(n,q,k,hq,hkv,sms,policy,out);
}

extern "C" int select_policy(int model,int n,const I*q,const I*k,int hq,int hkv,int sms,
                             double*scores) {
    if(model<0 || model>=MODEL_COUNT || !q || !k || !scores)return -4;
    double x[DIM];int best=0;
    for(int policy=0;policy<6;++policy) {
        int status=representation(n,q,k,hq,hkv,sms,policy,x);
        if(status)return status;
        scores[policy]=predict(MODELS[model],x);
        if(!std::isfinite(scores[policy]))return -5;
        if(scores[policy]<scores[best])best=policy;
    }
    return best;
}
