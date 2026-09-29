"""Frozen resource-policy evaluation families; no measurements used to select cases."""
import hashlib,json,random

def digest(x):
 return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def build():
 rows=[dict(id='exposed-two-request',family='exposed',q=[1024,63],cached=[128,16384]),
       dict(id='canary-small',family='canary',q=[16,32],cached=[32,64]),
       dict(id='canary-wide',family='canary',q=[512,128],cached=[1024,4096])]
 for phase,seed,ns,count in [('development',930117,(1,2,4,8),16),('confirmatory',930229,(3,5,7,11),24)]:
  rng=random.Random(seed)
  for j in range(count):
   n=ns[(j//4)%len(ns)]
   regime=('balanced','same','opposite','mixed')[j%4]
   existing={digest((x['q'],x['cached'])) for x in rows}
   for attempt in range(100):
    q=[rng.choice((16,32,48,64,96,128,192,256,384,512)) for _ in range(n)]
    k=[rng.choice((0,128,512,1024,2048,4096,8192,16384)) for _ in range(n)]
    if regime=='balanced':q=[q[0]]*n;k=[k[0]]*n
    elif regime=='same':q=sorted(q);k=sorted(k)
    elif regime=='opposite':q=sorted(q,reverse=True);k=sorted(k)
    if digest((q,k)) not in existing:break
   else:raise ValueError('unique fixture construction exhausted')
   rows.append(dict(id=f'{phase}-{j:02d}',family=phase,regime=regime,q=q,cached=k))

 if len({digest((x['q'],x['cached'])) for x in rows})!=len(rows):raise ValueError('duplicate geometry')
 return dict(schema=1,cases=rows,case_hash=digest(rows),primary='cap',secondary='wide',
             modes=['pristine','off','cap','wide'],dtypes=['float16','bfloat16'],
             layouts=['ragged','paged'],splits=['auto','unsplit'],blocks=8,
             regimes=['graph_one','graph16'],full_model=False)
if __name__=='__main__':print(json.dumps(build(),indent=2))
