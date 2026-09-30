from __future__ import annotations
import argparse,hashlib,json,random
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def run(a):
 pc=json.loads(a.pristine_complete.read_text());cc=json.loads(a.cap_complete.read_text())
 if pc['files']['decode-b1.json']!=sha(a.pristine) or cc['files']['decode-b1.json']!=sha(a.cap):raise ValueError('historical receipt mismatch')
 p=json.loads(a.pristine.read_text());c=json.loads(a.cap.read_text())
 if p['workload']!='decode' or p['concurrency']!=8 or p['workload_sha256']!=c['workload_sha256'] or p['errors'] or c['errors']:raise ValueError('wrong historical workload')
 pr={r['id']:r for r in p['requests']};cr={r['id']:r for r in c['requests']}
 ids=[f'decode-{i:02d}' for i in range(16)]
 if list(pr)!=ids or list(cr)!=ids or any(r['input_tokens']!=128 or len(r['tokens'])!=128 for r in pr.values()):raise ValueError('historical request contract changed')
 targets=[]
 for target_id,index in [('decode-11',74),('decode-13',114)]:
  rows=[]
  for i,rid in enumerate(ids):
   rng=random.Random(936644+i);input_ids=[rng.randrange(200,16000) for _ in range(128)]
   forced=list(pr[rid]['tokens'] if i<8 else pr[rid]['tokens'][:index])
   natural=i>=8
   rows.append(dict(id=rid,input_ids=input_ids,forced_tokens=forced,max_new_tokens=len(forced)+(1 if natural else 0),natural=natural,
    expected_pristine_next=(pr[rid]['tokens'][index] if natural else None),historical_cap_next=(cr[rid]['tokens'][index] if natural else None)))
  history=rows[ids.index(target_id)]['input_ids']+rows[ids.index(target_id)]['forced_tokens']
  targets.append(dict(id=target_id,output_index_zero_based=index,seq_len=128+index,target_history_sha256=digest(history),
   historical_tokens=[pr[target_id]['tokens'][index],cr[target_id]['tokens'][index]],rows=rows))
 out=dict(schema=1,historical_job=1639805,historical_file='decode-b1.json',concurrency=8,
  pristine_sha256=sha(a.pristine),cap_sha256=sha(a.cap),pristine_complete_sha256=sha(a.pristine_complete),cap_complete_sha256=sha(a.cap_complete),
  input_generator='random.Random(936644+i); randrange(200,16000) x128',targets=targets)
 out['case_hash']=digest(out);a.out.write_text(json.dumps(out,indent=2)+'\n');print(out['case_hash'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pristine',type=Path,required=True);p.add_argument('--cap',type=Path,required=True);p.add_argument('--pristine-complete',type=Path,required=True);p.add_argument('--cap-complete',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
