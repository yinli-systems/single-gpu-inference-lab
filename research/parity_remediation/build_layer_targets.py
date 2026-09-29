import argparse,hashlib,json,sys
from pathlib import Path
from evidence_contract import read_verified,first_difference
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('preserve target freeze')
 sys.path.insert(0,str(a.old/'source'));sys.path.insert(0,str(a.old/'serving-source-v3'))
 from http_bench import workloads
 q={r['id']:r for r in workloads()[1]['cells']}
 b=read_verified(a.old/'serving-runs/1639805/pristine/decode-b1.json');c=read_verified(a.old/'serving-runs/1639805/cap/decode-b1.json')
 b={r['id']:r['tokens'] for r in b['requests']};c={r['id']:r['tokens'] for r in c['requests']}
 targets={}
 for rid,index in [('decode-11',74),('decode-13',114)]:
  if first_difference(b[rid],c[rid])!=index:raise ValueError('old witness changed')
  history=q[rid]['input_ids']+b[rid][:index];h=hashlib.sha256(json.dumps(history,sort_keys=True).encode()).hexdigest()
  targets[h]=dict(request_id=rid,output_index_zero_based=index,seq_len=len(history),historical_tokens=[b[rid][index],c[rid][index]])
 a.out.write_text(json.dumps(targets,indent=2)+'\n');print(json.dumps(targets,indent=2))
