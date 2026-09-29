"""Plot every registered primary cell; never select a favorable GPU or dtype."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def plot(summary,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 d=json.loads(summary.read_text())
 if not d['complete'] or d['formal_runs']!=6 or d['timing_rows']!=20736:raise ValueError('complete formal matrix required')
 out.mkdir(parents=True,exist_ok=False)
 gpus=sorted({x['gpu'] for x in d['cells']})
 for gpu in gpus:
  fig,ax=plt.subplots(figsize=(7.6,4.8))
  for j,dtype in enumerate(('float16','bfloat16')):
   cells=[x for x in d['cells'] if x['gpu']==gpu and x['dtype']==dtype and x['primary']]
   cells.sort(key=lambda x:int(x['config'][1:]))
   if [x['config'] for x in cells]!=['n2','n4','n8','n16']:raise ValueError('missing primary cells')
   means=[x['same_minus_opposite_us']['mean'] for x in cells]
   low=[m-x['same_minus_opposite_us']['CI'][0] for m,x in zip(means,cells)]
   high=[x['same_minus_opposite_us']['CI'][1]-m for m,x in zip(means,cells)]
   if any(v<0 for v in low+high):raise ValueError('interval does not bracket estimate; do not clip it')
   ax.errorbar([i+(-.08 if j==0 else .08) for i in range(4)],means,yerr=[low,high],
               marker='o' if j==0 else 's',linestyle='none',capsize=4,label='FP16' if j==0 else 'BF16')
  ax.axhline(0,linestyle='--',linewidth=1)
  ax.set_xticks(range(4),['2','4','8','16'])
  ax.set_xlabel('Simultaneous prefill requests')
  ax.set_ylabel('Same-pairing minus opposite-pairing time (µs)')
  ax.set_title(gpu.replace('NVIDIA GeForce ','')+' · FlashInfer FA2 · graph replay')
  ax.grid(axis='y',alpha=.2);ax.legend()
  fig.text(.5,.025,'95% conditional process/block intervals · 3 process repeats\nShape contrasts are not optimization speedups; no full-model serving claim.',ha='center',fontsize=8)
  fig.tight_layout(rect=(0,.08,1,1))
  slug=gpu.lower().replace('nvidia geforce ','').replace(' ','-')
  fig.savefig(out/(slug+'.png'),dpi=160);plt.close(fig)
 for control,title in [('order_null','Row-order / physical-layout control'),('equal_work','Equal analytical-work control')]:
  selected=[c for c in d['cells'] if c['primary'] and c[control] is not None]
  selected.sort(key=lambda c:(c['gpu'],c['dtype'],int(c['config'][1:])))
  fig,ax=plt.subplots(figsize=(9.0,6.5))
  y=list(range(len(selected)))
  estimate=[c[control]['mean']/c[control]['epsilon_us'] for c in selected]
  left=[(c[control]['mean']-c[control]['CI'][0])/c[control]['epsilon_us'] for c in selected]
  right=[(c[control]['CI'][1]-c[control]['mean'])/c[control]['epsilon_us'] for c in selected]
  if any(x<0 for x in left+right):raise ValueError('interval does not bracket estimate')
  ax.errorbar(estimate,y,xerr=[left,right],fmt='o',capsize=3)
  ax.axvline(0,linewidth=1);ax.axvline(-1,linestyle='--',linewidth=1);ax.axvline(1,linestyle='--',linewidth=1)
  labels=[c['gpu'].split()[-1]+' '+('FP16' if c['dtype']=='float16' else 'BF16')+' '+c['config'] for c in selected]
  ax.set_yticks(y,labels);ax.invert_yaxis()
  ax.set_xlabel('Paired timing difference / registered equivalence tolerance')
  failed=sum(not c[control]['equivalent'] for c in selected)
  ax.set_title(title+f' · {failed}/{len(selected)} fail equivalence')
  ax.grid(axis='x',alpha=.2)
  fig.text(.5,.018,'90% conditional intervals. Equivalence requires the entire interval inside [-1, +1].\nFailed controls limit attribution; this is not an optimization leaderboard.',ha='center',fontsize=8)
  fig.tight_layout(rect=(0,.07,1,1))
  fig.savefig(out/(control+'.png'),dpi=160);plt.close(fig)
 print('PLOTS',len(gpus)+2,'matplotlib',matplotlib.__version__)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();plot(a.summary,a.out)
