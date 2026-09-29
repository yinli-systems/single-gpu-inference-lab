"""Assemble source plus immutable evidence and the CI CPU-reproduction receipt."""
import argparse,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2]
if a.out.exists():raise FileExistsError('preserve delivery')
paths=[x for x in (root/'research/section6').rglob('*') if x.is_file() and '__pycache__' not in x.parts]
paths += list((root/'research/order_guard').glob('*.py'))
paths += [root/'benchmarks/results/plan-order-mechanism'/n for n in ['plan_contract.py','measure_order.py','frozen_geometry.py']]
paths += [root/'.github/workflows/section6.yml']
with zipfile.ZipFile(a.out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(set(paths)):z.write(f,str(f.relative_to(root)))
 z.write(a.receipt,'CI_REPRODUCTION.json')
 z.writestr('README.txt','Start at research/section6/README.md. GPU evidence is archived; reproduction is CPU-only. A/C11 of12 runs, B and D complete, E not executed. No full-model speedup claim.\n')
print(a.out)
