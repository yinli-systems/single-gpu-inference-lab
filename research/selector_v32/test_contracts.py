import json,math,sys,unittest
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent;sys.path.insert(0,str(HERE))
from manifest import load,strict_opposite,plan_guard,digest
from release_gate import evaluate

class ManifestTests(unittest.TestCase):
 def test_frozen_hash_counts_and_freshness(self):
  m=load();self.assertEqual(len(m['cases']),34);self.assertEqual(sum(c['family']=='release' for c in m['cases']),30)
  receipt=json.loads((HERE/'FRESHNESS_RECEIPT.json').read_text())
  self.assertEqual(receipt['schema'],2);self.assertEqual(receipt['new_case_hash'],m['case_hash']);self.assertEqual(receipt['intersection_count'],0)
  def gd(c):return __import__('hashlib').sha256(json.dumps([c['q'],c['cached']],separators=(',',':')).encode()).hexdigest()
  new_d={gd(c) for c in m['cases']};prior={d for row in receipt['prior_manifests'] for d in row['geometry_sha256']}
  self.assertEqual(new_d,set(receipt['new_geometry_sha256']));self.assertFalse(new_d&prior);self.assertEqual(len(new_d),34)
 def test_expected_selector_contract(self):
  for c in load()['cases']:
   count=sum((q+31)//32 for q in c['q']);self.assertEqual(count,c['descriptor_count'])
   for gpu,sms in [('gpu_4090',128),('gpu_5090',170)]:
    self.assertEqual(plan_guard(c['q'],c['cached'],count,32,8,sms,False),c['expected_selector'][gpu])
 def test_regimes(self):
  for c in load()['cases']:
   self.assertEqual(strict_opposite(c['q'],c['cached']),c['regime'] in ('opposite','tied-opposite'))

class GateTests(unittest.TestCase):
 def cell(self,selected,execution,ratio=1.05,off=1.0,resolved=True):
  return {'selected':selected,'execution_mode':execution,'comparisons':{'guarded':{'ratio':ratio,'controls_resolve':resolved},'off':{'ratio':off,'controls_resolve':True,'disabled_overlay_resolves_one_percent':True},'paired_guarded':{'ratio':ratio/off,'controls_resolve':resolved}}}
 def draws(self,cells):
  return {(i,m):np.full(10000,math.log(c['comparisons'][m]['ratio'])) for i,c in enumerate(cells) for m in ('guarded','off','paired_guarded')}
 def numerics(self):return {m:{'qualifications':10,'exact_full_outputs':10,'max_abs_vs_pristine':0.0} for m in ('pristine','off','cap','guarded')}
 def test_pass_and_fail_closed(self):
  cells=[]
  for ex in ('eager_full_call','graph1_replay','graph16_replay'):
   cells+=[self.cell(True,ex,1.08),self.cell(False,ex,1.0)]
  self.assertTrue(evaluate(cells,self.draws(cells),self.numerics())['pass'])
  bad=[dict(c) for c in cells];bad[0]={**bad[0],'comparisons':{'guarded':{'ratio':.985,'controls_resolve':True},'off':{'ratio':1.,'controls_resolve':True,'disabled_overlay_resolves_one_percent':True},'paired_guarded':{'ratio':.985,'controls_resolve':True}}}
  self.assertFalse(evaluate(bad,self.draws(bad),self.numerics())['pass'])
 def test_control_and_numerics_block(self):
  cells=[self.cell(True,x,1.08) for x in ('eager_full_call','graph1_replay','graph16_replay')]
  cells[1]['comparisons']['guarded']['controls_resolve']=False
  self.assertFalse(evaluate(cells,self.draws(cells),self.numerics())['pass'])
  n=self.numerics();n['guarded']['exact_full_outputs']=9
  self.assertFalse(evaluate([self.cell(True,x,1.08) for x in ('eager_full_call','graph1_replay','graph16_replay')],self.draws([self.cell(True,x,1.08) for x in ('eager_full_call','graph1_replay','graph16_replay')]),n)['pass'])

if __name__=='__main__':unittest.main()
