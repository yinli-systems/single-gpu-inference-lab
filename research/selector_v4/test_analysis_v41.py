from __future__ import annotations
import csv,json,tempfile,unittest
from pathlib import Path
from research.selector_v4.analyze_v4 import (absolute_native_block_evidence,
    hierarchical_ratio_summary, simultaneous_ratio_lcb, telemetry_receipt, validate_binary_audit)

class AnalysisV41Tests(unittest.TestCase):
    def test_telemetry_stability_and_insufficient_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'t.csv'
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['utilization.gpu [%]','clocks.current.sm [MHz]']);w.writeheader()
                for i in range(20):w.writerow({'utilization.gpu [%]':'100 %','clocks.current.sm [MHz]':f'{2700+i%3} MHz'})
            self.assertTrue(telemetry_receipt(p)['stable'])
            p.write_text('utilization.gpu [%],clocks.current.sm [MHz]\n100 %,2700 MHz\n')
            self.assertFalse(telemetry_receipt(p)['stable'])

    def test_absolute_native_evidence_and_hierarchical_gate(self):
        key=("x","float16","ragged","auto")
        pristine=[];paired=[]
        for block in range(4):
            roles=("a","b","b","a") if block%2==0 else ("b","a","a","b")
            arms=("off","cap","cap","off") if block%2==0 else ("cap","off","off","cap")
            for pos,role in enumerate(roles):
                pristine.append({"case":"x","dtype":"float16","layout":"ragged","split":"auto","execution_mode":"graph1_replay","block":block,"position":pos,"arm":"pristine","role":role,"wall_us":100.0})
            for pos,arm in enumerate(arms):
                paired.append({"case":"x","dtype":"float16","layout":"ragged","split":"auto","execution_mode":"graph1_replay","block":block,"position":pos,"arm":arm,"role":"A" if arm=="off" else "B","wall_us":100.0 if arm=="off" else 80.0})
        evidence=absolute_native_block_evidence(pristine,paired,key,"graph1_replay")
        self.assertEqual(evidence["ratios"],(1.0,)*4)
        summary=hierarchical_ratio_summary({0:evidence["ratios"],1:evidence["ratios"],2:evidence["ratios"]},seed=7,draws=2000)
        self.assertEqual(summary["ratio"],1.0);self.assertEqual(simultaneous_ratio_lcb([summary]),1.0)

    def test_absolute_native_regression_is_not_hidden_by_cap_gain(self):
        repeats={i:(.985,)*16 for i in range(3)}
        summary=hierarchical_ratio_summary(repeats,seed=9,draws=2000)
        self.assertLess(summary["ratio"],.99);self.assertLess(simultaneous_ratio_lcb([summary]),.99)

    def test_absolute_native_rejects_incomplete_repeats(self):
        with self.assertRaisesRegex(ValueError,"repeats"):
            hierarchical_ratio_summary({0:(1.0,),1:(1.0,)},seed=1,draws=2000)

    def test_binary_audit_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.json';p.write_text(json.dumps({'kernel_symbol_isolation_compiled':True,'same_module_pairs':{'ragged':True,'paged':True},'binaries':[{'sha256':'a'*64,'bytes':3}]}))
            self.assertEqual(validate_binary_audit(p)['same_module_pairs'],{'ragged':True,'paged':True})
            x=json.loads(p.read_text());x['same_module_pairs']['paged']=False;p.write_text(json.dumps(x))
            with self.assertRaisesRegex(RuntimeError,'co-residence'):validate_binary_audit(p)
if __name__=='__main__':unittest.main()
