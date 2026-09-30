from __future__ import annotations
import csv,json,tempfile,unittest
from pathlib import Path
from research.selector_v4.analyze_v4 import telemetry_receipt,validate_binary_audit

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
    def test_binary_audit_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.json';p.write_text(json.dumps({'kernel_symbol_isolation_compiled':True,'same_module_pairs':{'ragged':True,'paged':True},'binaries':[{'sha256':'a'*64,'bytes':3}]}))
            self.assertEqual(validate_binary_audit(p)['same_module_pairs'],{'ragged':True,'paged':True})
            x=json.loads(p.read_text());x['same_module_pairs']['paged']=False;p.write_text(json.dumps(x))
            with self.assertRaisesRegex(RuntimeError,'co-residence'):validate_binary_audit(p)
if __name__=='__main__':unittest.main()
