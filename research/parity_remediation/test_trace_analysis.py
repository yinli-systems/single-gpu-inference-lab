import copy,unittest
from analyze_trace import compare,digest

def row(h='history',hash_='same',complete=True):
 r=dict(batch_size=1,forward_mode='DECODE',seq_lens=[202],input_ids=[7],history_sha256=[h],all_histories_complete=complete,logit_sha256=[hash_],top8_values=[[1.0]],top8_ids=[[7]],sampled_ids=[7]);r['batch_signature']=digest({k:r[k] for k in ('forward_mode','seq_lens','input_ids','history_sha256')});return r
class TraceAnalysisTests(unittest.TestCase):
 def test_matching(self):self.assertEqual(compare([row()],[row()])['exact_logit_signature_matches'],1)
 def test_different_history_not_paired(self):self.assertEqual(compare([row()],[row(h='different')])['matched_unique_batch_signatures'],0)
 def test_unknown_rejected(self):self.assertEqual(compare([row(h=None,complete=False)],[row()])['unknown_history_steps'],[1,0])
 def test_different_logits(self):self.assertEqual(len(compare([row()],[row(hash_='other')])['cross_arm_logit_differences']),1)
 def test_within_variability_not_blame_candidate(self):self.assertEqual(len(compare([row(),row(hash_='other')],[row()])['within_context_variation']),1)
 def test_baseline_variation_does_not_hide_novel_candidate(self):
  d=compare([row(),row(hash_='second')],[row(hash_='new')]);self.assertEqual(d['disjoint_cross_arm_signatures'],1);self.assertEqual(d['signatures_with_cap_only_variants'],1)
 def test_no_causal_promotion(self):
  d=compare([row()],[row()]);self.assertFalse(d['historical_failure_resolved']);self.assertFalse(d['physical_KV_or_hidden_state_equality_established'])
if __name__=='__main__':unittest.main()
