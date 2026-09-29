import json,unittest
from stream_evidence import StreamState

def frame(ids,n,finish=None):return json.dumps(dict(output_ids=ids,meta_info=dict(completion_tokens=n,finish_reason=finish)))
class StreamTests(unittest.TestCase):
 def test_delta(self):
  s=StreamState(2);s.accept(frame([1],1));s.accept(frame([2],2,{'type':'length'}));s.accept('[DONE]');self.assertTrue(s.result()['complete'])
 def test_cumulative(self):
  s=StreamState(2);s.accept(frame([1],1));s.accept(frame([1,2],2,{'type':'length'}));s.accept('[DONE]');self.assertEqual(s.tokens,[1,2]);self.assertTrue(s.result()['complete'])
 def test_missing_done_retained(self):
  s=StreamState(2);s.accept(frame([1,2],2,{'type':'length'}));self.assertFalse(s.result()['complete']);self.assertEqual(len(s.frames),1)
 def test_missing_finish_retained(self):
  s=StreamState(2);s.accept(frame([1,2],2));s.accept('[DONE]');self.assertFalse(s.result()['complete'])
 def test_early_done_retained(self):
  s=StreamState(3);s.accept(frame([1],1));s.accept('[DONE]');self.assertFalse(s.result()['complete']);self.assertEqual(s.result()['received_tokens'],1)
 def test_bad_prefix(self):
  s=StreamState(3);s.accept(frame([1,2],2))
  with self.assertRaises(ValueError):s.accept(frame([4,5,6],3))
 def test_error_frame_preserved(self):
  s=StreamState(2)
  with self.assertRaises(ValueError):s.accept('{"error":"aborted"}')
  self.assertEqual(len(s.frames),1)
 def test_repeat_frame(self):
  s=StreamState(2);s.accept(frame([1],1));s.accept(frame([1],1));s.accept(frame([2],2,{'type':'length'}));s.accept('[DONE]');self.assertEqual(s.tokens,[1,2])
if __name__=='__main__':unittest.main()
