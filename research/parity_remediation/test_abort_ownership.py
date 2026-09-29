import asyncio,types,unittest
from abort_ownership import bind,ORIGINAL,transform

class Tasks:
 def __init__(self):self.tasks=[]
 def add_task(self,fn):self.tasks.append(fn)
 async def __call__(self):
  for fn in self.tasks:await fn()
class Request:
 def __init__(self,rid,children=None):self.rid=rid;self.children=children;self.is_single=children is None
 def __getitem__(self,i):return self.children[i]
class State:
 def __init__(self,obj,finished=False):self.obj=obj;self.finished=finished
async def scenario(patched,kind):
 old=Request('reuse');new=Request('reuse');states={};aborted=[]
 child=Request('keep');batch=Request(['reuse','keep'],[old,child])
 obj=batch if kind=='batch' else old
 states['reuse']=State(new if kind=='replaced_before' else old,finished=kind=='finished')
 if kind=='removed':states.clear()
 if kind=='batch':states['reuse']=State(new);states['keep']=State(child)
 async def sleep(delay):
  if delay!=2:raise AssertionError('delay changed')
  if kind=='replaced_during':states['reuse']=State(new)
  if kind=='same_object_new_state':states['reuse']=State(old)
  if kind=='removed_during':states.clear()
 manager=types.SimpleNamespace(rid_to_state=states,abort_request=aborted.append)
 fn=bind(dict(asyncio=types.SimpleNamespace(sleep=sleep),BackgroundTasks=Tasks,GenerateReqInput=Request),patched)
 await fn(manager,obj)()
 return aborted
class CleanupTests(unittest.IsolatedAsyncioTestCase):
 async def test_original_reproduces_replacement_abort_before_delay(self):self.assertEqual(await scenario(False,'replaced_before'),['reuse'])
 async def test_original_reproduces_replacement_abort_during_delay(self):self.assertEqual(await scenario(False,'replaced_during'),['reuse'])
 async def test_fixed_preserves_replacement_before(self):self.assertEqual(await scenario(True,'replaced_before'),[])
 async def test_fixed_preserves_replacement_during(self):self.assertEqual(await scenario(True,'replaced_during'),[])
 async def test_fixed_checks_state_epoch_not_only_object(self):self.assertEqual(await scenario(True,'same_object_new_state'),[])
 async def test_original_live_still_aborted(self):self.assertEqual(await scenario(True,'active'),['reuse'])
 async def test_finished_not_aborted(self):self.assertEqual(await scenario(True,'finished'),[])
 async def test_removed_noop(self):self.assertEqual(await scenario(True,'removed'),[])
 async def test_removed_during_delay_noop(self):self.assertEqual(await scenario(True,'removed_during'),[])
 async def test_batch_only_owned_state(self):self.assertEqual(await scenario(True,'batch'),['keep'])
 def test_source_drift_rejected(self):
  with self.assertRaises(ValueError):transform('class Test:\n    pass\n')
 def test_exact_transform(self):
  self.assertIn('state.obj is request_obj',transform('class Test:\n'+ORIGINAL))
if __name__=='__main__':unittest.main()
