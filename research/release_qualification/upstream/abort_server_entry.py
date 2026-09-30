"""Read-only request registration/abort identity trace for current-main validation."""
import json,os,runpy,time
from pathlib import Path

def install():
 from sglang.srt.managers.tokenizer_manager import TokenizerManager
 path=Path(os.environ['SGI_ABORT_LOG']);path.parent.mkdir(parents=True,exist_ok=True)
 def emit(x):
  with path.open('a') as f:f.write(json.dumps(dict(time=time.monotonic(),pid=os.getpid(),**x))+'\n')
 init=TokenizerManager._init_req_state
 def initialize(self,obj,request=None):
  result=init(self,obj,request);items=[obj] if not hasattr(obj,'is_single') or obj.is_single else [obj[i] for i in range(len(obj.rid))]
  for item in items:
   state=self.rid_to_state.get(item.rid);emit(dict(event='register',rid=item.rid,object_id=id(item),state_id=id(state)))
  return result
 abort=TokenizerManager.abort_request
 def observed(self,rid='',abort_all=False):
  state=self.rid_to_state.get(rid);emit(dict(event='abort',rid=rid,abort_all=abort_all,target_object=id(state.obj) if state else None,target_state=id(state) if state else None))
  return abort(self,rid=rid,abort_all=abort_all)
 TokenizerManager._init_req_state=initialize;TokenizerManager.abort_request=observed;emit(dict(event='installed'))
install()
if __name__=='__main__':runpy.run_module('sglang.launch_server',run_name='__main__')
