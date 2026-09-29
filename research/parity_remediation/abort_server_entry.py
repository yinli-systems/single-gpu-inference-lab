"""Owned-server-only cleanup instrumentation and optional source-bound replacement."""
import asyncio,contextvars,hashlib,inspect,json,os,runpy,time
from pathlib import Path

def install():
    from fastapi import BackgroundTasks
    from sglang.srt.managers.io_struct import GenerateReqInput
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from abort_ownership import ORIGINAL,bind
    source=inspect.getsource(TokenizerManager.create_abort_task)
    if source!=ORIGINAL:raise RuntimeError('unreviewed installed cleanup source')
    mode=os.environ['SGI_ABORT_MODE'];path=Path(os.environ['SGI_ABORT_LOG']);path.parent.mkdir(parents=True,exist_ok=True)
    context=contextvars.ContextVar('sgi_abort_owner',default=None)
    def emit(record):
        with path.open('a') as f:f.write(json.dumps(dict(time=time.monotonic(),pid=os.getpid(),**record))+'\n')
    init=TokenizerManager._init_req_state
    def initialize(self,obj,request=None):
        result=init(self,obj,request)
        objects=[obj] if not hasattr(obj,'is_single') or obj.is_single else [obj[i] for i in range(len(obj.rid))]
        for item in objects:
            state=self.rid_to_state.get(item.rid)
            emit(dict(event='register',rid=item.rid,object_id=id(item),state_id=id(state)))
        return result
    abort=TokenizerManager.abort_request
    def observed_abort(self,rid='',abort_all=False):
        state=self.rid_to_state.get(rid)
        emit(dict(event='abort',rid=rid,abort_all=abort_all,cleanup_owner=context.get(),
            target_object=id(state.obj) if state is not None else None,
            target_state=id(state) if state is not None else None))
        return abort(self,rid=rid,abort_all=abort_all)
    original=TokenizerManager.create_abort_task
    selected=original if mode=='original' else bind(dict(asyncio=asyncio,BackgroundTasks=BackgroundTasks,GenerateReqInput=GenerateReqInput),True)
    def create(self,obj):
        tasks=selected(self,obj)
        for task in tasks.tasks:
            fn=task.func
            async def invoke(fn=fn,obj=obj):
                token=context.set(id(obj));emit(dict(event='cleanup-start',rid=obj.rid,owner=id(obj)))
                try:return await fn()
                finally:emit(dict(event='cleanup-end',rid=obj.rid,owner=id(obj)));context.reset(token)
            task.func=invoke
        return tasks
    TokenizerManager._init_req_state=initialize
    TokenizerManager.abort_request=observed_abort
    TokenizerManager.create_abort_task=create
    emit(dict(event='installed',mode=mode,original_method_sha256=hashlib.sha256(source.encode()).hexdigest()))
install()
if __name__=='__main__':runpy.run_module('sglang.launch_server',run_name='__main__')
