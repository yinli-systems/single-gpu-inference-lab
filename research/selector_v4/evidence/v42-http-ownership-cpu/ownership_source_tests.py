"""Run upstream registered ownership tests against exact production method spans.

This lightweight harness avoids SGLang's GPU import dependency on macOS; it is
source-exact behavioral evidence, not a claim that full upstream imports passed.
"""
from pathlib import Path
import argparse, ast, asyncio, copy, dataclasses, gc, hashlib, json, logging, os, pickle, subprocess, sys, types, unittest, uuid, weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

ROOT=Path(os.environ.get('SGI_OWNER_SOURCE_ROOT','/Users/kevin/Projects/sglang-http-ownership-upstream-20260930'))
TARGET='python/sglang/srt/managers/tokenizer_manager.py'
TEST='test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py'

@dataclasses.dataclass
class ReqState:
    out_list: list
    finished: bool
    event: object
    obj: object
    time_stats: object
    abort_generation: object = dataclasses.field(default_factory=object)
    dispatched: bool = False

class Stats:
    def __init__(self, **kwargs): pass
    def set_created_time(self, value): pass
    def set_finished_time(self): pass

class GenerateReqInput:
    rid=None; is_single=True
    def __init__(self, **kwargs):
        self.__dict__.update(received_time=0.0,stream=False,return_prompt_token_ids=False,external_trace_header=None,bootstrap_room=None)
        self.__dict__.update(kwargs)
    def __getitem__(self, index):
        cache=self.__dict__.setdefault('_sub_obj_cache',{})
        if index not in cache:cache[index]=GenerateReqInput(rid=self.rid[index],input_ids=self.input_ids[index],received_time=self.received_time)
        return cache[index]
    def normalize_batch_and_arguments(self):
        self.is_single=False;self.batch_size=1;self.parallel_sample_num=self.sampling_params['n']
        self.rid=[uuid.uuid4().hex for _ in range(self.parallel_sample_num)]
        self.input_ids=[self.input_ids]*self.parallel_sample_num
    def regenerate_rid(self): self.rid=uuid.uuid4().hex;return self.rid

class Tasks:
    def __init__(self): self.tasks=[]
    def add_task(self, fn): self.tasks.append(fn)
    async def __call__(self):
        for fn in self.tasks: await fn()


def extract_methods(source):
    tree=ast.parse(source)
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TokenizerManager')
    methods=[n for n in klass.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in ('create_abort_task','_init_req_state','_handle_batch_request','_release_req_states_on_failure')]
    return '\n'.join(ast.get_source_segment(source,n) for n in methods)


def run(baseline=False, baseline_file=None, regressions_only=False):
    source=(Path(baseline_file).read_text() if baseline_file else subprocess.check_output(['git','show','085c26f:'+TARGET],cwd=ROOT,text=True)) if baseline else (ROOT/TARGET).read_text()
    test=(ROOT/TEST).read_text(); namespace=dict(globals(),logger=logging.getLogger(__name__),BackgroundTasks=Tasks,APIServerReqTimeStats=Stats,EmbeddingReqInput=GenerateReqInput,fastapi=types.SimpleNamespace(Request=object),Optional=__import__('typing').Optional,Union=__import__('typing').Union)
    code='from __future__ import annotations\n'+extract_methods(source)
    exec(compile(code,str(ROOT/TARGET),'exec'),namespace)
    Manager=type('TokenizerManager',(),{name:namespace[name] for name in ('create_abort_task','_init_req_state','_handle_batch_request','_release_req_states_on_failure')})
    def make_tm(case):
        if sys.version_info < (3,10):
            try: asyncio.get_event_loop()
            except RuntimeError:
                loop=asyncio.new_event_loop();asyncio.set_event_loop(loop);case.addCleanup(loop.close)
        tm=Manager();tm.rid_to_state={};tm.encoder_dispatch_ready={};tm.enable_trace=False;tm.disaggregation_mode='none';return tm
    def make_state(rid='test_rid'):
        obj=Mock(spec=GenerateReqInput);obj.rid=rid;obj.stream=False
        return ReqState([],False,asyncio.Event(),obj,Stats())
    namespace.update(_make_tokenizer_manager=make_tm,_make_req_state=make_state,CustomTestCase=unittest.TestCase)
    tree=ast.parse(test)
    chosen=[n for n in tree.body if getattr(n,'name',None) in ('_make_generate_obj','TestDelayedAbortOwnership')]
    for n in chosen: exec(compile(ast.Module(body=[n],type_ignores=[]),str(ROOT/TEST),'exec'),namespace)
    module=types.ModuleType('sglang.srt.managers.tokenizer_manager');module.asyncio=asyncio
    names=['sglang','sglang.srt','sglang.srt.managers',module.__name__]
    for name in names: sys.modules[name]=module if name==module.__name__ else types.ModuleType(name)
    for parent,child in zip(names,names[1:]): setattr(sys.modules[parent],child.rsplit('.',1)[1],sys.modules[child])
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(namespace['TestDelayedAbortOwnership'])
    if regressions_only:
        suite=unittest.TestSuite(t for t in suite if t._testMethodName in ('test_parallel_sampling_handler_registers_expanded_requests','test_delayed_cleanup_does_not_retain_completed_state'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt={'baseline':baseline,'regressions_only':regressions_only,'test_count':result.testsRun,'failure_count':len(result.failures),'error_count':len(result.errors),'pass':result.wasSuccessful(),'method_sha256':hashlib.sha256(code.encode()).hexdigest(),'source_file_sha256':hashlib.sha256(source.encode()).hexdigest(),'test_file_sha256':hashlib.sha256(test.encode()).hexdigest(),'validation':'exact production method spans with minimal state/time-stat dependencies; full upstream imports not claimed'}
    print(json.dumps(receipt,indent=2));return result.wasSuccessful()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',action='store_true');p.add_argument('--baseline-file');p.add_argument('--regressions-only',action='store_true');a=p.parse_args();sys.exit(0 if run(a.baseline,a.baseline_file,a.regressions_only) else 1)
