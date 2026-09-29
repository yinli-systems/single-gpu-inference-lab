"""Test-scoped proposal: delayed cleanup may abort only its original request state.
Related prior art: SGLang PR38850 identifies reused-ID cleanup in its test harness.
This is not claimed as a novel defect or as a fix for token-value divergence.
"""
from __future__ import annotations
import ast,difflib,hashlib,textwrap
from pathlib import Path

ORIGINAL = """    def create_abort_task(self, obj: GenerateReqInput):
        # Abort the request if the client is disconnected.
        async def abort_request():
            await asyncio.sleep(2)
            rids = [obj.rid] if obj.is_single else obj.rid
            for rid in rids:
                if rid in self.rid_to_state:
                    self.abort_request(rid)

        background_tasks = BackgroundTasks()
        background_tasks.add_task(abort_request)
        return background_tasks
"""

REPLACEMENT = """    def create_abort_task(self, obj: GenerateReqInput):
        # Bind cleanup to the originating objects, never just a reusable RID.
        async def abort_request():
            requests = [obj] if obj.is_single else [obj[i] for i in range(len(obj.rid))]
            states = []
            for request_obj in requests:
                state = self.rid_to_state.get(request_obj.rid)
                if state is not None and state.obj is request_obj:
                    states.append((request_obj.rid, state))
            await asyncio.sleep(2)
            for rid, state in states:
                if self.rid_to_state.get(rid) is state and not state.finished:
                    self.abort_request(rid)

        background_tasks = BackgroundTasks()
        background_tasks.add_task(abort_request)
        return background_tasks
"""

def transform(source):
    if source.count(ORIGINAL)!=1:raise ValueError('unreviewed cleanup function')
    changed=source.replace(ORIGINAL,REPLACEMENT)
    ast.parse(changed)
    return changed

def bind(namespace,patched=True):
    code=textwrap.dedent(REPLACEMENT if patched else ORIGINAL)
    exec(compile(code,'<abort-ownership-proposal>','exec'),namespace)
    return namespace['create_abort_task']

if __name__=='__main__':
    import argparse,json
    a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);a.add_argument('--out',type=Path,required=True);x=a.parse_args()
    if x.out.exists():raise FileExistsError('preserve source evidence')
    old=x.source.read_text();new=transform(old);x.out.mkdir(parents=True)
    (x.out/'tokenizer-abort-ownership.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/python/sglang/srt/managers/tokenizer_manager.py',tofile='b/python/sglang/srt/managers/tokenizer_manager.py')))
    (x.out/'binding.json').write_text(json.dumps(dict(original_sha256=hashlib.sha256(old.encode()).hexdigest(),candidate_sha256=hashlib.sha256(new.encode()).hexdigest(),original_method_sha256=hashlib.sha256(ORIGINAL.encode()).hexdigest(),runtime_modified=False,upstream_accepted=False),indent=2)+'\n')
