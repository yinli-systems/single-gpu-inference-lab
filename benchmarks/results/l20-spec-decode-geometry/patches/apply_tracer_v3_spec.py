#!/usr/bin/env python3
"""Tracer v3 (spec decode) on top of tracer v2: draft-pass CUDA timing in the runner
step trace, and a per-iteration spec accounting trace from the scheduler.

  VLLM_EXP_STEP_TRACE   (existing) runner rows gain "draft_ms" (CUDA time of speculator.propose)
                        and "draft_rows" (requests in the batch)
  VLLM_EXP_SPEC_TRACE   scheduler writes one JSON line per update_from_output call:
                        {"t", "acc": [[req_id_suffix, num_computed_tokens, num_draft, num_accepted], ...]}
usage: apply_tracer_v3_spec.py <site-packages/vllm>
"""
import sys
sp = sys.argv[1]

# --- runner: draft timing ---
p = f"{sp}/v1/worker/gpu/model_runner.py"
s = open(p).read()
if "draft_ms" not in s:
    a1 = "            with use_workspace_lane(self._draft_workspace_lane):\n                draft_tokens = self.speculator.propose(\n"
    assert s.count(a1) == 1
    s = s.replace(a1, "            if _EXP_STEP_TIMER is not None:\n                _EXP_STEP_TIMER.draft_begin(input_batch)\n" + a1, 1)
    a2 = "            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n"
    assert s.count(a2) == 1
    s = s.replace(a2, a2 + "            if _EXP_STEP_TIMER is not None:\n                _EXP_STEP_TIMER.draft_end()\n", 1)
    a3 = "    def end(self):\n        if self.current is None:\n"
    assert s.count(a3) == 1
    s = s.replace(a3, '''    def draft_begin(self, input_batch):
        if self.current is None:
            return
        ev = torch.cuda.Event(enable_timing=True); ev.record()
        self.current["_d0"] = ev
        self.current["draft_rows"] = int(getattr(input_batch, "num_reqs", 0))

    def draft_end(self):
        if self.current is None or "_d0" not in self.current:
            return
        ev = torch.cuda.Event(enable_timing=True); ev.record()
        self.current["_d1"] = ev

''' + a3, 1)
    # in drain: compute draft_ms when both events are present
    a4 = 'rec["cuda_ms"] = '
    idx = s.index(a4, s.index("def drain"))
    line_end = s.index("\n", idx)
    s = s[:line_end + 1] + '''            if "_d0" in rec and "_d1" in rec:
                rec["draft_ms"] = round(rec["_d0"].elapsed_time(rec["_d1"]), 3)
                rec.pop("_d0"); rec.pop("_d1")
''' + s[line_end + 1:]
    open(p, "w").write(s)
    print("runner: draft timing installed")
else:
    print("runner: already")

# --- scheduler: spec accounting ---
p = f"{sp}/v1/core/sched/scheduler.py"
s = open(p).read()
if "_EXP_SPEC_TRACE" not in s:
    a = "logger = init_logger(__name__)\n"
    assert s.count(a) == 1
    s = s.replace(a, a + '''
import json as _sjson
import os as _sos
import time as _stime
_EXP_SPEC_TRACE = _sos.environ.get("VLLM_EXP_SPEC_TRACE")
_EXP_SPEC_FH = None


def _exp_spec_write(rows):
    global _EXP_SPEC_FH
    if not rows:
        return
    if _EXP_SPEC_FH is None:
        _EXP_SPEC_FH = open(_EXP_SPEC_TRACE, "a", buffering=1)
    _EXP_SPEC_FH.write(_sjson.dumps({"t": _stime.monotonic(), "acc": rows}) + "\\n")
''', 1)
    a5 = "                num_draft_tokens = len(scheduled_spec_token_ids)\n                num_sampled = self.num_sampled_tokens_per_step\n                num_accepted = max(len(generated_token_ids) - num_sampled, 0)\n"
    assert s.count(a5) == 1
    s = s.replace(a5, a5 + "                if _EXP_SPEC_TRACE:\n                    _exp_spec_rows.append([req_id[-8:], request.num_computed_tokens, num_draft_tokens, num_accepted])\n", 1)
    # declare the list at the top of update_from_output and flush at the end of the loop
    a6 = "    ) -> dict[int, EngineCoreOutputs]:\n"
    i6 = s.index(a6, s.index("def update_from_output"))
    s = s[:i6 + len(a6)] + "        _exp_spec_rows = []\n" + s[i6 + len(a6):]
    a7 = "            # Free encoder inputs only after the step has actually executed.\n"
    assert s.count(a7) == 1
    # find the end of the for-loop: flush right before the first line after the loop that is at 8-space indent following a7; simplest: flush lazily at next call start
    s = s.replace("        _exp_spec_rows = []\n", "        _exp_spec_rows = []\n        if _EXP_SPEC_TRACE and getattr(self, '_exp_spec_pending', None):\n            _exp_spec_write(self._exp_spec_pending)\n            self._exp_spec_pending = None\n", 1)
    s = s.replace(a7, "            if _EXP_SPEC_TRACE and _exp_spec_rows:\n                self._exp_spec_pending = list(_exp_spec_rows)\n" + a7, 1)
    open(p, "w").write(s)
    print("scheduler: spec accounting installed")
else:
    print("scheduler: already")
