"""Apply tracer v2 to an installed vLLM by anchors (robust to line offsets)."""
import pathlib, re, sys
SP = pathlib.Path(sys.argv[1])
src = pathlib.Path(sys.argv[2]).read_text()  # the v2 diff, used only to lift helper blocks

def block(diff, start_marker, end_marker):
    lines = [l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    txt = "\n".join(lines)
    a = txt.index(start_marker); b = txt.index(end_marker, a)
    return txt[a:b]

# ---- core.py ----
p = SP / "vllm/v1/engine/core.py"; s = p.read_text()
if "_exp_trace_iteration" not in s:
    helper = "\n" + block(src, "import json as _json", "        if _EXP_ITER_TRACE and scheduler_output is not None:").rstrip() + "\n"
    anchor = "\nlogger = init_logger(__name__)\n"
    assert s.count(anchor) == 1; s = s.replace(anchor, anchor + helper, 1)
    old = "        iteration_details.elapsed_ms = (time.monotonic() - start_time) * 1000\n        self._iteration_index = iteration_index + 1\n"
    assert s.count(old) == 1
    s = s.replace(old, old + "        if _EXP_ITER_TRACE and scheduler_output is not None:\n            _exp_trace_iteration(scheduler_output, iteration_details, start_time)\n")
    p.write_text(s); print("core.py patched")
else:
    print("core.py already patched")

# ---- model_runner.py ----
p = SP / "vllm/v1/worker/gpu/model_runner.py"; s = p.read_text()
if "_EXP_STEP_TIMER" not in s:
    helper = "\n" + block(src, "import json as _json\nimport os as _os\nimport time as _time", "_EXP_STEP_TIMER = _ExpStepTimer(_EXP_STEP_TRACE) if _EXP_STEP_TRACE else None") + "_EXP_STEP_TIMER = _ExpStepTimer(_EXP_STEP_TRACE) if _EXP_STEP_TRACE else None\n"
    anchor = "\nlogger = init_logger(__name__)\n"
    assert s.count(anchor) == 1; s = s.replace(anchor, anchor + helper, 1)
    # anchor: right after the zero-token early return that follows dispatch_cg_and_sync_dp
    old = ("            return self._merge_ec_connector_no_forward(scheduler_output, empty_output)\n\n"
           "        if not dummy_run:\n            # Common case.\n")
    assert s.count(old) == 1, s.count(old)
    s = s.replace(old, ("            return self._merge_ec_connector_no_forward(scheduler_output, empty_output)\n\n"
                        "        if not dummy_run and _EXP_STEP_TIMER is not None:\n            _EXP_STEP_TIMER.begin(scheduler_output, batch_desc)\n"
                        "        if not dummy_run:\n            # Common case.\n"))
    m = re.search(r"    def sample_tokens\(\n        self, grammar_output: GrammarOutput \| None\n    \) -> AsyncOutput \| ModelRunnerOutput \| None:\n", s)
    assert m, "sample_tokens signature not found"
    wrapper = ('    def sample_tokens(\n        self, grammar_output: GrammarOutput | None\n    ) -> AsyncOutput | ModelRunnerOutput | None:\n'
               '        if _EXP_STEP_TIMER is None:\n            return self._sample_tokens_impl(grammar_output)\n'
               '        try:\n            return self._sample_tokens_impl(grammar_output)\n        finally:\n            _EXP_STEP_TIMER.end()\n\n'
               '    def _sample_tokens_impl(\n        self, grammar_output: GrammarOutput | None\n    ) -> AsyncOutput | ModelRunnerOutput | None:\n')
    s = s[:m.start()] + wrapper + s[m.end():]
    p.write_text(s); print("model_runner.py patched")
else:
    print("model_runner.py already patched")
