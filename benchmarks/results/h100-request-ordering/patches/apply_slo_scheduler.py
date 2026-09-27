"""Append exp_slo_scheduler.py to an installed vLLM 0.29 scheduler.py (idempotent).

  python apply_slo_scheduler.py SITE_PACKAGES

The appended code is inert unless VLLM_EXP_SLO_ORDER=1 is set in the engine's environment.
"""
import pathlib
import sys

sp = pathlib.Path(sys.argv[1])
target = sp / "vllm/v1/core/sched/scheduler.py"
src = (pathlib.Path(__file__).parent / "exp_slo_scheduler.py").read_text()
text = target.read_text()
if "_exp_slo_reorder" in text:
    print("scheduler.py already patched")
else:
    for needed in ("class Scheduler(", "SchedulingPolicy"):
        assert needed in text, f"anchor {needed!r} not found"
    target.write_text(text.rstrip("\n") + "\n\n\n" + src)
    print("scheduler.py patched")
