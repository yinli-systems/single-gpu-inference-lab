#!/usr/bin/env python3
"""Tracer v4: record the adaptive-verification decision in the runner step trace.
Adds to each step row (when adaptive verification is on):
  av_budget      chosen draft-slot budget, av_max_budget, av_num_reqs,
  av_non_draft   non-draft target tokens, av_pred_ms predicted step cost of the chosen budget,
  av_pred_draft_ms / av_pred_verify_ms  the two table terms,
  av_est_accepted estimated accepted tokens at the chosen budget
usage: apply_tracer_v4_adaptive.py <site-packages/vllm>
"""
import sys
sp = sys.argv[1]
p = f"{sp}/v1/worker/gpu/spec_decode/adaptive_verification.py"
s = open(p).read()
if "av_pred_ms" in s:
    print("already"); sys.exit(0)
a = "        draft_budget = int(np.argmax(num_tokens_to_estimated_accepted_tokens / costs))\n"
assert s.count(a) == 1
s = s.replace(a, a + '''        self._exp_pending_decision = {  # experiment: picked up by the runner step tracer at begin()
            "av_budget": draft_budget, "av_max_budget": int(max_draft_budget), "av_num_reqs": int(num_reqs),
            "av_non_draft": int(num_non_draft_tokens_total), "av_pred_ms": float(costs[draft_budget]),
            "av_pred_draft_ms": float(draft_cost_ms[len(req_ids)]),
            "av_pred_verify_ms": float(verify_cost_ms[num_non_draft_tokens_total + draft_budget]),
            "av_est_accepted": float(num_tokens_to_estimated_accepted_tokens[draft_budget]),
        }
''', 1)
open(p, "w").write(s)
print("adaptive decision stashed")

# runner: hand the stashed decision to the step timer at begin()
p = f"{sp}/v1/worker/gpu/model_runner.py"
s = open(p).read()
if "_exp_pending_decision" not in s:
    a = "            _EXP_STEP_TIMER.begin(scheduler_output, batch_desc)\n"
    assert s.count(a) == 1
    s = s.replace(a, "            _EXP_STEP_TIMER.begin(scheduler_output, batch_desc)\n            if self.adaptive_verification is not None and getattr(self.adaptive_verification, '_exp_pending_decision', None):\n                _EXP_STEP_TIMER.current.update(self.adaptive_verification._exp_pending_decision)\n                self.adaptive_verification._exp_pending_decision = None\n", 1)
    open(p, "w").write(s)
    print("runner: decision pickup installed")
else:
    print("runner: already")
