import sys
p=f"{sys.argv[1]}/v1/worker/gpu/spec_decode/adaptive_verification.py"; s=open(p).read()
if "VLLM_EXP_FORCE_AV_BUDGET" in s: print("already"); sys.exit(0)
a="        draft_budget = int(np.argmax(num_tokens_to_estimated_accepted_tokens / costs))\n"
assert s.count(a)==1
s=s.replace(a, a+"        _f = _os_env_force_budget()\n        if _f is not None:  # experiment: hindsight budget sweep\n            draft_budget = min(_f, int(max_draft_budget))\n",1)
s=s.replace("logger = init_logger(__name__)\n","logger = init_logger(__name__)\n\n\ndef _os_env_force_budget():\n    import os as _o\n    v = _o.environ.get('VLLM_EXP_FORCE_AV_BUDGET')\n    return int(v) if v else None\n",1)
open(p,'w').write(s); print("force-budget override installed")
