import sys
p=sys.argv[1]; s=open(p).read()
if '"exp": getattr(scheduler_output' in s: print("already"); sys.exit(0)
a='                "attn_proxy": attn_proxy,\n'
assert s.count(a)==1
s=s.replace(a, a+'                "exp": getattr(scheduler_output, "exp_decision", None),\n')
open(p,'w').write(s); print("tracer exp field added")
