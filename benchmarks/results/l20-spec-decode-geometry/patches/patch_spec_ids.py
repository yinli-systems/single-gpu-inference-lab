import sys
p=sys.argv[1]; s=open(p).read()
old="                    _exp_spec_rows.append([req_id[-8:], request.num_computed_tokens, num_draft_tokens, num_accepted])\n"
new="                    _exp_spec_rows.append([req_id[-8:], request.num_computed_tokens, num_draft_tokens, num_accepted, list(scheduled_spec_token_ids), list(generated_token_ids)])\n"
assert s.count(old)==1 or "list(scheduled_spec_token_ids)" in s
if s.count(old)==1: open(p,'w').write(s.replace(old,new)); print("ids added")
else: print("already")
