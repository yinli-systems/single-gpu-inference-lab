"""Freeze v3.2.2 release authorization from exposed canaries and completed causal diagnosis."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
QUALIFICATION_REVISION='3.2.2';MEASUREMENT_REVISION='3.2.1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(x,msg):
 if not x:raise RuntimeError(msg)
def min_group(result,scenario,timer,field):return min(float(g[field]) for g in result['groups'] if g['scenario']==scenario and g['timer']==timer)
def min_ci(result,scenario,timer):return min(float(g['conditional_CI95'][0]) for g in result['groups'] if g['scenario']==scenario and g['timer']==timer)
def evaluate_authorization(canaries,diagnosis,expected):
 need(set(canaries)=={'gpu_4090','gpu_5090'},'missing canary GPU');identities=set()
 canary_receipt={}
 for gpu,c in canaries.items():
  need(c['measurement_contract_revision']==MEASUREMENT_REVISION and c['stage']=='canary' and c['gpu']==gpu,'canary identity')
  need(c['canary_gate']['pass'] is False,'historical canary verdict must remain HOLD')
  r=c['canary_gate']['requirements'];need(r['numerical_exact'] and r['selected_nonempty'] and r['selected_point_worst_at_least_0_98'] and r['disabled_overlay_point_worst_at_least_0_98'],'canary prerequisite')
  identities.add((c['case_hash'],c['source_archive_sha256'],c['official_overlay_sha256']))
  canary_receipt[gpu]=dict(hardware=c['hardware'],historical_requirements=r,historical_pass=False)
 need(len(identities)==1,'canary provenance drift');case_hash,parent_archive,overlay=next(iter(identities))
 need(case_hash==expected['case_hash'] and parent_archive==expected['parent_source_archive_sha256'] and overlay==expected['official_overlay_sha256'],'frozen identity mismatch')
 need(diagnosis.get('all_failed_canaries_retained') is True and diagnosis.get('release_gate_pass') is False,'diagnosis scope')
 diag_receipt={}
 for gpu in ('gpu_4090','gpu_5090'):
  r=diagnosis['results'][gpu];need(r['noninjected_rows']==8640 and r['positive_controls']==15 and r['qualified_case_repeats']==15,'diagnostic matrix')
  need(r['fallback_graph_metadata_mismatches']==0,'fallback graph metadata mismatch')
  pooled=min_group(r,'independent','pooled_events','paired_wall_ratio');pooled_lcb=min_ci(r,'independent','pooled_events')
  wall=min_group(r,'independent','wall_only','paired_wall_ratio');wall_lcb=min_ci(r,'independent','wall_only')
  null=min_group(r,'same_graph_null','pooled_events','paired_wall_ratio');null_lcb=min_ci(r,'same_graph_null','pooled_events')
  need(pooled>=.995 and pooled_lcb>=.985 and wall>=.995 and wall_lcb>=.985 and null>=.995 and null_lcb>=.985,'diagnostic fallback equivalence')
  driver=list(canaries[gpu]['hardware'].values())[0]['driver'];need(driver in r['hardware'],'diagnosis/canary driver mismatch')
  diag_receipt[gpu]=dict(independent_pooled_worst=pooled,independent_pooled_min_lcb=pooled_lcb,independent_wall_worst=wall,independent_wall_min_lcb=wall_lcb,same_graph_null_pooled_worst=null,same_graph_null_pooled_min_lcb=null_lcb,hardware=r['hardware'])
 return dict(release_qualification_authorized=True,qualification_revision=QUALIFICATION_REVISION,measurement_contract_revision=MEASUREMENT_REVISION,case_hash=case_hash,parent_source_archive_sha256=parent_archive,official_overlay_sha256=overlay,canaries=canary_receipt,diagnosis=diag_receipt,old_canary_gate_remains_hold=True,absolute_whole_policy_is_audit_only=True,release_cases_consumed_at_authorization=0,default_promotion=False,serving_promotion=False,historical_token_divergence_resolved=False)
def authorize(root):
 root=Path(root);inputs=json.loads((root/'receipts'/'authorization-inputs.json').read_text())
 for name,h in inputs['files'].items():need(sha(root/'receipts'/name)==h,'authorization input hash '+name)
 canaries={gpu:json.loads((root/'receipts'/inputs['canaries'][gpu]).read_text()) for gpu in ('gpu_4090','gpu_5090')};diagnosis=json.loads((root/'receipts'/inputs['diagnosis']).read_text())
 out=evaluate_authorization(canaries,diagnosis,inputs);need(out['release_cases_consumed_at_authorization']==0,'freshness');return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);print(json.dumps(authorize(p.parse_args().root),indent=2))
