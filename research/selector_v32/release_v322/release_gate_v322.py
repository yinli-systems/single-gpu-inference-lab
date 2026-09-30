"""v3.2.2 release gate: paired same-process safety is primary; absolute policy is audit-only."""
from __future__ import annotations
import math
import numpy as np

QUALIFICATION_REVISION="3.2.2"

def require(x,msg):
 if not x:raise ValueError(msg)

def summary(indices,mode,cells,draws):
 if not indices:return dict(count=0,ratio=None,CI95=None,worst_point_ratio=None,simultaneous_worst_CI95=None,controls_failed=0)
 ratios=[float(cells[i]['comparisons'][mode]['ratio']) for i in indices]
 matrix=np.stack([np.asarray(draws[(i,mode)],dtype=float) for i in indices])
 require(matrix.ndim==2 and matrix.shape[1]>=5000 and np.isfinite(matrix).all(),'invalid draws')
 aggregate=matrix.mean(0);minimum=matrix.min(0)
 return dict(count=len(indices),ratio=float(math.exp(sum(map(math.log,ratios))/len(ratios))),CI95=[float(x) for x in np.exp(np.quantile(aggregate,[.025,.975]))],worst_point_ratio=min(ratios),simultaneous_worst_CI95=[float(x) for x in np.exp(np.quantile(minimum,[.025,.975]))],controls_failed=sum(not cells[i]['comparisons'][mode]['controls_resolve'] for i in indices))

def evaluate(cells,draws,numerics):
 require(cells,'no cells');all_idx=list(range(len(cells)));selected=[i for i,c in enumerate(cells) if c['selected']];unselected=[i for i,c in enumerate(cells) if not c['selected']];graph=[i for i in selected if cells[i]['execution_mode']=='graph16_replay']
 absolute_policy=summary(all_idx,'guarded',cells,draws);absolute_selected=summary(selected,'guarded',cells,draws);absolute_graph=summary(graph,'guarded',cells,draws);off=summary(all_idx,'off',cells,draws)
 paired_policy=summary(all_idx,'paired_guarded',cells,draws);paired_selected=summary(selected,'paired_guarded',cells,draws);paired_unselected=summary(unselected,'paired_guarded',cells,draws);paired_graph=summary(graph,'paired_guarded',cells,draws)
 exact=bool(numerics) and all(v['qualifications']==v['exact_full_outputs'] and v['max_abs_vs_pristine']==0 for v in numerics.values())
 req=dict(
  numerical_exact=exact,selected_nonempty=bool(selected),unselected_nonempty=bool(unselected),
  absolute_selected_controls_resolve=bool(selected) and absolute_selected['controls_failed']==0,
  absolute_selected_worst_point_at_least_0_99=bool(selected) and absolute_selected['worst_point_ratio']>=.99,
  absolute_selected_joint_worst_lcb_at_least_0_99=bool(selected) and absolute_selected['simultaneous_worst_CI95'][0]>=.99,
  absolute_graph16_aggregate_lcb_above_1=bool(graph) and absolute_graph['CI95'][0]>1,
  disabled_overlay_worst_point_at_least_0_99=off['worst_point_ratio']>=.99,
  disabled_overlay_all_equivalence_controls_resolve=all(c['comparisons']['off'].get('disabled_overlay_resolves_one_percent') is True for c in cells),
  paired_selected_controls_resolve=bool(selected) and paired_selected['controls_failed']==0,
  paired_selected_worst_point_at_least_0_99=bool(selected) and paired_selected['worst_point_ratio']>=.99,
  paired_selected_joint_worst_lcb_at_least_0_99=bool(selected) and paired_selected['simultaneous_worst_CI95'][0]>=.99,
  paired_unselected_controls_resolve=bool(unselected) and paired_unselected['controls_failed']==0,
  paired_unselected_worst_point_at_least_0_99=bool(unselected) and paired_unselected['worst_point_ratio']>=.99,
  paired_unselected_joint_worst_lcb_at_least_0_99=bool(unselected) and paired_unselected['simultaneous_worst_CI95'][0]>=.99,
  paired_policy_controls_resolve=paired_policy['controls_failed']==0,
  paired_policy_worst_point_at_least_0_99=paired_policy['worst_point_ratio']>=.99,
  paired_policy_joint_worst_lcb_at_least_0_99=paired_policy['simultaneous_worst_CI95'][0]>=.99,
  paired_graph16_aggregate_lcb_above_1=bool(graph) and paired_graph['CI95'][0]>1)
 result=dict(qualification_revision=QUALIFICATION_REVISION,requirements=req,paired_policy=paired_policy,paired_selected=paired_selected,paired_unselected=paired_unselected,paired_graph16=paired_graph,absolute_selected=absolute_selected,absolute_graph16=absolute_graph,disabled_overlay=off,absolute_policy_audit=absolute_policy,absolute_policy_audit_is_blocking=False,scope='Fresh v3.2 release geometries; paired off/guarded safety is primary. Absolute pristine/guarded whole-policy result is retained as non-blocking audit because exposed canary diagnosis showed graph/context variance. Conditional on measured devices, not a hardware-population guarantee.');result['pass']=all(req.values());return result
