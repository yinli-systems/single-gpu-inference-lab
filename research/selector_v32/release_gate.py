"""Fail-closed v3.2 gate over deployment-matched paired cells."""
from __future__ import annotations
import math
from typing import Any
import numpy as np

def require(x,msg):
 if not x:raise ValueError(msg)

def summary(indices,mode,cells,draws):
 if not indices:return dict(count=0,ratio=None,CI95=None,worst_point_ratio=None,simultaneous_worst_CI95=None,controls_failed=0)
 ratios=[float(cells[i]['comparisons'][mode]['ratio']) for i in indices]
 matrix=np.stack([np.asarray(draws[(i,mode)],dtype=float) for i in indices]);require(matrix.ndim==2 and matrix.shape[1]>=5000 and np.isfinite(matrix).all(),'invalid draws')
 aggregate=matrix.mean(0);minimum=matrix.min(0)
 return dict(count=len(indices),ratio=float(math.exp(sum(map(math.log,ratios))/len(ratios))),CI95=[float(x) for x in np.exp(np.quantile(aggregate,[.025,.975]))],worst_point_ratio=min(ratios),simultaneous_worst_CI95=[float(x) for x in np.exp(np.quantile(minimum,[.025,.975]))],controls_failed=sum(not cells[i]['comparisons'][mode]['controls_resolve'] for i in indices))

def evaluate(cells,draws,numerics):
 require(cells,'no cells');selected=[i for i,c in enumerate(cells) if c['selected']];graph=[i for i in selected if cexec(cells[i])=='graph16_replay'];unselected=[i for i,c in enumerate(cells) if not c['selected']]
 policy=summary(list(range(len(cells))),'guarded',cells,draws);sel=summary(selected,'guarded',cells,draws);g16=summary(graph,'guarded',cells,draws);unsel=summary(unselected,'guarded',cells,draws);off=summary(list(range(len(cells))),'off',cells,draws)
 exact=bool(numerics) and all(v['qualifications']==v['exact_full_outputs'] and v['max_abs_vs_pristine']==0 for v in numerics.values())
 req=dict(numerical_exact=exact,selected_nonempty=bool(selected),selected_controls_resolve=bool(selected) and sel['controls_failed']==0,graph16_selected_controls_resolve=bool(graph) and g16['controls_failed']==0,selected_worst_point_at_least_0_99=bool(selected) and sel['worst_point_ratio']>=.99,selected_joint_worst_lcb_at_least_0_99=bool(selected) and sel['simultaneous_worst_CI95'][0]>=.99,graph16_selected_aggregate_lcb_above_1=bool(graph) and g16['CI95'][0]>1,policy_worst_point_at_least_0_99=policy['worst_point_ratio']>=.99,off_overlay_worst_point_at_least_0_99=off['worst_point_ratio']>=.99)
 return {'pass':all(req.values()),'requirements':req,'policy':policy,'selected':sel,'graph16_selected':g16,'unselected':unsel,'off_overlay':off,'scope':'Frozen v3.2 geometries on one GPU family; paired process/block bootstrap; conditional on measured devices, not a hardware-population guarantee.'}

def cexec(c):return c['execution_mode']
