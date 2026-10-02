"""Lightweight training-only geometry-held-out prior; advisory, never authorises cap."""
import numpy as np

FEATURES=('batch','max_cached','descriptor_count','kv_heads','sm_count',
          'smem_per_sm','dtype_bf16','layout_paged','actual_split','graph_replays')


def _fit(x,y):
    center=x.mean(axis=0);scale=x.std(axis=0);scale[scale==0]=1
    z=np.column_stack((np.ones(len(x)),(x-center)/scale));weights=np.zeros(z.shape[1])
    for _ in range(1500):
        p=1/(1+np.exp(-np.clip(z@weights,-30,30)))
        penalty=.01*weights;penalty[0]=0
        weights-=.15*(z.T@(p-y)/len(y)+penalty)
    return center,scale,weights


def _predict(x,center,scale,weights):
    z=np.column_stack((np.ones(len(x)),(x-center)/scale))
    return 1/(1+np.exp(-np.clip(z@weights,-30,30)))


def fit_training_prior(rows,*,environment_key):
    if not rows or not environment_key:raise ValueError('Bound training rows required')
    if any(r['role']!='train' or r['environment_key']!=environment_key for r in rows):
        raise ValueError('Scoring/held-out-policy data and other environments cannot fit a prior')
    groups=sorted({r['geometry_id'] for r in rows})
    base={'environment_key':environment_key,'feature_schema':list(FEATURES),
          'geometry_groups':groups,'training_rows':len(rows),'qualification_authority':False,
          'pruning_validated':False,'serving_promotion':False,'default_promotion':False}
    if len(groups)<5:
        return {**base,'status':'INSUFFICIENT_GEOMETRY_SUPPORT','model':None,
                'reason':'At least five training geometries for leave-one-geometry-out analysis; two exposed shapes cannot validate extrapolation'}
    x=np.asarray([r['features'] for r in rows],dtype=float);y=np.asarray([r['wins_by_one_percent'] for r in rows],dtype=float)
    if x.shape!=(len(rows),len(FEATURES)) or not np.isfinite(x).all() or not np.isin(y,[0,1]).all():
        raise ValueError('Malformed structural training features or labels')
    predictions=[]
    for group in groups:
        mask=np.asarray([r['geometry_id']!=group for r in rows]);center,scale,weights=_fit(x[mask],y[mask])
        p=_predict(x[~mask],center,scale,weights)
        predictions.extend({'geometry_id':group,'probability':float(a),'label':bool(b)} for a,b in zip(p,y[~mask]))
    center,scale,weights=_fit(x,y)
    model={'center':center.tolist(),'scale':scale.tolist(),'weights':weights.tolist(),
           'feature_min':x.min(axis=0).tolist(),'feature_max':x.max(axis=0).tolist()}
    return {**base,'status':'FITTED_ADVISORY_ONLY','model':model,
            'leave_one_geometry_out':predictions,
            'probability_scope':'Regularised logistic model score; not a calibrated slowdown-risk probability'}


def advise(model,*,environment_key,features,static_eligible=True):
    result={'action':'calibrate','probability':None,'qualification_authority':False,
            'resource_authorised':False,'reason':'uncertain_or_unvalidated_prior'}
    if not static_eligible:return {**result,'action':'native','reason':'static_ineligible'}
    if model.get('environment_key')!=environment_key:
        return {**result,'action':'native','reason':'environment_identity_mismatch'}
    if model.get('status')!='FITTED_ADVISORY_ONLY':return result
    x=np.asarray(features,dtype=float);m=model['model']
    if x.shape!=(len(FEATURES),) or not np.isfinite(x).all():raise ValueError('Invalid prior query')
    if np.any(x<np.asarray(m['feature_min'])) or np.any(x>np.asarray(m['feature_max'])):return result
    p=float(_predict(x[None],np.asarray(m['center']),np.asarray(m['scale']),np.asarray(m['weights']))[0])
    # Even >0.9 cannot authorise Resource; <0.1 cannot prune until validation.
    return {**result,'probability':p,'proposed_native_pruning':p<.1,
            'reason':'empirical_first_use_calibration_required'}
