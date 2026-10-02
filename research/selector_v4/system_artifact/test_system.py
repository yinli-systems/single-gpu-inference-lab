import math

import numpy as np
import pytest

from research.selector_v4.system_artifact.metrology import describe, regret_summary
from research.selector_v4.system_artifact.prior import FEATURES, advise, fit_training_prior
from research.selector_v4.system_artifact.risk import (
    constrained_research_choice,
    tail_diagnostic,
    upper_binomial,
    zero_failure_trials,
)


def test_process_uncertainty_survives_constant_blocks():
    cube=np.repeat(np.array([.95,1.,1.05])[:,None,None],24,axis=1)
    result=describe(cube,seed=17)
    assert result['hierarchical_ci95'][0]<.97 and result['hierarchical_ci95'][1]>1.03
    assert result['real_independent_processes']==3 and result['within_process_block_mean_log_variance']<1e-30


def test_balanced_windows_are_not_independent_replicates():
    cube=np.tile([.5,2.],(3,24,1));r=describe(cube,seed=17)
    assert r['geomean']==1 and r['hierarchical_ci95']==[1,1]
    assert r['within_block_paired_window_log_variance']>0 and not r['window_resampling']


def test_regret_keeps_signed_noise_and_percent_units():
    r=regret_summary([-.001,0,.006,.02,.06],definition='signed measured difference')
    assert r['negative_observation_count']==1 and r['above_0_5_percent_count']==3
    assert r['above_1_percent_count']==2 and r['above_5_percent_count']==1
    assert math.isclose(r['median'],.006)


def test_zero_failures_need_real_independent_trials():
    assert zero_failure_trials()==2995
    assert upper_binomial(0,5)>.45
    assert upper_binomial(0,2994)>.001 and upper_binomial(0,2995)<.001
    assert upper_binomial(1,10)>upper_binomial(0,10) and upper_binomial(10,10)==1


def test_risk_gate_does_not_turn_bootstrap_draws_into_trials():
    tail=tail_diagnostic([{'process_id':str(i),'slowdown_event':False} for i in range(3)])
    r=constrained_research_choice(base_checks={'exact':True},tail=tail,static_eligible=True)
    assert r['choice']=='native' and not r['default_promotion']
    with pytest.raises(ValueError):tail_diagnostic([{'process_id':'same','slowdown_event':False}]*2)


def rows(groups=2):
    return [{'role':'train','environment_key':'gpu-A-eager','geometry_id':str(g),
             'features':[float(g)]*len(FEATURES),'wins_by_one_percent':g%2==0} for g in range(groups)]


def test_two_exposed_shapes_cannot_validate_prior_or_prune():
    m=fit_training_prior(rows(),environment_key='gpu-A-eager')
    assert m['status']=='INSUFFICIENT_GEOMETRY_SUPPORT'
    assert advise(m,environment_key='gpu-A-eager',features=[0]*len(FEATURES))['action']=='calibrate'


def test_prior_rejects_policy_leakage_and_environment_alias():
    r=rows();r[0]['role']='policy'
    with pytest.raises(ValueError):fit_training_prior(r,environment_key='gpu-A-eager')
    m=fit_training_prior(rows(),environment_key='gpu-A-eager')
    assert advise(m,environment_key='gpu-B-eager',features=[0]*len(FEATURES))['action']=='native'


def test_fitted_prior_never_authorises_resource_or_unvalidated_pruning():
    m=fit_training_prior(rows(8),environment_key='gpu-A-eager')
    assert len(m['leave_one_geometry_out'])==8
    for x in (0,4,7,100):
        r=advise(m,environment_key='gpu-A-eager',features=[x]*len(FEATURES))
        assert r['action']=='calibrate' and r['resource_authorised'] is False
