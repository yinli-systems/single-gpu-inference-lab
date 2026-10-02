"""Exact binomial tail diagnostic; bootstrap draws never become real trials."""
import math


def _cdf(k,n,p):
    if p<=0:return 1.0
    if p>=1:return float(k==n)
    terms=[math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)+
           i*math.log(p)+(n-i)*math.log1p(-p) for i in range(k+1)]
    m=max(terms)
    return math.exp(m)*math.fsum(math.exp(x-m) for x in terms)


def upper_binomial(k,n,*,confidence=.95):
    if not isinstance(n,int) or not isinstance(k,int) or not 0<=k<=n or n<1 or not 0<confidence<1:
        raise ValueError('Invalid independent trial counts or confidence')
    if k==n:return 1.0
    alpha=1-confidence
    if k==0:return -math.expm1(math.log(alpha)/n)
    lo,hi=k/n,1.0
    for _ in range(80):
        mid=(lo+hi)/2
        if _cdf(k,n,mid)>alpha:lo=mid
        else:hi=mid
    return hi


def zero_failure_trials(*,epsilon=.001,confidence=.95):
    if not 0<epsilon<1 or not 0<confidence<1:raise ValueError('Invalid tail target')
    # Strict upper_bound < epsilon, as requested; equality is insufficient.
    return math.floor(math.log1p(-confidence)/math.log1p(-epsilon))+1


def tail_diagnostic(process_trials,*,epsilon=.001,confidence=.95):
    if not process_trials or len({x['process_id'] for x in process_trials})!=len(process_trials):
        raise ValueError('Require actual distinct independent process IDs')
    if any(type(x['slowdown_event']) is not bool for x in process_trials):
        raise ValueError('Every complete process must retain its boolean failure event')
    n=len(process_trials);k=sum(x['slowdown_event'] for x in process_trials)
    upper=upper_binomial(k,n,confidence=confidence)
    return {'independent_processes':n,'slowdown_processes':k,'one_sided_confidence':confidence,
            'binomial_upper_bound':upper,'epsilon':epsilon,'tail_target_supported':upper<epsilon,
            'zero_failure_trials_needed':zero_failure_trials(epsilon=epsilon,confidence=confidence),
            'event':'At least one paired block slower than Native by >1% in the complete process',
            'assumptions':'Independent exchangeable future processes for this fixed source/environment/geometry and protocol; not a production-wide guarantee',
            'bootstrap_draws_are_trials':False,'qualification_authority':False}


def constrained_research_choice(*,base_checks,tail,static_eligible):
    # This prototype cannot manufacture a serving certificate or widen tactics.
    checks={'static_eligible':static_eligible is True,'base_safety_checks':bool(base_checks) and
            all(type(v) is bool and v for v in base_checks.values()),
            'declared_tail_target':tail.get('tail_target_supported') is True}
    return {'choice':'resource' if all(checks.values()) else 'native','checks':checks,
            'scope':'Offline risk-constraint ablation only; not an executable/certified runner',
            'qualification_authority':False,'default_promotion':False,'serving_promotion':False}
