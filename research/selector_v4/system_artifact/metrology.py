"""Preserve process and balanced paired-block units; never trim measurements."""
import math

import numpy as np


def paired_cube(values):
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=3 or min(a.shape)<1 or not np.isfinite(a).all() or np.any(a<=0):
        raise ValueError('Expected positive finite [process, block, paired window] ratios')
    if a.shape[0]<2 or a.shape[1]<2:
        raise ValueError('At least two real processes and two complete blocks required')
    return a


def hierarchical_log_draws(values, *, seed, draws=20000):
    logs=np.log(paired_cube(values))
    if draws<2000:
        raise ValueError('Insufficient bootstrap budget')
    p,b,_=logs.shape
    # The two mirrored windows are a fixed balanced design, not IID replicates.
    # Resample whole paired blocks; preserve their order and both arm pairs.
    block_logs=logs.mean(axis=2)
    rng=np.random.default_rng(seed)
    processes=rng.integers(p,size=(draws,p))
    blocks=rng.integers(b,size=(draws,p,b))
    return block_logs[processes[:,:,None],blocks].mean(axis=(1,2))


def describe(values, *, seed, draws=20000):
    logs=np.log(paired_cube(values));blocks=logs.mean(axis=2);processes=blocks.mean(axis=1)
    samples=hierarchical_log_draws(values,seed=seed,draws=draws)
    p,b,w=logs.shape
    return {'geomean':math.exp(float(logs.mean())),
            'hierarchical_ci95':np.exp(np.quantile(samples,[.025,.975])).tolist(),
            'real_independent_processes':p,'blocks_per_process':b,'paired_windows_per_block':w,
            'bootstrap_draws':draws,'seed':seed,'window_resampling':False,
            'resampling_unit':'Processes, then whole balanced paired blocks; mirrored windows preserved',
            'observed_process_mean_log_variance':float(processes.var(ddof=1)),
            'within_process_block_mean_log_variance':float(blocks.var(axis=1,ddof=1).mean()),
            'within_block_paired_window_log_variance':float(logs.var(axis=2,ddof=1).mean()) if w>1 else None,
            'variance_scope':'Descriptive variances of observed nested means/residuals; not unbiased random-effect components',
            'minimum_observed_paired_ratio':float(np.exp(logs.min())),
            'trimmed_samples':0,'qualification_authority':False}


def regret_summary(values, *, definition):
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Regret population must be nonempty and finite')
    return {'definition':definition,'units':'fraction; percent = fraction * 100',
            'count':len(a),'median':float(np.median(a)),'mean':float(a.mean()),
            'p90':float(np.quantile(a,.90)),'p95':float(np.quantile(a,.95)),
            'p99':float(np.quantile(a,.99)),'worst':float(a.max()),
            'negative_observation_count':int((a<0).sum()),
            'above_0_5_percent_count':int((a>.005).sum()),
            'above_1_percent_count':int((a>.01).sum()),
            'above_5_percent_count':int((a>.05).sum()),
            'population_scope':'Descriptive fixed workload/fold population, not IID independent trials'}
