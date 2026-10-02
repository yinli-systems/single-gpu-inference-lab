"""Rebuild scope-bound research figures/tables from immutable retained evidence."""
import argparse
import csv
import hashlib
import html
import importlib.util
import json
import math
import tarfile
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from research.selector_v4.exposed_diagnostics.analyze_public_path_prequal import check_rows
from research.selector_v4.system_artifact.metrology import describe, regret_summary
from research.selector_v4.system_artifact.prior import fit_training_prior
from research.selector_v4.system_artifact.risk import tail_diagnostic

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def geo(a):return math.exp(float(np.log(a).mean()))


def source_inputs():
    spec=json.loads((ROOT/'artifact/input-manifest.json').read_text());docs={}
    for name,entry in spec.items():
        path=ROOT/entry['path']
        if sha(path)!=entry['sha256']:raise ValueError('Frozen input changed: '+name)
        if path.suffix=='.json':docs[name]=json.loads(path.read_text())
    return spec,docs


def read_raw(spec,summary,cache):
    path=ROOT/spec['public_raw']['path'];expected=summary['files']
    names={name for name in expected if name.startswith('runs/gpu_5090-') and
           name.endswith(('environment.json','windows.json','complete.json','frozen-choice-before-scoring.json'))}
    # Cache is optional; each cached original byte string is checked against the
    # immutable original complete-source summary. No reserialised-float hashes.
    if cache.exists():data=json.loads(cache.read_text())
    else:
        data={}
        with tarfile.open(path,'r|gz') as t:
            for m in t:
                name=m.name.removeprefix('campaign/')
                if name in names:data[name]=t.extractfile(m).read().decode()
        cache.parent.mkdir(parents=True,exist_ok=True);save(cache,data)
    if set(data)!=names:raise ValueError('Missing or extra original raw measurement members')
    for name,value in data.items():
        if hashlib.sha256(value.encode()).hexdigest()!=expected[name]:raise ValueError('Raw member changed: '+name)
    return {n:json.loads(v) for n,v in data.items()}


def analyze(raw,summary):
    fold_rows=[];metrology=[];prior_rows=[];tails=[]
    for case in (0,1):
        job=('1647200','1647201')[case];prefix=f'runs/gpu_5090-{job}'
        keys=raw[prefix+'/policy-0/complete.json']['cells']
        for key in sorted(keys):
            gains=[];native_pristine=[];training=[];process_trials=[]
            for rep in range(3):
                base=prefix+f'/policy-{rep}/'+key
                c=raw[base+'/complete.json'];p=check_rows(raw[base+'/windows.json'],'policy')
                pristine=check_rows(raw[prefix+f'/pristine-{rep}/'+key+'/windows.json'],'pristine')
                t=check_rows(raw[prefix+f'/train-{rep}/'+key+'/windows.json'],'train')
                tc=raw[prefix+f'/train-{rep}/'+key+'/complete.json']
                gains.append([[b['native'][i]/b['policy'][i] for i in range(2)] for b in p])
                native_pristine.append([[geo(a['native'])/geo(b['native'])] for a,b in zip(pristine,p)])
                ratios=[[b['native'][i]/b['candidate'][i] for i in range(2)] for b in t]
                training.append(ratios)
                env=raw[prefix+f'/train-{rep}/environment.json']
                actual_cap=tc['certificate_accepted'] and tc['managed_tactic']==1 and tc['choice']=='resource'
                if actual_cap:
                    process_trials.append({'process_id':str(env['pid']),
                        'slowdown_event':bool(np.any(np.asarray(ratios).prod(axis=1)**.5<1/1.01))})
                    # No fitted probabilities are manufactured: current data
                    # has only two geometry groups, regardless of fold count.
                    prior_rows.append({'role':'train','environment_key':'75544a17-gpu_5090-'+c['execution'],
                                       'geometry_id':env['case']['id'],'features':None,
                                       'wins_by_one_percent':geo(np.asarray(ratios).ravel())>1.01})
                arms={arm:geo([geo(b[arm]) for b in p]) for arm in ('native','oracle','policy')}
                available=arms['policy']/min(arms.values())-1
                signed=arms['policy']/min(arms['native'],arms['oracle'])-1
                known_cap=c['certificate_accepted'] and c['managed_tactic']==1
                reference=next(r for r in summary['records'] if r['case']==case and r['key']==key and r['rep']==rep)
                assert math.isclose(geo(np.asarray(gains[-1]).ravel()),reference['policy_over_native_speedup'],rel_tol=1e-12)
                fold_rows.append({'case':case,'key':key,'rep':rep,'execution':c['execution'],
                    'dtype':c['dtype'],'layout':c['layout'],'selected':c['choice']=='resource',
                    'available_arm_excess':available,'signed_independent_measurement_regret':signed,
                    'native_cap_counterfactual_measured':known_cap,
                    'oracle_candidate':'resource' if known_cap else 'native_only',
                    'chosen_label_lookup_regret':arms['native' if c['choice']=='native' else 'oracle']/min(arms['native'],arms['oracle'])-1})
            seed=int(hashlib.sha256(f'{case}:{key}:system-artifact'.encode()).hexdigest()[:8],16)
            metrology.append({'case':case,'key':key,'policy_over_native':describe(gains,seed=seed),
                              'native_over_pristine':describe(native_pristine,seed=seed^41),
                              'training_candidate':describe(training,seed=seed^73)})
            if len(process_trials)==3:tails.append({'case':case,'key':key,**tail_diagnostic(process_trials)})
    assert len(fold_rows)==144 and len(metrology)==48
    available=regret_summary([r['available_arm_excess'] for r in fold_rows],definition='Independent policy latency / minimum of measured Native, managed-oracle and policy arm means - 1; includes policy duplicate to reproduce the existing nonnegative excess definition')
    assert math.isclose(available['p99'],summary['metrics']['regret']['p99'],abs_tol=2e-14)
    native_cap=[r['chosen_label_lookup_regret'] for r in fold_rows if r['native_cap_counterfactual_measured']]
    prior={}
    for mode in sorted({r['environment_key'] for r in prior_rows}):
        rows=[r for r in prior_rows if r['environment_key']==mode]
        prior[mode]=fit_training_prior(rows,environment_key=mode)
    return {'scope':'Retrospective complete 5090 column/arithmetic supplement on two exposed geometries; original dual qualification HOLD',
        'available_arm_excess':available,
        'signed_independent_measurement_regret':regret_summary([r['signed_independent_measurement_regret'] for r in fold_rows],definition='Independent policy latency / minimum of Native and managed-oracle arm means - 1; negative differences retained as measurement noise'),
        'chosen_label_oracle_lookup_regret':regret_summary([r['chosen_label_lookup_regret'] for r in fold_rows],definition='Choose the Native/oracle label frozen before scoring; score using the same independent oracle table, excludes policy dispatch cost'),
        'measured_native_cap_lookup_regret':regret_summary(native_cap,definition='Same independent label lookup but restricted to certified measured Native+Resource candidate pools') if native_cap else None,
        'native_cap_measured_folds':len(native_cap),'native_cap_unmeasured_folds':144-len(native_cap),
        'fold_rows':fold_rows,'metrology':metrology,'strict_tail_audit':tails,'analytical_prior':prior,
        'raw_scoring_rows_retained':144*144,'trimmed_windows':0,'qualification_authority':False,
        'default_promotion':False,'serving_promotion':False,'historical_token_divergence_resolved':False}


def figures(out,docs,analysis):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'sgi-v42-artifact-v1',
                         'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    outputs=[]
    def finish(fig,name,title,caption):
        fig.suptitle(title,fontsize=14,fontweight='bold',x=.06,ha='left',y=.96)
        fig.text(.06,.025,textwrap.fill(caption,width=140),fontsize=8,ha='left',va='bottom',color='#444444')
        fig.subplots_adjust(left=.19 if name=='figure1_resource_intervention' else .13,right=.96,bottom=.24,top=.80,wspace=.5)
        for extension in ('svg','png'):
            path=out/(name+'.'+extension)
            fig.savefig(path,dpi=170,metadata={'Date':None} if extension=='svg' else {'Software':'SGI frozen research artifact'})
            outputs.append(path.name)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.8))
    labels=['Native','48 / 48 KiB','64 / 48 KiB','64 / 64 KiB']
    keys=['native','attribute49152_launch49152','attribute65536_launch49152','attribute65536_launch65536']
    for ax,gpu in zip(axes,('gpu_4090','gpu_5090')):
        data=docs['intervention']['results'][gpu]['layouts']['paged1_cached_prefix']['event_median_us']['warm']
        ax.barh(labels,[data[k] for k in keys],color=['#6b7280','#9ca3af','#9ca3af','#167a7a'])
        ax.invert_yaxis();ax.set_xlabel('Median CUDA-event latency (microseconds)');ax.set_title(gpu.replace('gpu_','RTX '))
    finish(fig,'figure1_resource_intervention','Resource attributes and actual launch capacity are different interventions',
           'Labels show attribute / launch shared-memory capacity (KiB). Exposed page1 prefix diagnostic, fixed math/data; event medians are mechanism observations, not release or HTTP timing.')
    fig,ax=plt.subplots(figsize=(9,4.8));x=np.arange(2);width=.30
    for i,(name,key) in enumerate((('Selected','paired_selected'),('Fallback','paired_unselected'))):
        values=[docs['static'+g]['release_gate'][key]['worst_point_ratio'] for g in ('4090','5090')]
        ax.bar(x+(i-.5)*width,values,width,label=name,color=('#167a7a','#d68732')[i])
    ax.axhline(.99,color='#a32929',linestyle='--',label='Unchanged 0.99 floor');ax.set_ylim(.97,1.16)
    ax.set_xticks(x,['RTX4090','RTX5090']);ax.set_ylabel('Worst measured speedup ratio');ax.legend(loc='upper right')
    finish(fig,'figure2_static_selector_failure','Fresh release exposes selected and fallback failures',
           'Original v3.2.2 fresh release, 30 geometries. Both original verdicts HOLD; selected gains cannot override fallback or control failures.')
    fig,ax=plt.subplots(figsize=(10,5));ax.set_axis_off();ax.set_xlim(0,10);ax.set_ylim(0,5)
    boxes=[(.2,3.2,2.5,1,'Stable Native\nPlan / Run / kernel'),(3.5,3.2,2.5,1,'Bound eligibility\nmode + GPU + geometry'),
           (7,3.2,2.5,1,'Empirical safety gate\nNative or Resource'),(3.5,1,2.5,1,'Missing / stale / uncertain\nNative fallback'),
           (7,1,2.5,1,'Private metadata Graph\nDiagnostic only; no serving cert')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.1',facecolor='#eef4f4',edgecolor='#167a7a'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center')
    for a,b in (((2.8,3.7),(3.3,3.7)),((6.1,3.7),(6.8,3.7)),((4.75,3.1),(4.75,2.1)),((8.25,3.1),(8.25,2.1))):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color='#444444'))
    finish(fig,'figure3_native_transparent_architecture','A research tactic system with explicit authority boundaries',
           'Stable path remains separate. Retrospective risk/prior analyses do not certify tactics. Actual private Graph metadata tests are a separate scope.')
    fig,ax=plt.subplots(figsize=(9,4.8));values=np.sort([r['available_arm_excess']*100 for r in analysis['fold_rows']])
    ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',color='#167a7a',linewidth=2)
    ax.set_xlabel('Available-arm excess (%)');ax.set_ylabel('Empirical CDF over 144 fixed folds');ax.set_ylim(0,1.03)
    ax.axvline(analysis['available_arm_excess']['p99']*100,color='#d68732',linestyle='--',label='P99');ax.legend()
    finish(fig,'figure4_available_arm_excess','Small observed excess is not a global forced-cap oracle guarantee',
           f'Complete 5090 supplemental scope; {analysis["native_cap_unmeasured_folds"]}/144 folds lack a measured Native+Cap pool. Policy duplicates remain in this metric.')
    fig,ax=plt.subplots(figsize=(9,4.8));values=[docs['v411'+g]['metrics']['selected_geomean'] for g in ('4090','5090')]
    ax.bar([0,1],values,color='#167a7a',width=.6);ax.axhline(1,color='#555555',linestyle='--')
    ax.set_xticks([0,1,2,3],['RTX4090','RTX5090','H100\nnot measured','B200\nnot measured']);ax.set_ylim(0,1.6)
    ax.set_ylabel('Selected geomean speedup ratio');ax.set_xlim(-.6,3.6)
    for i in (2,3):ax.text(i,.75,'No evidence',ha='center',color='#777777')
    finish(fig,'figure5_cross_gpu_scope','Two consumer GPU families; datacenter extrapolation remains untested',
           'Source-bound v4.1.1 exposed development, selected subsets only. Original qualification HOLD; no fresh-generalization or H100/B200 claim.')
    fig,ax=plt.subplots(figsize=(9,4.8));ax.set_axis_off()
    lines=['Current complete four-model paired HTTP qualification: NOT COMPLETED',
           'Throughput / TTFT / TPOT / strict SLO Pareto curves: NOT AVAILABLE',
           'Historical original token divergence: 2/432, UNRESOLVED',
           'default_promotion = false        serving_promotion = false',
           'Full HTTP follows archived dual development and formal kernel PASS']
    for i,line in enumerate(lines):ax.text(.01,.92-i*.18,textwrap.fill(line,width=82),transform=ax.transAxes,fontsize=10,color='#a32929' if i<3 else '#333333')
    finish(fig,'figure6_serving_evidence_gap','Serving results are an explicit evidence gap',
           'This status figure contains no invented SLO curve or throughput value. GPU/kernel observations do not establish end-to-end request gains.')
    return outputs


def run(out):
    if out.exists():raise FileExistsError('Preserve earlier reproductions; choose a new output directory')
    spec,docs=source_inputs();out.mkdir(parents=True)
    verifier_path=ROOT/spec['graph_verifier']['path']
    module_spec=importlib.util.spec_from_file_location('frozen_graph_audit',verifier_path)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    verified_graph=module.verify(verifier_path.parent)
    if verified_graph!=docs['graph']:raise ValueError('Graph facts differ from the full raw-byte/launch audit')
    raw=read_raw(spec,docs['public5090'],ROOT/'artifact/.cache/raw5090.json')
    analysis=analyze(raw,docs['public5090']);save(out/'analysis.json',analysis)
    assert docs['authority']['default_promotion'] is False and docs['authority']['serving_promotion'] is False
    figures_out=figures(out,docs,analysis)
    tables={
        'table1_correctness':{'graph_metadata_diagnostic':docs['graph'],'historical_parity':docs['history']['results'] if 'results' in docs['history'] else docs['history']},
        'table2_kernel':{'v411_4090':docs['v4114090']['metrics'],'v411_5090':docs['v4115090']['metrics'],'public5090_supplement':docs['public5090']['metrics'],'regret_extensions':{k:v for k,v in analysis.items() if 'regret' in k or k=='available_arm_excess'}},
        'table3_slo':{'status':'NOT_QUALIFIED_NOT_IMPUTED','request_per_s':None,'input_tok_per_s':None,'output_tok_per_s':None,'ttft_percentiles':None,'tpot_percentiles':None,'itl':None,'strict_slo_goodput':None}}
    save(out/'tables.json',tables)
    readable_tables={
        'table1_correctness':[['scope','result','evidence'],['Private dual Graph metadata','Diagnostic PASS',f"{verified_graph['total_gpu_cells']} cells; {verified_graph['total_metadata_epochs']} epochs; {verified_graph['actual_traced_resource_attention_launches']} traced attention launches"],['Original v4.2 canary','HOLD','Original 2/432 token divergence unresolved'],['Full current HTTP','Not qualified','No promoted serving result']],
        'table2_kernel':[['scope','metric','value','unit'],['RTX4090 v4.1.1 selected exposed','geomean',docs['v4114090']['metrics']['selected_geomean'],'ratio'],['RTX5090 v4.1.1 selected exposed','geomean',docs['v4115090']['metrics']['selected_geomean'],'ratio'],['RTX5090 supplemental 144 fixed folds','available-arm excess P99',analysis['available_arm_excess']['p99']*100,'percent'],['RTX5090 supplemental 144 fixed folds','unmeasured Native+Cap pools',analysis['native_cap_unmeasured_folds'],'folds']],
        'table3_slo':[['metric','value','status']]+[[x,'unavailable','NOT_QUALIFIED_NOT_IMPUTED'] for x in tables['table3_slo'] if x!='status']}
    for name,rows in readable_tables.items():
        with (out/(name+'.csv')).open('w',newline='') as f:csv.writer(f).writerows(rows)
    source_files=sorted((ROOT/'research/selector_v4/system_artifact').glob('*.py'))+[ROOT/'Makefile',ROOT/'artifact/requirements.lock',ROOT/'artifact/input-manifest.json',ROOT/'research/selector_v4/exposed_diagnostics/analyze_public_path_prequal.py',ROOT/'research/selector_v4/exposed_diagnostics/public_path_contract.py',ROOT/'research/selector_v4/public_qualification/choice_integrity.py',ROOT/'research/selector_v4/public_qualification/contract.py',ROOT/'research/selector_v4/public_qualification/telemetry.py',verifier_path]
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    outputs={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    manifest={'schema':1,'result':'REPRODUCED_SCOPED_RESEARCH_ARTIFACT_NOT_QUALIFICATION','inputs':spec,
        'outputs':outputs,'source_files':source_hashes,'figures':figures_out,'numpy_version':np.__version__,'matplotlib_version':matplotlib.__version__,
        'raw_archive_all_original_failures_retained':True,'fresh_cases_consumed':0,'qualification_authority':False,
        'default_promotion':False,'serving_promotion':False,'historical_token_divergence_resolved':False,
        'original_canary':'HOLD','public_repeat':'RUNNING_AT_LAST_SNAPSHOT','hardware_uncovered':['H100','B200'],
        'full_http_qualified':False,'notes':['Figure6 is an evidence-gap status, not an SLO measurement','Mirrored windows are fixed paired-block members, not independent bootstrap trials','No new scoring samples, changed frozen decisions or thresholds']}
    save(out/'manifest.json',manifest)
    page=['<!doctype html><meta charset="utf-8"><title>Resource-sensitive attention research artifact</title>',
          '<style>body{font:16px system-ui;max-width:1050px;margin:40px auto;line-height:1.5}img{width:100%}code{background:#eee}a{color:#167a7a}</style>',
          '<h1>Resource-sensitive attention: reproducible evidence</h1>',
          '<p>Research artifact only. Original canary HOLD; default and serving promotion false; original 2/432 unresolved.</p>',
          '<p>Complete 5090 retrospective analysis, separately scoped dual Graph functional evidence, and retained negative results. Current full HTTP and H100/B200 results are unavailable.</p>']
    for name in figures_out:
        if name.endswith('.svg'):page.append(f'<figure><img src="{name}"><figcaption>{name.replace("_"," ")}</figcaption></figure>')
    for name,rows in readable_tables.items():
        page.append('<h2>'+html.escape(name.replace('_',' '))+'</h2><table border="1" cellpadding="8" cellspacing="0">')
        for row in rows:page.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>')
        page.append('</table>')
    page.extend(['<p><a href="analysis.json">Metrology, risk, prior and regret analysis</a> · <a href="tables.json">Tables</a> · <a href="manifest.json">Source/output hashes and scopes</a></p>'])
    (out/'index.html').write_text('\n'.join(page)+'\n')
    manifest['outputs']['index.html']=sha(out/'index.html');save(out/'manifest.json',manifest)
    print(json.dumps({'out':str(out),'figures':len(figures_out)//2,'fold_records':len(analysis['fold_rows']),
                      'native_cap_unmeasured_folds':analysis['native_cap_unmeasured_folds'],'qualification_authority':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
