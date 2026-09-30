"""Combine the 28 strictly-fresh primary cases with the preregistered two-case extension.

The original 30-case primary remains authoritative. This analysis is a separately
labelled, non-promotional sensitivity and reconstructs independent campaign
bootstrap draws from stored process/block log bases.
"""
from __future__ import annotations
import argparse,functools,hashlib,json,math,re
from pathlib import Path
from typing import Any
import numpy as np
from release_gate import evaluate_release_gate

ARM_OFFSET={'pristine':0,'off':7,'cap':13,'guarded':29}
CAMPAIGN_SEED={'primary':936612,'extension':1936612}

def require(condition: bool,message: str)->None:
    if not condition: raise ValueError(message)

def sha(path: Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path: Path)->dict[str,Any]:
    require(path.is_file(),f'missing input {path}')
    return json.loads(path.read_text())

@functools.lru_cache(None)
def campaign_weights(campaign: str,shard: int,arm: str)->np.ndarray:
    require(campaign in CAMPAIGN_SEED,'unknown campaign namespace')
    require(arm in ARM_OFFSET,'unknown bootstrap arm')
    seed=CAMPAIGN_SEED[campaign]+int(shard)*101+ARM_OFFSET[arm]
    rng=np.random.default_rng(seed);weights=np.zeros((10000,24),dtype=float)
    for draw in range(10000):
        for process in rng.integers(0,3,size=3):
            for block in rng.integers(0,8,size=8):weights[draw,process*8+block]+=1
    return weights/24

def normalized_environment(summary: dict[str,Any])->dict[str,Any]:
    env=summary['deployment_environment'];common=dict(env['common'])
    for key in ('case_hash','source_archive_sha256'):common.pop(key,None)
    builds={mode:{'mode':value['mode'],'header_sha256':value['header_sha256']}
            for mode,value in env['mode_builds'].items()}
    return {'common':common,'mode_builds':builds}

def measurement_contract(summary: dict[str,Any])->dict[str,str]:
    source=summary.get('source_hashes')
    require(isinstance(source,dict),'measurement source hashes missing')
    required=('measure.py','prepare_guarded.py')
    require(all(isinstance(source.get(name),str) and len(source[name])==64 for name in required),'measurement contract hashes missing')
    return {name:source[name] for name in required}

def combined_numerics(primary: dict[str,Any],extension: dict[str,Any])->dict[str,dict[str,Any]]:
    out={}
    for mode in ('pristine','off','cap','guarded'):
        a=primary['numerics'][mode];b=extension['numerics'][mode]
        out[mode]={
            'qualifications':int(a['qualifications'])+int(b['qualifications']),
            'exact_full_outputs':int(a['exact_full_outputs'])+int(b['exact_full_outputs']),
            'max_abs_vs_pristine':max(float(a['max_abs_vs_pristine']),float(b['max_abs_vs_pristine'])),
            'FP32_max_abs':max(float(a['FP32_max_abs']),float(b['FP32_max_abs'])),
            'FP32_vectors':int(a['FP32_vectors'])+int(b['FP32_vectors']),
        }
    return out

def rebuild_draws(cells: list[dict[str,Any]],campaign: str,offset: int,
                  drawcache: dict[tuple[int,str],np.ndarray])->None:
    for local_index,cell in enumerate(cells):
        basis=cell.get('bootstrap_basis');require(isinstance(basis,dict),'bootstrap basis missing')
        require(set(basis)=={'pristine','off','cap','guarded'},'incomplete bootstrap basis')
        pristine=np.asarray(basis['pristine']['main_log'],dtype=float)
        require(pristine.shape==(24,) and np.isfinite(pristine).all(),'bad pristine bootstrap basis')
        for mode in ('off','cap','guarded'):
            candidate=np.asarray(basis[mode]['main_log'],dtype=float)
            repeats=np.asarray(basis[mode]['repeat_log'],dtype=float)
            require(candidate.shape==(24,) and repeats.shape==(24,) and np.isfinite(candidate).all() and np.isfinite(repeats).all(),'bad candidate bootstrap basis')
            point=float(math.exp(float(pristine.mean()-candidate.mean())))
            require(math.isclose(point,float(cell['comparisons'][mode]['ratio']),rel_tol=1e-11,abs_tol=1e-12),'stored point ratio/basis mismatch')
            drawcache[(offset+local_index,mode)]=(campaign_weights(campaign,int(cell['shard']),'pristine')@pristine-
                                                   campaign_weights(campaign,int(cell['shard']),mode)@candidate)

def combine(primary_path: Path,extension_path: Path,amendment_path: Path,
            preregistration_path: Path,analysis_commit: str)->dict[str,Any]:
    require(re.fullmatch(r'[0-9a-f]{40}',analysis_commit or '') is not None,'invalid analysis commit')
    primary=load_json(primary_path);extension=load_json(extension_path)
    amendment=load_json(amendment_path);prereg=load_json(preregistration_path)
    require(primary.get('stage')=='release' and extension.get('stage')=='release','release summaries required')
    require(primary.get('gpu')==extension.get('gpu'),'GPU family cannot be mixed')
    require(primary.get('manifest_hash')==prereg.get('primary_manifest_case_hash'),'primary manifest mismatch')
    require(extension.get('manifest_hash')==prereg.get('extension_case_hash'),'extension manifest mismatch')
    require(primary.get('measurement_source_commit')==prereg.get('primary_measurement_source_commit'),'primary measurement commit mismatch')
    require(prereg.get('selector_rule_changed') is False and prereg.get('primary_manifest_changed') is False and prereg.get('merge_into_primary_claim') is False,'invalid extension preregistration')
    require(amendment.get('original_30_case_primary_gate_unchanged') is True,'primary gate mutation')
    excluded=set(amendment.get('secondary_strict_freshness_sensitivity_excludes',[]))
    require(excluded==set(prereg.get('excluded_primary_duplicates',excluded)) or excluded=={'holdout-v3-opp-00','holdout-v3-opp-01'},'freshness exclusion mismatch')
    require(normalized_environment(primary)==normalized_environment(extension),'deployment environment/overlay differs across campaigns')
    require(measurement_contract(primary)==measurement_contract(extension),'measurement program differs across campaigns')
    primary_cells=[cell for cell in primary['cells'] if cell['case'] not in excluded]
    extension_cells=list(extension['cells'])
    primary_cases={cell['case'] for cell in primary_cells};extension_cases={cell['case'] for cell in extension_cells}
    prereg_cases={case['id'] for case in prereg['cases']}
    require(len(primary_cases)==28 and len(extension_cases)==2 and extension_cases==prereg_cases,'strict-freshness case coverage mismatch')
    require(primary_cases.isdisjoint(extension_cases),'case IDs overlap across campaigns')
    require(len(primary_cells)==28*48 and len(extension_cells)==2*48,'cell matrix incomplete')
    cells=primary_cells+extension_cells;drawcache={}
    rebuild_draws(primary_cells,'primary',0,drawcache)
    rebuild_draws(extension_cells,'extension',len(primary_cells),drawcache)
    numerics=combined_numerics(primary,extension)
    gate=evaluate_release_gate(cells,drawcache,numerics)
    primary_pass=bool(primary['release_gate']['pass'])
    extension_pass=bool(extension['extension_candidate_gate']['pass'])
    return {
        'schema':1,'gpu':primary['gpu'],'complete':True,
        'analysis_source_commit':analysis_commit,
        'input_sha256':{
            'primary_summary':sha(primary_path),'extension_summary':sha(extension_path),
            'freshness_amendment':sha(amendment_path),'extension_preregistration':sha(preregistration_path),
        },
        'primary_measurement_source_commit':primary['measurement_source_commit'],
        'extension_measurement_source_commit':extension['measurement_source_commit'],
        'primary_analysis_source_commit':primary['analysis_source_commit'],
        'extension_analysis_source_commit':extension['analysis_source_commit'],
        'environment_identity':normalized_environment(primary),
        'measurement_contract':measurement_contract(primary),
        'excluded_primary_boundary_cases':sorted(excluded),
        'strictly_fresh_primary_cases':sorted(primary_cases),
        'strictly_fresh_extension_cases':sorted(extension_cases),
        'cells':cells,'numerics':numerics,'combined_strict_freshness_gate':gate,
        'primary_gate_pass':primary_pass,'extension_candidate_gate_pass':extension_pass,
        'combined_candidate_gate_pass':bool(gate['pass']),
        'performance_evidence_support':bool(primary_pass and extension_pass and gate['pass']),
        'single_gpu_release_gate_pass':False,
        'serving_validation_eligible':False,
        'default_promotion':False,'serving_promotion':False,
        'scope':'Non-promotional sensitivity. The original 30-case primary remains authoritative; historical token divergence remains unresolved.',
    }

def main(args: argparse.Namespace)->None:
    require(not args.out.exists(),'preserve combined analysis output')
    result=combine(args.primary_summary,args.extension_summary,args.freshness_amendment,args.extension_preregistration,args.analysis_commit)
    args.out.mkdir(parents=True)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    gate=result['combined_strict_freshness_gate']
    lines=['# Selector-v3 combined strict-freshness sensitivity','',
           f"GPU: `{result['gpu']}`",'',
           f"**Combined candidate gate: {'PASS' if gate['pass'] else 'HOLD'}**",'',
           '- This result is non-promotional and cannot override the original 30-case primary.',
           '- Selected Graph16: count %s; geomean %.6f; 95%% CI [%.6f, %.6f]; point worst %.6f; joint-min LCB %.6f.'%(
               gate['selected']['count'],gate['selected']['ratio'],*gate['selected']['CI95'],gate['selected']['worst_point_ratio'],gate['selected']['simultaneous_worst_CI95'][0]),
           '- Requirements: `'+json.dumps(gate['requirements'],sort_keys=True)+'`','']
    (args.out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cells','environment_identity')},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--primary-summary',type=Path,required=True)
    parser.add_argument('--extension-summary',type=Path,required=True)
    parser.add_argument('--freshness-amendment',type=Path,required=True)
    parser.add_argument('--extension-preregistration',type=Path,required=True)
    parser.add_argument('--analysis-commit',required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
