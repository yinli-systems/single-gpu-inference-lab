"""Reject stale, corrupt or non-paired evidence before computing a statistic."""
from __future__ import annotations
import hashlib,json,math,csv,io
from pathlib import Path
from manifest import load,tactic_identity,plan_guard,digest
from measurement_contract import REVISION,check_eager_window,expected_sequence

EXECUTIONS=("eager_full_call","graph1_replay","graph16_replay")

def require(condition, message):
    if not condition: raise ValueError(message)

def checked_files(path, complete):
    required={"environment.json","measurements.json","qualification.json","progress.json","memory.json"}
    files=complete.get("files",{})
    require(required <= set(files), "missing artifact hash")
    for name, expected in files.items():
        require(Path(name).name == name and name not in (".",".."), "unsafe artifact path")
        require(hashlib.sha256((path/name).read_bytes()).hexdigest()==expected, "artifact digest mismatch: "+name)

def validate_run(path, mode, rep, stage, cases, blocks):
    path=Path(path)
    complete=json.loads((path/"complete.json").read_text())
    require(complete.get("complete") is True, "incomplete run")
    checked_files(path,complete)
    env=json.loads((path/"environment.json").read_text())
    rows=json.loads((path/"measurements.json").read_text())
    quals=json.loads((path/"qualification.json").read_text())
    require(env.get("measurement_contract_revision")==REVISION,"old measurement revision cannot promote new code")
    require(env.get("profiled") is False and env.get("nsight_compute_excluded") is True,"profiled data cannot enter release")
    require((env.get("mode"),env.get("rep"),env.get("stage"))==(mode,rep,stage),"environment/run identity")
    require(env.get("cases")==cases and env.get("case_hash")==load()["case_hash"],"frozen manifest mismatch")
    source=path.parent.parent/"source"
    for name in ("measure.py","manifest.py","measurement_contract.py","native_policy.py","prepare_guarded.py"):
        require(hashlib.sha256((source/name).read_bytes()).hexdigest()==env.get("source",{}).get(name),"measured source mismatch: "+name)
    hardware=list(csv.DictReader(io.StringIO(env["hardware"]["out"]),skipinitialspace=True))
    require(len(hardware)==1,"ambiguous physical GPU")
    hw=hardware[0]; case_map={c['id']:c for c in cases}
    basic={(c['id'],dt,layout,split) for c in cases for dt in ('float16','bfloat16') for layout in ('ragged','paged') for split in ('auto','unsplit')}
    qmap={}
    expected_arms={'pristine'} if mode=='pristine' else {'off','cap','guarded'}
    for q in quals:
        key=(q['case'],q['dtype'],q['layout'],q['split'])
        require(key in basic and key not in qmap,"qualification coordinate")
        require(q.get('measurement_contract_revision')==REVISION and q.get('eager_graph_exact') is True,"qualification revision")
        require(set(q['arms'])==expected_arms,"missing or extra arm")
        calls=q.get('eager_calls');require(type(calls) is int and calls>=16 and calls<=4096,"calibrated count")
        c=case_map[key[0]];require(q['expected_selector']==c['expected_selector'],"selector expectation changed")
        require(q.get('ordered_pairs_sha256')==digest(list(zip(c['q'],c['cached']))),'ordered geometry checksum')
        if mode=='paired':
            plans={arm:entry['plan_info'] for arm,entry in q['arms'].items()}
            require(all(len(v)==16 for v in plans.values()) and len({tuple(v[:-1]) for v in plans.values()})==1,'actual candidate plan core mismatch')
            require(plans['off'][-1]==0 and plans['cap'][-1]==1,'native policy flags')
            g=plans['guarded'];decision=plan_guard(c['q'],c['cached'],g[0],32,8,env['num_sm'],bool(g[14]))
            require(q['guard_expected']==decision and bool(g[15])==decision,'frozen selector rule mismatch')
        for arm,qual in q['arms'].items():
            require(qual.get('pristine_exact') is True,"nonexact arm")
            require(qual.get('execution_checks')==dict.fromkeys(EXECUTIONS,True),"separate replay correctness missing")
            for h in ('out_sha256','lse_sha256'):
                require(len(qual[h])==64 and all(ch in '0123456789abcdef' for ch in qual[h]),"invalid tensor hash")
            for ex in EXECUTIONS:
                payload=q['identity_payloads'][arm][ex];ei=payload['environment'];op=payload['operation']
                require(tactic_identity(**payload)==q['tactic_identities'][arm][ex],"tactic identity digest")
                require(ei['gpu_uuid']==hw['uuid'].strip() and ei['driver']==hw['driver_version'].strip(),"GPU/driver alias")
                require(ei['gpu_name']==env['gpu']==hw['name'].strip() and ei['num_sms']==env['num_sm'],'GPU architecture alias')
                require(all(ei[x]==str(env[x]) for x in ('torch','cuda','flashinfer')),'runtime identity alias')
                require(ei['selector_version']==REVISION and ei['profiling_mode']=='unprofiled',"policy/profiling alias")
                require(ei['source_archive_sha256']==env['source_archive_sha256'] and ei['official_overlay_sha256']==env['official_overlay_sha256'],"source identity alias")
                require(op['execution_mode']==ex and op['q']==c['q'] and op['cached']==c['cached'],"operation geometry alias")
                require((op['dtype'],op['layout'],op['requested_split'])==key[1:],"operation mode alias")
                require(op['plan_signature']==qual['plan_info'],"plan identity alias")
                require(op['ordered_pairs_sha256']==q['ordered_pairs_sha256'],'ordered pair identity alias')
                require(op['actual_split']==('split' if bool(qual['plan_info'][14]) else 'unsplit'),'actual split alias')
                require(op['backend']=='fa2' and op['causal'] is True and (op['num_qo_heads'],op['num_kv_heads'],op['head_dim_qk'],op['head_dim_vo'])==(32,8,128,128),'unsupported operator identity')
                require(op['page_size']==(1 if key[2]=='ragged' else 16),'page size alias')
                expected_count=calls if ex=='eager_full_call' else (16 if ex=='graph1_replay' else 1)
                require(op['window_repeats']==expected_count,"window identity alias")
        qmap[key]=q
    require(set(qmap)==basic,"incomplete qualification set")
    groups=('position',) if mode=='pristine' else ('guarded','cap')
    expected={(k,b,ex,g,p) for k in basic for b in range(blocks) for ex in EXECUTIONS for g in groups for p in range(4)}
    observed=set()
    for row in rows:
        k=(row['case'],row['dtype'],row['layout'],row['split']);key=(k,row['block'],row['execution_mode'],row['comparison_group'],row['position'])
        require(key in expected and key not in observed,"timing coordinate missing/duplicate/out of contract")
        observed.add(key);arm,role=expected_sequence(mode,key[3],key[1])[key[4]]
        require((row['arm'],row['role'],row['mode'],row['rep'])==(arm,role,mode,rep),"not the frozen treatment ABBA/BAAB")
        for field in ('wall_us','device_us','window_elapsed_us'):
            require(math.isfinite(row[field]) and row[field]>0,"invalid timing")
        calls=qmap[k]['eager_calls'] if key[2]=='eager_full_call' else 16
        require(row['kernel_calls']==calls,"window count mismatch")
        require(math.isclose(row['window_elapsed_us'],row['wall_us']*calls,rel_tol=1e-10),"normalization mismatch")
        if key[2]=='eager_full_call':check_eager_window(row['window_elapsed_us'],calls)
        require(row['tactic_identity']==qmap[k]['tactic_identities'][arm][key[2]],"row identity alias")
    require(observed==expected and len(rows)==complete['rows']==complete['expected'],"incomplete timing matrix")
    require(len(quals)==complete['qualifications'],"qualification count mismatch")
    memory=json.loads((path/'memory.json').read_text())
    memory_keys={(x['case'],x['dtype'],x['layout'],x['split']) for x in memory}
    require(len(memory)==len(basic) and memory_keys==basic,'incomplete memory lifecycle audit')
    for row in memory:
        require(0<=row['allocated_after_gc']<=row['reserved_after_gc']<=row['total_device_bytes'],'invalid CUDA memory accounting')
    return env,complete,rows,quals
