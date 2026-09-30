"""Strict v4 run-evidence validation before any statistic is computed."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import hashlib,json,math
from manifest import load
from autotune import canonical_hash
from measurement_contract import REVISION,EAGER_MIN_WINDOW_US

EXECUTIONS=('eager_full_call','graph1_replay','graph16_replay')
GROUPS={'calibration':('candidate','null'),'evaluation':('chosen','oracle','null')}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(value,message):
    if not value:raise ValueError(message)

def expected_cases(stage,shard,shards):
    m=load();base=[c for c in m['cases'] if c['family']==stage]
    return [c for i,c in enumerate(base) if i%shards==shard]

def load_run(root:Path,path:Path,*,mode:str,stage:str,gpu:str,rep:int,shard:int,shards:int,decisions:Path|None=None):
    root=Path(root);path=Path(path);complete=json.loads((path/'complete.json').read_text())
    need(complete.get('complete') is True,'incomplete run')
    required={'environment.json','rows.json','qualification.json','memory.json','progress.json'}
    files=complete.get('files',{});need(required<=set(files),'missing artifact hash')
    for name,h in files.items():
        need(Path(name).name==name,'unsafe artifact name');need(sha(path/name)==h,'artifact digest '+name)
    env=json.loads((path/'environment.json').read_text());rows=json.loads((path/'rows.json').read_text());quals=json.loads((path/'qualification.json').read_text());memory=json.loads((path/'memory.json').read_text())
    need((env['mode'],env['stage'],env['rep'],env['shard'],env['shards'])==(mode,stage,rep,shard,shards),'run identity')
    need(env['measurement_revision']==REVISION and env['profiled'] is False,'measurement identity')
    partition={'gpu_4090':'NVIDIA GeForce RTX 4090','gpu_5090':'NVIDIA GeForce RTX 5090'}
    need(gpu in partition and env['gpu_name']==partition[gpu],'GPU identity')
    cases=expected_cases(stage,shard,shards);need(env['cases']==cases,'case assignment')
    manifest=load();need(env['case_hash']==manifest['case_hash'] and env['release_hash']==manifest['release_hash'],'manifest identity')
    source=root/'source'
    for name,h in env.get('source',{}).items():need(sha(source/name)==h,'measured source drift '+name)
    need(env['source_archive_sha256']==(root/'receipts/source-archive.sha256').read_text().strip(),'source archive identity')
    need(env['overlay_sha256']==sha(root/'overlays/v4/RESOURCE_BINDING.json'),'overlay identity')
    if mode=='evaluation':
        need(decisions is not None and env.get('decisions_sha256')==sha(decisions),'decision identity')
    basic={(c['id'],dt,layout,split) for c in cases for dt in manifest['dtypes'] for layout in manifest['layouts'] for split in manifest['splits']}
    qmap={}
    for q in quals:
        key=(q['case'],q['dtype'],q['layout'],q['split']);need(key in basic and key not in qmap,'qualification coordinate')
        need(q.get('exact') is True and q.get('cap_supported')==(not bool(q['native_plan'][14])),'qualification status')
        need(q['native_plan'][:-1]==q['cap_plan'][:-1]==q['pool_plan'][:-1],'plan core')
        need(q['native_plan'][-1]==0 and (q['cap_plan'][-1]==1 if q['cap_supported'] else q['cap_plan'][-1]==0),'policy flags')
        need(bool(q['pool_plan'][-1])==bool(q['candidate_pool']['eligible']),'candidate-pool agreement')
        for ex in EXECUTIONS:
            item=q['identities'][ex];need(item['sha256']==canonical_hash(item['payload']),'identity hash')
        if mode=='evaluation':need(set(q['chosen'])==set(EXECUTIONS) and set(q['chosen'].values())<={'native','cap'},'chosen tactics')
        qmap[key]=q
    need(set(qmap)==basic and len(quals)==complete['qualifications'],'qualification completeness')
    groups=GROUPS[mode];expected={(key,b,ex,g,pos) for key in basic for b in range(8) for ex in EXECUTIONS for g in groups for pos in range(4)};seen=set()
    for row in rows:
        key=(row['case'],row['dtype'],row['layout'],row['split']);coord=(key,row['block'],row['execution_mode'],row['comparison_group'],row['position'])
        need(coord in expected and coord not in seen,'row coordinate');seen.add(coord)
        need(row['mode']==mode and row['rep']==rep and row['execution_mode'] in EXECUTIONS,'row identity')
        seqA=(0,3) if row['block']%2==0 else (1,2);need(row['role']==('A' if row['position'] in seqA else 'B'),'ABBA role')
        need(math.isfinite(row['wall_us']) and row['wall_us']>0 and math.isfinite(row['device_us']) and row['device_us']>0,'timing')
        if row['execution_mode']=='eager_full_call':need(row['wall_us']*row['kernel_calls']>=EAGER_MIN_WINDOW_US,'eager window')
        need(row['identity_sha256']==qmap[key]['identities'][row['execution_mode']]['sha256'],'row tactic identity')
        need(row['arm'] in ('native','cap') and row['actual_arm'] in ('native','cap'),'arm')
        if not qmap[key]['cap_supported']:need(row['actual_arm']=='native','unsupported cap executed')
        if mode=='evaluation' and row['comparison_group']=='chosen':
            chosen=qmap[key]['chosen'][row['execution_mode']];need(row['actual_arm']==('native' if row['role']=='A' else chosen),'chosen execution')
    need(seen==expected and len(rows)==complete['rows'],'row completeness')
    memory_keys={(x['case'],x['dtype'],x['layout'],x['split']) for x in memory};need(len(memory)==len(basic) and memory_keys==basic,'memory audit')
    for item in memory:need(0<=item['allocated_after_gc']<=item['reserved_after_gc']<=item['total_device_bytes'],'memory accounting')
    return env,rows,quals,complete
