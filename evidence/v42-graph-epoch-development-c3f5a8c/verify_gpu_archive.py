"""Independent complete-byte and actual-launch audit of this minimal driver."""
import hashlib
import json
import math
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            value.update(block)
    return value.hexdigest()


def verify(root):
    controller=json.loads((root/'controller.json').read_text())
    proof=json.loads((root/'verification-receipt.json').read_text())
    index=json.loads((root/'archive-members.json').read_text())
    archive=root/'raw-complete.tar.gz'
    assert controller['terminal'] and controller['state']=='DUAL_GRAPH_EPOCH_DIAGNOSTIC_PASS'
    assert all(controller[x] is False for x in ('qualification_authority','default_promotion',
                                               'serving_promotion','historical_token_divergence_resolved'))
    assert sha(archive)==proof['archive_sha256']==controller['archive']['archive_sha256']
    assert archive.stat().st_size==proof['archive_bytes']
    raw={};seen=set()
    with tarfile.open(archive,'r|gz') as t:
        for member in t:
            assert member.isfile() and member.name in index and member.name not in seen
            keep=(member.name.startswith('runs/') and member.name.endswith('.json') or
                  member.name.startswith('receipts/') and member.name.endswith(('.json','.xml','.txt')) or
                  member.name in ('binding.json','harness/harness-binding.json') or
                  member.name.startswith('sass-') and member.name.endswith('/receipt.json'))
            digest=hashlib.sha256();data=[]
            with t.extractfile(member) as stream:
                for block in iter(lambda:stream.read(1 << 20),b''):
                    digest.update(block)
                    if keep:data.append(block)
            assert index[member.name]=={'sha256':digest.hexdigest(),'bytes':member.size},member.name
            seen.add(member.name)
            if keep:raw[member.name]=b''.join(data)
    assert seen==set(index) and len(seen)==proof['verified_members']==1994
    binding=json.loads(raw['binding.json'])
    committed=json.loads(raw['harness/harness-binding.json'])
    assert binding['harness_commit']==committed['harness_commit']=='c3f5a8cba2b382b151a9908477bf3728f60ffea7'
    assert all(binding[x] is False for x in ('qualification_authority','default_promotion',
                                           'serving_promotion','historical_token_divergence_resolved'))
    for name,digest in committed['source_files'].items():assert index['harness/'+name]['sha256']==digest
    expected={f'{d}-{l}-{p}-graph{n}' for d in ('float16','bfloat16') for l in ('NHD','HND')
              for p in ('packed','tuple') for n in (1,16)}
    cards={};total_launches=0;total_epochs=0
    for gpu,job in controller['jobs'].items():
        assert controller['analyses'][gpu]['pass'] and not controller['analyses'][gpu]['failures']
        assert all(controller['analyses'][gpu][x] is False for x in ('qualification_authority',
                   'default_promotion','serving_promotion','historical_token_divergence_resolved'))
        assert controller['slurm'][job]=={'state':'COMPLETED','exit':'0:0'}
        assert raw[f'receipts/exit-{job}.txt'].strip()==b'0'
        cases=ET.fromstring(raw[f'receipts/tests-{job}.xml']).findall('.//testcase')
        assert len(cases)==24 and all(c.find(x) is None for c in cases for x in ('skipped','error','failure'))
        for stage in ('helper','native','sdk','tooling'):
            a=raw[f'receipts/{stage}-pre-{job}.txt'];b=raw[f'receipts/{stage}-post-{job}.txt']
            assert a==b and a.strip() and all(line.endswith(b': OK') for line in a.splitlines())
        prefix=f'runs/{gpu}-{job}/'
        records={Path(name).parent.name:json.loads(data) for name,data in raw.items()
                 if name.startswith(prefix) and name.endswith('/result.json')}
        assert set(records)==expected
        launches=0;copy_sizes=set()
        for cell,result in records.items():
            assert result['cell']==cell and gpu[4:] in result['gpu_name'] and result['gpu_uuid']
            assert all(result[x] is True for x in ('pass_functional','stale_graph_never_replayed',
                'unchanged_capture_across_three_metadata_epochs','unannounced_inference_tensor_write_detected',
                'changed_ordered_geometry_native'))
            assert all(result[x] is False for x in ('qualification_authority','default_promotion',
                'serving_promotion','historical_token_divergence_resolved','managed_certificate_reused'))
            probe=result['probe'];count=result['graph_replays_per_call']
            assert count in (1,16) and count==int(cell.rsplit('graph',1)[1])
            assert probe['fixed_geometry_metadata_epochs']==3 and probe['captured_buffers_owned']
            assert probe['metadata_readback_included'] and probe['resource_graph_serving_qualified'] is False
            assert len(probe['copied_bytes_per_epoch'])==3 and all(x>0 for x in probe['copied_bytes_per_epoch'])
            copy_sizes.update(probe['copied_bytes_per_epoch'])
            epochs=result['epochs'];assert len(epochs)==3
            assert len({e['physical_pages_sha256'] for e in epochs})==3
            assert len({e['output_lse_sha256'][0] for e in epochs})==3
            for number,epoch in enumerate(epochs,1):
                assert epoch['epoch']==number and epoch['output_lse_exact'] and epoch['native_after_resource_exact']
                wall=epoch['full_transaction_with_native_reference_plus_four_calls_wall_ms']
                assert math.isfinite(wall) and wall>0
                data=raw[prefix+cell+f'/epoch{number}-trace.json']
                assert hashlib.sha256(data).hexdigest()==epoch['trace_sha256']
                events=json.loads(data)['traceEvents']
                kernels=[e for e in events if e.get('cat')=='kernel' and 'BatchPrefillWithPagedKV' in e.get('name','')]
                assert len(kernels)==epoch['attention_launches']==4*count
                assert all('ResourceKernel' in e['name'] and e['args']['shared memory']==65536 for e in kernels)
                launches+=len(kernels);total_epochs+=1
        sass=json.loads(raw[f'sass-{job}/receipt.json'])
        assert sass['pass'] and sass['paired_kernels']==176
        assert not sass['missing'] and not sass['extra'] and not sass['mismatches']
        assert sass['native']==sass['resource'] and len(sass['native'])==176
        for artifact in sass['artifacts']:
            binary=str(Path(artifact['binary']).relative_to(Path(binding['root'])))
            assert index[binary]['sha256']==artifact['binary_sha256']
            assert index[f'sass-{job}/'+artifact['sass']]['sha256']==artifact['sass_sha256']
        cards[gpu]={'job':job,'gpu_cells':16,'metadata_epochs':48,'pytest_passed':24,
                    'independent_sass_pairs':176,'actual_resource_attention_launches':launches,
                    'copied_bytes_per_epoch_observed':sorted(copy_sizes)}
        total_launches+=launches
    assert total_epochs==96 and total_launches==3264
    return {'pass_fixed_capture_metadata_minimal_driver':True,'scope':'Diagnostic only; fixed exposed geometry and private captured metadata/workspaces',
            'harness_commit':binding['harness_commit'],'verified_archive_members':len(seen),
            'archive_sha256':proof['archive_sha256'],'archive_bytes':proof['archive_bytes'],
            'cards':cards,'total_gpu_cells':32,'total_metadata_epochs':total_epochs,
            'actual_traced_resource_attention_launches':total_launches,'actual_shared_memory_bytes':65536,
            'full_http_qualified':False,'resource_graph_serving_qualified':False,'fresh_cases_consumed':0,
            'qualification_authority':False,'default_promotion':False,'serving_promotion':False,
            'original_canary_verdict':'HOLD','historical_token_divergence_resolved':False}


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    result=verify(root)
    (root/'local-full-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
