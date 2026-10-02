"""Failed diagnostic populations must remain readable and fail closed."""
import json

from research.selector_v4.exposed_diagnostics.graph_epoch_controller import validate


def test_missing_files_are_hold_not_success(tmp_path):
    result = validate(tmp_path,'123','gpu_4090')
    assert not result['pass']
    assert 'missing_pytest_xml' in result['failures']
    assert 'incomplete_16_gpu_cells' in result['failures']
    assert not result['qualification_authority']


def test_partial_failure_record_still_has_an_archivable_verdict(tmp_path):
    cell = tmp_path/'runs/gpu_5090-123/float16-NHD-tuple-graph1'
    cell.mkdir(parents=True)
    # A constructor/assertion failure can occur before the final probe summary.
    # This fixture is a partial failed record, not manufactured GPU evidence.
    (cell/'result.json').write_text(json.dumps({'cell':cell.name,'epochs':[],
                                               'graph_replays_per_call':1,'gpu_name':'RTX5090'}))
    result = validate(tmp_path,'123','gpu_5090')
    assert not result['pass'] and result['cells']==1
    assert any('functional_contract' in x for x in result['failures'])
    assert any('incomplete_epochs' in x for x in result['failures'])


def test_gpu_skips_fail_completeness_even_with_24_testcase_names(tmp_path):
    receipts = tmp_path/'receipts';receipts.mkdir()
    (receipts/'tests-123.xml').write_text('<testsuites><testsuite>'+''.join(
        f'<testcase name="case{i}"><skipped/></testcase>' for i in range(24))+'</testsuite></testsuites>')
    result = validate(tmp_path,'123','gpu_4090')
    assert not result['pass']
    assert 'incomplete_or_failed_24_case_population' in result['failures']


def test_corrupt_partial_json_still_receives_archivable_hold(tmp_path):
    cell = tmp_path/'runs/gpu_5090-123/float16-NHD-tuple-graph1'
    cell.mkdir(parents=True)
    (cell/'result.json').write_text('{"epochs":')
    result = validate(tmp_path,'123','gpu_5090')
    assert not result['pass'] and result['cells']==1
    assert result['failures'][0].startswith('unreadable_evidence:JSONDecodeError:')


def test_corrupt_pytest_xml_still_receives_archivable_hold(tmp_path):
    receipts = tmp_path/'receipts';receipts.mkdir()
    (receipts/'tests-123.xml').write_text('<testsuites><testsuite>')
    result = validate(tmp_path,'123','gpu_4090')
    assert not result['pass']
    assert result['failures'][0].startswith('unreadable_evidence:ParseError:')
