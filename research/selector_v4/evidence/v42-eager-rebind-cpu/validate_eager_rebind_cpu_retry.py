"""Private CPU patch proof on a normal package; never mutates the shared env."""
import hashlib
import json
import os
import subprocess
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

base = Path('/ssd/scxi253')
out = base / 'sgi-eager-rebind-cpu-20261001T1133Z-retry'
out.mkdir()
source = base / 'sgi-official-wheel-validation-bab48695-20261001/wheel-overlay'
overlay = base / 'sgi-eager-rebind-cpu-20261001T1133Z/overlay'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
archive = base / 'eager-rebind-source-lint.tar.gz'
assert (overlay / 'test-support/pytest').is_dir()
patched = {}
with tarfile.open(archive) as t:
    for member in t.getmembers():
        assert member.isfile()
        path = overlay / member.name
        assert path.resolve().is_relative_to(overlay.resolve())
        assert sha(path) == hashlib.sha256(t.extractfile(member).read()).hexdigest()
        patched[member.name] = sha(path)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
           PYTHONPATH=str(overlay) + ':' + str(overlay / 'test-support'), PYTHONNOUSERSITE='1',
           FLASHINFER_WORKSPACE_BASE=str(out / 'cache'), XDG_CACHE_HOME=str(out / 'xdg'))
python = base / 'q7b-engines-20260925T1247Z/envs/sglang312/bin/python'
args = [str(python), '-m', 'pytest', '-q',
        '--confcutdir=tests/experimental/prefill_resource',
        'tests/experimental/prefill_resource', '--junitxml=' + str(out / 'tests.xml')]
start = time.monotonic()
with (out / 'tests.log').open('w') as f:
    result = subprocess.run(args, cwd=overlay, env=env, stdout=f, stderr=subprocess.STDOUT)
cases = ET.parse(out / 'tests.xml').findall('.//testcase')
fail = any(ET.parse(out / 'tests.xml').findall('.//' + k) for k in ('failure', 'error'))
skips = sum(case.find('skipped') is not None for case in cases)
passed = len(cases) - skips
receipt = dict(cpu_pass=result.returncode == 0 and not fail and passed == 66 and skips == 20,
               cpu_tests_passed=passed, gpu_skips=skips, returncode=result.returncode,
               elapsed_seconds=time.monotonic() - start, prior_normal_source=str(source),
               prior_source_commit='bab48695bbe72ba23c49bad69818b2975f3de6f7',
               patch_sha256=sha(archive), patched_files=patched,
               helper_sha256=sha(Path(__file__)), private_overlay_reused_without_source_mutation=True,
               prior_launcher_failure_retained=str(base / 'sgi-eager-rebind-cpu.log'),
               command=args, shared_environment_changed=False, gpu_qualified=False,
               normal_wheel_for_patched_source_built=False, fresh_cases_consumed=0,
               default_promotion=False, serving_promotion=False,
               files={n: sha(out / n) for n in ('tests.log', 'tests.xml')})
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
assert receipt['cpu_pass']
