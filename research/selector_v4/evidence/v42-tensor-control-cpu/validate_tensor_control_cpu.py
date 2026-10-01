"""Derived CPU patch proof; shared normal package is read-only via symlinks."""
import hashlib
import json
import os
import subprocess
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

base = Path('/ssd/scxi253')
out = base / 'sgi-tensor-control-cpu-20261001T1315Z'
out.mkdir()
source = base / 'sgi-official-wheel-validation-d3d080ce-20261001/wheel-overlay'
overlay = out / 'overlay'
patch = base / 'tensor-control-source.tar.gz'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
with tarfile.open(patch) as t:
    members = t.getmembers()
    assert all(m.isfile() for m in members)
    contents = {m.name: t.extractfile(m).read() for m in members}

def link_tree(old, new, prefix=''):
    new.mkdir()
    for p in old.iterdir():
        relative = prefix + p.name
        if relative in contents:
            (new / p.name).write_bytes(contents[relative])
        elif p.is_dir() and any(n.startswith(relative + '/') for n in contents):
            link_tree(p, new / p.name, relative + '/')
        else:
            (new / p.name).symlink_to(p, target_is_directory=p.is_dir())

link_tree(source, overlay)
for n, data in contents.items():
    assert (overlay / n).is_file() and not (overlay / n).is_symlink()
    assert sha(overlay / n) == hashlib.sha256(data).hexdigest()
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
           PYTHONPATH=str(overlay) + ':' + str(overlay / 'test-support'), PYTHONNOUSERSITE='1',
           PYTHONDONTWRITEBYTECODE='1', FLASHINFER_WORKSPACE_BASE=str(out / 'cache'),
           XDG_CACHE_HOME=str(out / 'xdg'))
python = base / 'q7b-engines-20260925T1247Z/envs/sglang312/bin/python'
args = [str(python), '-m', 'pytest', '-q',
        '--confcutdir=tests/experimental/prefill_resource',
        'tests/experimental/prefill_resource', '--junitxml=' + str(out / 'tests.xml')]
start = time.monotonic()
with (out / 'tests.log').open('w') as f:
    result = subprocess.run(args, cwd=overlay, env=env, stdout=f, stderr=subprocess.STDOUT)
xml = ET.parse(out / 'tests.xml') if (out / 'tests.xml').exists() else None
cases = xml.findall('.//testcase') if xml is not None else []
fail = xml is None or any(xml.findall('.//' + k) for k in ('failure', 'error'))
skips = sum(case.find('skipped') is not None for case in cases)
passed = len(cases) - skips
receipt = dict(cpu_pass=result.returncode == 0 and not fail and passed == 70 and skips == 20,
               cpu_tests_passed=passed, gpu_skips=skips, returncode=result.returncode,
               elapsed_seconds=time.monotonic() - start, prior_normal_source=str(source),
               prior_source_commit='d3d080ce36bf074136d4b2093b775a8cb1d7dfc5',
               patch_sha256=sha(patch), patched_files={n: sha(overlay / n) for n in contents},
               helper_sha256=sha(Path(__file__)), read_only_dependencies_via_symlink=True,
               command=args, shared_environment_changed=False, gpu_qualified=False,
               normal_wheel_for_patched_source_built=False, fresh_cases_consumed=0,
               default_promotion=False, serving_promotion=False,
               files={n: sha(out / n) for n in ('tests.log', 'tests.xml') if (out / n).exists()})
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
assert receipt['cpu_pass']
