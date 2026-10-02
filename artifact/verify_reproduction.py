"""Independent audit of the retained local/Paracloud CPU reproductions."""
import hashlib
import json
import math
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'artifact/persisted-reproduction'
REMOTE_ROOT = LOCAL / 'paracloud'
REMOTE = REMOTE_ROOT / 'artifact/generated-paracloud'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    index = json.loads((REMOTE_ROOT / 'archive-index.json').read_text())
    archive = REMOTE_ROOT / 'raw-cpu-reproduction.tar.gz'
    assert sha(archive) == index['archive_sha256']
    assert archive.stat().st_size == index['archive_bytes']
    seen = set()
    with tarfile.open(archive) as stream:
        for member in stream:
            assert member.isfile() and member.name in index['members'] and member.name not in seen
            data = stream.extractfile(member).read()
            entry = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
            assert entry == index['members'][member.name], member.name
            assert sha(REMOTE_ROOT / member.name) == entry['sha256'], member.name
            seen.add(member.name)
    assert seen == set(index['members'])
    manifests = []
    for root in (LOCAL, REMOTE):
        manifest = json.loads((root / 'manifest.json').read_text())
        for name, digest in manifest['outputs'].items():
            assert sha(root / name) == digest, name
        assert manifest['original_canary'] == 'HOLD'
        for field in ('qualification_authority', 'default_promotion', 'serving_promotion',
                      'historical_token_divergence_resolved', 'full_http_qualified'):
            assert manifest[field] is False, field
        manifests.append(manifest)
    assert manifests[0]['inputs'] == manifests[1]['inputs']
    assert manifests[0]['source_files'] == manifests[1]['source_files']
    for path, digest in manifests[0]['source_files'].items():
        assert sha(ROOT / path) == digest, path
    numeric = []

    def compare(a, b, path):
        assert type(a) is type(b), path
        if isinstance(a, dict):
            assert set(a) == set(b), path
            for key in a:
                compare(a[key], b[key], path + '.' + key)
        elif isinstance(a, list):
            assert len(a) == len(b), path
            for number, (x, y) in enumerate(zip(a, b)):
                compare(x, y, path + '.' + str(number))
        elif isinstance(a, float):
            assert math.isfinite(a) and math.isfinite(b), path
            assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14), path
            numeric.append(abs(a-b))
        else:
            assert a == b, path

    for name in ('analysis.json', 'tables.json'):
        compare(json.loads((LOCAL / name).read_text()), json.loads((REMOTE / name).read_text()), name)
    for path in LOCAL.glob('*.svg'):
        assert sha(path) == sha(REMOTE / path.name), path.name
    for path in LOCAL.glob('*.png'):
        assert np.array_equal(np.asarray(Image.open(path)), np.asarray(Image.open(REMOTE / path.name))), path.name
    for path in LOCAL.glob('*.csv'):
        assert sha(path) == sha(REMOTE / path.name), path.name
    receipt = json.loads((LOCAL / 'cross-platform-reproduction.json').read_text())
    assert receipt['numeric_float_comparisons'] == len(numeric)
    assert receipt['maximum_absolute_float_difference'] == max(numeric)
    assert len(numeric) == 1604 and len(seen) == 23
    return {'pass_independent_retained_reproduction_audit': True,
            'verified_remote_members': len(seen), 'numeric_values': len(numeric),
            'maximum_absolute_difference': max(numeric), 'six_svg_bytes_identical': True,
            'six_png_decoded_pixels_identical': True, 'original_canary': 'HOLD',
            'qualification_authority': False, 'default_promotion': False,
            'serving_promotion': False, 'historical_token_divergence_resolved': False}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
