"""Compare all compiled official native attention kernels against pristine SASS.

Only PC labels and whitespace are normalized. Instructions, operand immediates,
encoded words and resource metadata are retained. Missing/different kernels fail.
"""
from __future__ import annotations
import hashlib
import json
import re
import subprocess
from pathlib import Path

NATIVE_NAMES = ('BatchPrefillWithRaggedKVCacheKernel', 'BatchPrefillWithPagedKVCacheKernel')


def parse_sass(text):
    result = {}
    architecture = None
    for block in re.split(r'(?=\s*Function\s*:\s*)', text):
        arch = re.findall(r'arch\s*=\s*(sm_\d+)', block)
        if arch: architecture = arch[-1]
        match = re.search(r'Function\s*:\s*(\S+)', block)
        if not match or not any(name in match.group(1) for name in NATIVE_NAMES): continue
        if architecture is None: raise RuntimeError('SASS architecture missing')
        lines = []
        for line in block[match.end():].splitlines():
            # Section boundaries/file headers are outside instruction identity.
            if not line.strip() or set(line.strip()) == {'-'}: continue
            if 'Fatbin' in line or line.strip().startswith(('code for ', 'arch =', 'identifier =', 'host =', 'compile_size =')): break
            line = re.sub(r'/\*[0-9a-fA-F]{4,8}\*/', '', line)
            lines.append(' '.join(line.split()))
        if not any(';' in line for line in lines): raise RuntimeError('empty native SASS')
        key = architecture+':'+match.group(1)
        value = hashlib.sha256('\n'.join(lines).encode()).hexdigest()
        if key in result and result[key] != value: raise RuntimeError('conflicting native SASS '+key)
        result[key] = value
    return result


def collect(workspace, outdir, label):
    files = sorted((Path(workspace)/'.cache/flashinfer').rglob('*.so'))
    if not files: raise RuntimeError('missing '+label+' JIT binaries')
    symbols = {}; receipts = []
    for index,path in enumerate(files):
        proc = subprocess.run(['cuobjdump','--dump-sass',str(path)],capture_output=True,text=True,timeout=180)
        if proc.returncode: raise RuntimeError('cuobjdump failed: '+proc.stderr[-2000:])
        parsed = parse_sass(proc.stdout)
        if not parsed: continue
        raw = outdir / f'{label}-{index}.sass';raw.write_text(proc.stdout)
        for key,value in parsed.items():
            if key in symbols and symbols[key] != value: raise RuntimeError('cross-module SASS drift '+key)
            symbols[key] = value
        receipts.append({'binary':str(path),'binary_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'sass':str(raw),'sass_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
                         'native_symbols':len(parsed)})
    if not all(any(name in key for key in symbols) for name in NATIVE_NAMES):
        raise RuntimeError('incomplete '+label+' native kernel coverage')
    return symbols,receipts


def compare_native_sass(pristine_workspace, candidate_workspace, outdir):
    outdir = Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    pristine, pp = collect(pristine_workspace,outdir,'pristine')
    candidate, cp = collect(candidate_workspace,outdir,'candidate')
    missing = sorted(set(pristine)-set(candidate));extra=sorted(set(candidate)-set(pristine))
    mismatches = sorted(k for k in pristine.keys() & candidate.keys() if pristine[k]!=candidate[k])
    result={'native_sass_identity':not (missing or extra or mismatches),'native_kernel_count':len(pristine),
            'missing':missing,'extra':extra,'mismatches':mismatches,'pristine':pristine,'candidate':candidate,
            'artifacts':pp+cp,'normalization':'PC labels/whitespace only; operands, encoding and resource metadata retained'}
    (outdir/'identity.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
