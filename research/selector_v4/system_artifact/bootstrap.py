"""One-command isolated CPU reproduction; never touches CUDA/serving envs."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]


def run(out):
    if not (3,9)<=sys.version_info[:2]<=(3,12):
        raise RuntimeError('Frozen artifact dependency set requires Python3.9–3.12')
    env=ROOT/'artifact/.venv';python=env/'bin/python'
    if not python.exists():subprocess.run([sys.executable,'-m','venv',str(env)],check=True)
    lock=ROOT/'artifact/requirements.lock'
    stamp=env/'sgi-lock.sha256'
    import hashlib
    digest=hashlib.sha256(lock.read_bytes()).hexdigest()
    if not stamp.exists() or stamp.read_text()!=digest:
        subprocess.run([str(python),'-m','pip','install','--require-hashes','--only-binary=:all:',
                        '-r',str(lock)],check=True)
        stamp.write_text(digest)
    cpu_env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
             'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':str(ROOT/'artifact/.cache/matplotlib')}
    subprocess.run([str(python),'-m','research.selector_v4.system_artifact.reproduce','--out',str(out)],
                   cwd=ROOT,env=cpu_env,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'artifact/generated');a=p.parse_args();run(a.out)
