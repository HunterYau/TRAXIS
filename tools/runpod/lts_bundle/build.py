"""Create a self-contained local transfer snapshot; no network operations."""
import argparse
import re
import hashlib
import json
from pathlib import Path
import tarfile
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))


def build(package='lts-six-frequency-003'):
    if not re.fullmatch(r'lts-six-frequency-[0-9]+',package): raise ValueError('Invalid package name')
    from examples import pzt_local_time_study as study
    plan_path=ROOT/'outputs/simulation-runs/lts-005/plan.json'
    plan=json.loads(plan_path.read_text())
    study.validate_plan(plan)
    if plan['identity']!=study.identity(): raise ValueError('Source snapshot changed; reprepare plan')
    if plan['frequencies_hz']!=study.FREQUENCIES or plan['tasks']!=['control-kernel','lts-kernel']:
        raise ValueError('Wrong benchmark schedule')
    dest=ROOT/'outputs/runpod-transfer'/package
    dest.mkdir(parents=True,exist_ok=False)
    paths=[]
    for folder in ('traxis','examples','tests'):
        paths.extend((ROOT/folder).rglob('*.py'))
    paths.extend(Path(__file__).parent.iterdir())
    paths.extend([ROOT/'tools/runpod/requirements-linux-lock.txt',
        ROOT/'tools/runpod/requirements-lts-eigensolver-linux-lock.txt',
        ROOT/'configs/pzt/resolved-wandowski-ideal-restored.json',
        ROOT/'configs/pzt/nce51-noliac-ver1404.json',
        ROOT/'docs/local-time-stepping-benchmark.md',
        ROOT/'docs/lts-stability-acceleration.md',plan_path])
    paths=sorted(set(p for p in paths if p.is_file()))
    rows=[]
    for p in paths:
        name=p.relative_to(ROOT).as_posix()
        if p.is_symlink() or any(s.startswith('.') for s in p.relative_to(ROOT).parts):
            raise ValueError('Hidden or linked file rejected: '+name)
        rows.append(dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    manifest=dict(schema=1,package=package,files=rows,
        plan='outputs/simulation-runs/lts-005',requires='Linux x86_64, Python 3.12, CUDA 12.9-compatible driver',
        full_simulation_started=False,includes_experimental_recordings=False,
        includes_credentials=False,includes_existing_results=False)
    mf=dest/'bundle-manifest.json';mf.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    archive=dest/('traxis-'+package+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:
        for p in paths: tf.add(p,arcname='TRAXIS/'+p.relative_to(ROOT).as_posix(),recursive=False)
        tf.add(mf,arcname='TRAXIS/bundle-manifest.json',recursive=False)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    (dest/'archive.sha256').write_text(digest+'  '+archive.name+'\n',encoding='ascii')
    print(json.dumps(dict(folder=str(dest),archive=archive.name,files=len(rows),bytes=archive.stat().st_size,sha256=digest)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--package',default='lts-six-frequency-003')
    build(parser.parse_args().package)
