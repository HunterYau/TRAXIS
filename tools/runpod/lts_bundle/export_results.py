"""Archive current results plus hashes, including incomplete-run metadata."""
import hashlib
import json
from pathlib import Path
import tarfile
ROOT=Path(__file__).resolve().parents[3]

if __name__=='__main__':
    folder=ROOT/'outputs/simulation-runs/lts-005'
    status=folder/'status.json'
    if not status.is_file() or json.loads(status.read_text()).get('status') not in ('completed','interrupted_or_failed','preflight_failed'):
        raise RuntimeError('Wait for completion/interruption before exporting a consistent result snapshot')
    paths=sorted(p for p in folder.rglob('*') if p.is_file() and not p.is_symlink())
    manifest={p.relative_to(folder).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:12]
    archive=ROOT/f'lts-results-{digest}.tar.gz'
    mf=ROOT/f'lts-results-{digest}.json';mf.write_text(json.dumps(manifest,indent=2))
    with tarfile.open(archive,'w:gz') as tf:
        for p in paths: tf.add(p,arcname='lts-005/'+p.relative_to(folder).as_posix(),recursive=False)
        tf.add(mf,arcname='results-sha256.json',recursive=False)
    print(json.dumps(dict(archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        files=len(paths),note='Export completed artifacts after the run to ensure a consistent snapshot.')))
