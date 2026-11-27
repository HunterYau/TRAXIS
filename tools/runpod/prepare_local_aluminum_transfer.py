"""Build a verified source/plan archive. No network, GPU, or pod operations."""
import argparse,hashlib,io,json,tarfile,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(Path(__file__).parent))
from local_aluminum_preflight import validate

def digest(data):return hashlib.sha256(data).hexdigest()
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--destination',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();relative=root.relative_to(PROJECT);validate(root)
    dest=a.destination.resolve();dest.relative_to(PROJECT/'outputs');dest.mkdir(parents=True,exist_ok=False)
    paths=[]
    for folder in ('traxis','examples','tests','configs','docs','tools'):
        paths.extend(f for f in (PROJECT/folder).rglob('*') if f.is_file() and f.suffix in ('.py','.json','.sh','.ps1','.txt','.md')
            and not any(part in ('__pycache__','.git','.venv','.ssh','cache') for part in f.relative_to(PROJECT).parts))
    paths.extend(PROJECT/name for name in ('pyproject.toml','requirements-plots.txt'))
    paths.extend(root/name for name in ('plan.json','status.json','README.txt'))
    files={};payload={}
    for path in sorted(set(paths)):
        if path.is_symlink():raise ValueError('Linked source rejected: '+str(path))
        name=path.relative_to(PROJECT).as_posix();data=path.read_bytes()
        if path.suffix=='.sh':data=data.replace(b'\r\n',b'\n')
        if any(line.startswith(b'-----BEGIN ') and b'PRIVATE KEY' in line for line in data.splitlines()):raise ValueError('Private key detected')
        payload[name]=data;files[name]=digest(data)
    manifest=dict(files=files,study_root=relative.as_posix(),simulations_started=0,pod_actions=False,
        baseline_waveforms_required=False,credentials_included=False,
        excluded=['Git internals','virtual environments','CUDA caches','old outputs','SSH keys and known-host files'])
    archive=dest/'traxis-local-aluminum.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for name,data in [*payload.items(),('local-aluminum-manifest.json',json.dumps(manifest,indent=2).encode())]:
            entry=tarfile.TarInfo('TRAXIS/'+name);entry.size=len(data);entry.mode=0o644;tf.addfile(entry,io.BytesIO(data))
    (dest/'transfer-manifest.json').write_text(json.dumps(manifest,indent=2))
    (dest/'SHA256SUMS').write_text(digest(archive.read_bytes())+'  '+archive.name+'\n',encoding='ascii')
    with tarfile.open(archive) as tf:
        for name,h in files.items():
            if digest(tf.extractfile('TRAXIS/'+name).read())!=h:raise ValueError('Archive verification failed')
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,files=len(files),archive_checksums_passed=True)))
if __name__=='__main__':main()
