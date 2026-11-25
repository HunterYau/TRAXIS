"""Package/launch/export a prepared study; download-only watcher. No pod API."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time

PROJECT=Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def alive(root):
    return _owner_alive(Path(root)/'scheduler.json') or _owner_alive(Path(root)/'launcher.json')


def _owner_alive(path):
    if not path.exists():return False
    data=json.loads(path.read_text())
    try:
        import psutil
        p=psutil.Process(data['pid'])
        return p.status()!=psutil.STATUS_ZOMBIE and abs(p.create_time()-data['created_unix'])<1
    except ImportError:
        p=Path('/proc')/str(data['pid'])/'stat'
        return p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z'
    except psutil.NoSuchProcess:return False


def bundle(root,archive):
    sys.path.insert(0,str(PROJECT));from examples.pzt_mesh_protocol import validate
    validate(root);relative=root.resolve().relative_to(PROJECT.resolve())
    paths=[]
    for name in ('traxis','examples','tests'):
        paths.extend((PROJECT/name).glob('*.py'))
    paths.extend((PROJECT/'configs/pzt-mesh-convergence').glob('*.json'))
    paths.extend((PROJECT/'configs/pzt').glob('*.json'))
    paths.extend((PROJECT/'tools/runpod').glob('*.py'));paths.extend((PROJECT/'tools/runpod').glob('*.txt'))
    paths.extend((PROJECT/'tools/runpod').glob('*.sh'))
    paths.extend((PROJECT/'tools/runpod/lts_bundle').glob('*.py'))
    paths.extend([PROJECT/'requirements-plots.txt',PROJECT/'docs/pzt-mesh-convergence.md'])
    paths.extend(p for p in root.rglob('*') if p.is_file())
    paths=sorted(set(paths));files={p.relative_to(PROJECT).as_posix():sha(p) for p in paths}
    manifest=dict(files=files,root=relative.as_posix(),baseline_included=True,baseline_rerun=False,
                  pod_actions=False,credentials_included=False)
    archive=Path(archive);archive.parent.mkdir(parents=True,exist_ok=True)
    if archive.exists():raise FileExistsError('Bundle exists; choose a new archive name')
    mf=archive.with_name(archive.name+'.manifest.json');mf.write_text(json.dumps(manifest,indent=2))
    with tarfile.open(archive,'w:gz') as tf:
        for p in paths:
            if p.is_symlink():raise ValueError('Linked source rejected')
            tf.add(p,arcname='TRAXIS/'+p.relative_to(PROJECT).as_posix(),recursive=False)
        tf.add(mf,arcname='TRAXIS/pzt-mesh-bundle-manifest.json',recursive=False)
    (archive.with_name(archive.name+'.sha256')).write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps(dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files=len(files))))


def check_bundle():
    manifest=json.loads((PROJECT/'pzt-mesh-bundle-manifest.json').read_text())
    for name,digest in manifest['files'].items():
        p=PROJECT/name;p.resolve().relative_to(PROJECT.resolve())
        if p.is_symlink() or sha(p)!=digest:raise ValueError('Bundle checksum mismatch: '+name)
    sys.path.insert(0,str(PROJECT));from examples.pzt_mesh_protocol import validate
    p=validate(PROJECT/manifest['root'])
    print(json.dumps(dict(ready=True,configurations=len(p['cases']),baseline_rerun=False,full_simulations_started=0)))


def launch(root,python,backend,retry_failed):
    sys.path.insert(0,str(PROJECT));from examples.pzt_mesh_protocol import validate
    validate(root)
    if alive(root):raise RuntimeError('Scheduler already running')
    owner=root/'launcher.json'
    if owner.exists():owner.rename(root/('launcher.previous-'+str(time.time_ns())+'.json'))
    # Use the explicit local-disk interpreter to avoid network volume header IO.
    env=os.environ.copy();env['CUPY_CACHE_DIR']='/tmp/traxis-pzt-mesh-cupy' if os.name!='nt' else str(root/'cupy-cache')
    sys.path.insert(0,str(PROJECT/'tools/runpod/lts_bundle'))
    from cpu_threads import available_threads
    count=available_threads();env.update(OPENBLAS_NUM_THREADS=str(count),OMP_NUM_THREADS=str(count),MKL_NUM_THREADS=str(count))
    command=[str(python),'-u','-m','examples.pzt_mesh_study','--run','--root',str(root.resolve()),'--stability-backend',backend]
    if retry_failed:command.append('--retry-failed')
    with (root/'console.log').open('ab',buffering=0) as log:
        flags=dict(start_new_session=True) if os.name!='nt' else dict(creationflags=subprocess.CREATE_NO_WINDOW)
        process=subprocess.Popen(command,cwd=PROJECT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,**flags)
    import psutil
    owner.write_text(json.dumps(dict(pid=process.pid,created_unix=psutil.Process(process.pid).create_time(),python=str(python),cpu_threads=count,
        automatic_download='PC watcher required',automatic_pod_stop=False),indent=2))
    print(owner.read_text())


def export(root):
    if alive(root):raise RuntimeError('Wait for scheduler completion before a consistent full export')
    state=json.loads((root/'status.json').read_text())
    if state['status'] not in ('completed','completed_with_failures','interrupted','stopped_after_failure'):
        raise RuntimeError('Run is not in a terminal state; inspect the worker before exporting')
    # Certificates are retained; the large optional warm hint is regenerable.
    files=[p for p in root.rglob('*') if p.is_file() and p.name!='warm-latest.npz']
    hashes={p.relative_to(root).as_posix():sha(p) for p in files}
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    dest=PROJECT/'outputs/pzt-mesh-exports';dest.mkdir(parents=True,exist_ok=True)
    archive=dest/(digest+'.tar.gz');manifest=dest/(digest+'.json')
    manifest.write_text(json.dumps(hashes,indent=2))
    with tarfile.open(archive,'w:gz') as tf:
        for p in files:
            if p.is_symlink():raise ValueError('Linked output rejected')
            tf.add(p,arcname='study/'+p.relative_to(root).as_posix(),recursive=False)
        tf.add(manifest,arcname='download-sha256.json',recursive=False)
    print(json.dumps(dict(archive=str(archive),sha256=sha(archive),files=len(hashes))))


def watch(args):
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9.-]*',args.host or '') or not 1<=args.port<=65535:raise ValueError('Invalid SSH endpoint')
    if not re.fullmatch(r'/workspace/[A-Za-z0-9_-]+/TRAXIS',args.remote_project or ''):raise ValueError('Use an isolated /workspace/NAME/TRAXIS project')
    if not re.fullmatch(r'outputs/simulation-runs/[A-Za-z0-9_-]+',args.remote_root):raise ValueError('Invalid remote study root')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.detach:
        argv=[sys.executable,'-u',str(__file__),*[v for v in sys.argv[1:] if v!='--detach']]
        with (output/'download-watcher.log').open('ab',buffering=0) as log:
            flags=dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name=='nt' else dict(start_new_session=True)
            p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,**flags)
        print(json.dumps(dict(watcher_pid=p.pid,pod_stop=False,output=str(output))));return
    options=['-i',str(args.key.resolve(strict=True)),'-o','BatchMode=yes','-o','ConnectTimeout=15',
             '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(args.known_hosts.resolve(strict=True))]
    host='root@'+args.host
    def ssh(command,timeout=60):
        return subprocess.check_output(['ssh',*options,'-p',str(args.port),host,command],text=True,timeout=timeout)
    prefix='cd '+shlex.quote(args.remote_project)+' && python3 tools/runpod/pzt_mesh_transfer.py '
    while True:
        try:
            data=json.loads(ssh(prefix+'status --root '+shlex.quote(args.remote_root)))
            print(json.dumps(data),flush=True)
            if data['status']['status'] in ('completed','completed_with_failures','interrupted','stopped_after_failure') and not data['alive']:
                info=json.loads(ssh(prefix+'export --root '+shlex.quote(args.remote_root),timeout=3600))
                archive=info['archive']
                if not re.fullmatch(re.escape(args.remote_project)+r'/outputs/pzt-mesh-exports/[0-9a-f]{16}\.tar\.gz',archive):
                    raise ValueError('Unexpected export path')
                destination=output/('pzt-mesh-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
                destination.mkdir(exist_ok=False)
                local=destination/'results.tar.gz'
                subprocess.run(['scp',*options,'-P',str(args.port),host+':'+archive,str(local)],check=True,timeout=3600)
                if sha(local)!=info['sha256']:raise ValueError('Archive hash mismatch')
                with tarfile.open(local) as tf:
                    for member in tf.getmembers():
                        (destination/member.name).resolve().relative_to(destination)
                        if not (member.isfile() or member.isdir()):raise ValueError('Archive link/special entry rejected')
                    tf.extractall(destination,filter='data')
                hashes=json.loads((destination/'download-sha256.json').read_text())
                for name,digest in hashes.items():
                    p=destination/'study'/name;p.resolve().relative_to((destination/'study').resolve())
                    if sha(p)!=digest:raise ValueError('Result file checksum mismatch: '+name)
                (output/'download-verified.json').write_text(json.dumps(dict(destination=str(destination),files=len(hashes),pod_left_running=True),indent=2))
                print('Downloaded and verified all results. Pod left running. '+str(destination),flush=True)
                return
        except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
            print(type(exc).__name__+': download pending; retrying in 30 seconds. Pod unchanged.',flush=True)
        time.sleep(30)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['bundle','check-bundle','launch','status','export','watch'])
    p.add_argument('--root',type=Path);p.add_argument('--archive',type=Path)
    p.add_argument('--python',type=Path,default=Path(sys.executable));p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--stability-backend',choices=['auto','cpu','gpu','original'],default='auto')
    p.add_argument('--host');p.add_argument('--port',type=int);p.add_argument('--key',type=Path);p.add_argument('--known-hosts',type=Path)
    p.add_argument('--remote-project');p.add_argument('--remote-root',default='outputs/simulation-runs/pzt-mesh-convergence-002')
    p.add_argument('--output',type=Path);p.add_argument('--detach',action='store_true');args=p.parse_args()
    if args.action=='bundle':bundle(args.root,args.archive)
    elif args.action=='check-bundle':check_bundle()
    elif args.action=='launch':launch(args.root,args.python,args.stability_backend,args.retry_failed)
    elif args.action=='export':export(args.root)
    elif args.action=='watch':watch(args)
    else:
        status=json.loads((args.root/'status.json').read_text())
        print(json.dumps(dict(status=status,alive=alive(args.root))))


if __name__=='__main__':main()
