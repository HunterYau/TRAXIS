"""Group development outputs by purpose, preserving each original folder intact.

Preview by default. --apply moves entries within this project's outputs only.
The durable organization.json map also resolves historical input paths.
"""
import argparse
from pathlib import Path

from examples.plate_outputs import read_saved, owner_alive, text_file
from traxis.storage import canonical_hash, sha256_file, write_json

PROJECT = Path(__file__).resolve().parents[1]
GROUPS = {
    'simulation-runs':'Complete experimental simulation runs and their frequency tests.',
    'solver-development':'Earlier solver methods, geometry, mesh/time studies and checks, grouped by method.',
    'gpu-and-performance':'GPU execution, memory, speed and resource checks.',
    'pzt-development':'PZT physics, receiver/circuit behavior, variation and readiness checks.',
    'experimental-references':'Downloaded real experiment data, reference searches and initial validation work.',
    'result-analyses':'Additional amplitude and waveform analyses of saved experiments.',
    'test-infrastructure':'Progress logging, regression wrappers and test/tool support.',
    'science-log':'Files prepared for science log updates.',
    'supporting-files':'Older standalone scripts, archives, configuration snapshots and miscellaneous outputs.',
}


def category(name, is_directory=True):
    if name in GROUPS or name in ('organization.json','README.txt','SIMULATION-RESULTS.txt'):
        return None
    if not is_directory:
        suffix=Path(name).suffix.lower()
        kind='scripts' if suffix in ('.py','.ps1','.cmd') else 'archives' if suffix=='.zip' else 'settings-and-reports'
        return f'supporting-files/{kind}'
    prefix=name.split('-')[0]
    if prefix in ('embedded','enriched','tensor','mapped','layered','highorder','compact','matrixfree','iterative'):
        return f'solver-development/{prefix}'
    if prefix in ('recess','fullplate'):
        return 'solver-development/geometry'
    if prefix in ('green','discrete'):
        return 'solver-development/green-functions'
    if prefix=='elmer':
        return 'solver-development/independent-elmer-reference'
    if prefix=='lamb':
        return 'solver-development/wave-theory'
    if prefix in ('gpu','ram','hardware','mass'):
        return 'gpu-and-performance'
    if prefix in ('pzt','piezoelectric','loaded','damped','model'):
        return 'pzt-development'
    if prefix in ('experimental','reference'):
        return 'experimental-references'
    if prefix=='resolved':
        return 'result-analyses'
    if prefix=='science':
        return 'science-log'
    if prefix in ('progress','live','regression','recovered'):
        return 'test-infrastructure'
    return 'supporting-files/other'


def snapshot(path):
    files=sorted((p for p in path.rglob('*') if p.is_file()),key=lambda p:str(p)) if path.is_dir() else [path]
    entries=[(str(p.relative_to(path)) if path.is_dir() else p.name,p.stat().st_size) for p in files]
    important={str(p.relative_to(path)) if path.is_dir() else p.name:sha256_file(p)
               for p in files if p.name in ('result.json','comparison.json','waveforms.npz','protocol.json')}
    return dict(files=len(files),bytes=sum(size for _,size in entries),
                path_and_size_sha256=canonical_hash(entries),important_content_sha256=important)


def organize(root, *, apply=False):
    root=Path(root).resolve()
    root.relative_to(PROJECT.resolve())
    if root.name!='outputs':
        raise ValueError('Expected the project outputs directory')
    manifest_path=root/'organization.json'
    manifest=read_saved(manifest_path) or {'schema':1,'moves':{}}
    candidates=[p for p in sorted(root.iterdir()) if category(p.name,p.is_dir()) is not None]
    if not apply:
        for source in candidates:
            print(f'{source.name} -> {category(source.name,source.is_dir())}/{source.name}')
        return manifest
    # Do not move active historical queues, including continuation owners.
    for source in candidates:
        if source.is_dir():
            for owner_path in source.glob('*owner.json'):
                owner=read_saved(owner_path)
                if owner and 'created' in owner and 'started_unix' not in owner:
                    owner={**owner,'started_unix':owner['created']}
                if owner and 'pid' in owner and owner_alive(owner) is not False:
                    raise RuntimeError(f'Cannot move a live or unverified queue: {source}')
    for source in candidates:
        if source.is_symlink():
            raise ValueError(f'Refusing to move a linked output: {source}')
        destination=root/category(source.name,source.is_dir())/source.name
        destination.resolve().relative_to(root)
        source.resolve().relative_to(root)
        if destination.exists():
            raise FileExistsError(destination)
        before=snapshot(source)
        manifest['moves'][source.name]=dict(destination=destination.relative_to(root).as_posix(),
                                             verified=False,inventory=before)
        write_json(manifest_path,manifest,overwrite=True)
        destination.parent.mkdir(parents=True,exist_ok=True)
        source.rename(destination)
        if snapshot(destination)!=before:
            raise RuntimeError(f'Moved inventory differs: {destination}')
        manifest['moves'][source.name]['verified']=True
        write_json(manifest_path,manifest,overwrite=True)
    # Recover verification after an interrupted organization operation.
    for name,entry in manifest['moves'].items():
        if not entry['verified']:
            destination=root/entry['destination']
            if destination.exists() and snapshot(destination)==entry['inventory']:
                entry['verified']=True
            else:
                raise RuntimeError(f'Unverified previous move: {name}; original data has not been discarded')
    write_json(manifest_path,manifest,overwrite=True)
    lines=['Output directory','',*('  '+name+'/ - '+description for name,description in GROUPS.items()),'',
           'Every original folder keeps its name and contents within its group.',
           'organization.json records old-to-new paths, file counts, sizes and verification hashes.',
           'Historical receipts retain original paths; the current simulation resolves its moved input files.',
           'Old development tools with literal paths may need the path from this index when reused.',
           '', 'Original name -> current location:']
    lines.extend(f'{name} -> {entry["destination"]}' for name,entry in sorted(manifest['moves'].items()))
    text_file(root/'README.txt','\n'.join(lines)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    manifest=organize(PROJECT/'outputs',apply=args.apply)
    if args.apply:
        print(f'Grouped and verified {len(manifest["moves"])} original output entries. No simulations launched.')


if __name__=='__main__':
    main()
