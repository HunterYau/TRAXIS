"""Package the remedy study and verified inputs. No upload or cloud compute."""
import argparse
import io
import json
from pathlib import Path
import tarfile

from tools.runpod.prepare_transfer import PROJECT, CODE_DIRS, CODE_TOP, git, sha, check_path
from examples import pzt_mor_remedies as study


def collect(root):
    plan = study.load_plan(root)
    if json.loads((root / 'status.json').read_text())['status'] != 'prepared':
        raise ValueError('Transfer expects an unstarted remedy schedule')
    selected = {}
    for name in filter(None, git('ls-files', '-z', '--cached', '--others', '--exclude-standard').decode().split('\0')):
        p = Path(name)
        if p.parts[0] in CODE_DIRS or name in CODE_TOP or (len(p.parts) == 1 and name.startswith('requirements')):
            selected[PROJECT / p] = 'code-and-documentation'
    config = plan['configuration']
    for p in [study.baseline.queue.organized_output(PROJECT / config['data_file']),
              PROJECT / config['restored_from']['prepared_receipt'], PROJECT / 'outputs/organization.json']:
        selected[p] = 'scientific-input'
    reference = PROJECT / study.baseline.REFERENCE
    for name in ['download-manifest.json', *plan['reference']['manifest']]:
        selected[reference / name] = 'verified-full-order-reference'
    for name in ['plan.json', 'status.json', 'findings.json', 'all-results.txt']:
        selected[root / name] = 'prepared-unstarted-remedies'
    control = PROJECT / 'outputs/simulation-runs/runpod-h100-nvl-mor-005'
    for name in ['findings.txt', 'findings.json', 'build-mor/build.json']:
        selected[control / name] = 'historical-original-era-control'
    rows = []
    for path, role in sorted(selected.items()):
        check_path(path)
        if path.stat().st_size > 32 * 1024**2:
            raise ValueError('Unexpected large transfer file: ' + str(path))
        content = path.read_bytes()
        if any(line.strip().startswith(b'-----BEGIN ') and b'PRIVATE KEY-----' in line
               for line in content.splitlines()):
            raise ValueError('Private key material forbidden: ' + str(path))
        rows.append(dict(path=path.relative_to(PROJECT).as_posix(), role=role,
                         bytes=path.stat().st_size, sha256=sha(path)))
    return plan, rows


def prepare(destination, root):
    destination = destination.resolve(); root = root.resolve()
    destination.relative_to((PROJECT / 'outputs/runpod-transfer').resolve())
    root.relative_to(PROJECT.resolve())
    if destination.exists():
        raise FileExistsError('Use a new transfer destination')
    plan, rows = collect(root)
    manifest = dict(schema=1, study_root=root.relative_to(PROJECT).as_posix(),
        preflight_fingerprint=plan['preflight']['fingerprint'], files=rows,
        commit=git('rev-parse', 'HEAD').decode().strip(),
        working_tree_status=git('status', '--short').decode(),
        cloud_actions_performed=False, simulations_started=False)
    payload = (json.dumps(manifest, indent=2) + '\n').encode()
    destination.mkdir(parents=True)
    archive_path = destination / 'traxis-mor-remedies.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for row in rows:
            p = PROJECT / row['path']
            if sha(p) != row['sha256']:
                raise ValueError('File changed while packaging: ' + str(p))
            info = archive.gettarinfo(str(p), arcname='TRAXIS/' + row['path'])
            info.uid = info.gid = 0; info.uname = info.gname = ''; info.mode = 0o644
            with p.open('rb') as f:
                archive.addfile(info, f)
        info = tarfile.TarInfo('TRAXIS/remedies-transfer-manifest.json')
        info.size = len(payload); info.mode = 0o644
        archive.addfile(info, io.BytesIO(payload))
    (destination / 'remedies-transfer-manifest.json').write_bytes(payload)
    (destination / 'SHA256SUMS').write_text(sha(archive_path) + '  ' + archive_path.name + '\n', encoding='ascii')
    print(f'PREPARED: {len(rows)} files, {archive_path.stat().st_size / 1024**2:.2f} MiB; no upload/run.')
    print(archive_path)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, default=study.ROOT)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args(); prepare(a.destination, a.root)


if __name__ == '__main__': main()
