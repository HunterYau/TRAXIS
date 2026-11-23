"""Build a portable snapshot; never provision resources or run a simulation."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

PROJECT = Path(__file__).resolve().parents[2]
CODE_DIRS = {'traxis', 'examples', 'tests', 'configs', 'docs', 'tools'}
CODE_TOP = {'README.md', 'AGENTS.md', 'pyproject.toml', '.gitignore', '.gitattributes'}
REFERENCE_DIRS = (
    'outputs/pzt-development/analytic-drive-ab-20261122',
    'outputs/pzt-development/full-plate-convergence-20261121',
    'outputs/simulation-runs/run-010-wandowski-restored-ideal-properties',
    'outputs/result-analyses',
)
REFERENCE_SUFFIXES = {'.json', '.txt', '.md', '.png', '.csv'}
PREPARED_STUDIES = (
    'outputs/simulation-runs/runpod-broadband-benchmark',
    'outputs/simulation-runs/runpod-broadband-convergence',
)
FORBIDDEN_PARTS = {'.git', '.venv', '__pycache__', '.ssh', '.codex', '.agents',
                   '.aws', 'cache', 'previous-attempts'}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-c', f'safe.directory={PROJECT.as_posix()}',
                                    *args], cwd=PROJECT)


def check_path(path):
    relative = path.relative_to(PROJECT)
    if (any(part in FORBIDDEN_PARTS for part in relative.parts)
            or any(parent.is_symlink() for parent in (path, *path.parents))
            or path.name.startswith('.env') or path.suffix in {'.pem', '.key', '.pfx'}):
        raise ValueError(f'Forbidden transfer path: {relative}')
    path.resolve().relative_to(PROJECT.resolve())
    if not path.is_file():
        raise FileNotFoundError(path)


def collect():
    from examples import resolved_pzt_queue as queue
    proof = queue.preflight()  # CPU input/checksum checks only.
    selected = {}
    names = git('ls-files', '-z', '--cached', '--others', '--exclude-standard').decode().split('\0')
    for name in filter(None, names):
        relative = Path(name)
        if (relative.parts[0] in CODE_DIRS or name in CODE_TOP
                or (len(relative.parts) == 1 and name.startswith('requirements') and name.endswith('.txt'))):
            selected[PROJECT / relative] = 'code-and-documentation'
    config = proof['configuration']
    required = [queue.organized_output(PROJECT / config['data_file']),
                PROJECT / config['restored_from']['prepared_receipt'],
                PROJECT / 'outputs/organization.json']
    for path in required:
        selected[path] = 'required-input'
    if (PROJECT / 'examples/pzt_broadband_study.py').exists():
        from examples.pzt_broadband_study import load_plan
        for name in PREPARED_STUDIES:
            root = PROJECT / name
            load_plan(root)
            if json.loads((root / 'status.json').read_text())['status'] != 'prepared':
                raise ValueError('This transfer expects an unstarted study: ' + name)
            for name in ('plan.json', 'status.json', 'findings.json', 'findings.txt',
                         'experimental-results.txt', 'frozen-source.zip'):
                selected[root / name] = 'prepared-unstarted-study'
    for root_name in REFERENCE_DIRS:
        root = PROJECT / root_name
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in root.rglob('*'):
            if (path.is_file() and not FORBIDDEN_PARTS.intersection(path.relative_to(PROJECT).parts)
                    and (path.suffix in REFERENCE_SUFFIXES or path.name == 'waveforms.npz')):
                selected.setdefault(path, 'historical-reference-only')
    rows = []
    for path, role in sorted(selected.items()):
        check_path(path)
        if path.stat().st_size > 32 * 1024**2:
            raise ValueError(f'Unexpected large selected file: {path}')
        if role == 'code-and-documentation' and path.suffix in {'.py', '.json', '.toml', '.md', '.txt', '.sh', '.ps1'}:
            content = path.read_bytes()
            if any(line.strip() in {b'-----BEGIN OPENSSH PRIVATE KEY-----',
                                    b'-----BEGIN RSA PRIVATE KEY-----',
                                    b'-----BEGIN PRIVATE KEY-----',
                                    b'-----BEGIN EC PRIVATE KEY-----'} for line in content.splitlines()):
                raise ValueError(f'Private-key material in selected file: {path}')
        rows.append(dict(path=path.relative_to(PROJECT).as_posix(), role=role,
                         bytes=path.stat().st_size, sha256=sha(path)))
    return proof, rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--destination', type=Path, required=True,
                        help='new directory under project outputs')
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.relative_to((PROJECT / 'outputs').resolve())
    if destination.exists():
        raise FileExistsError('Choose a new destination; existing bundles are never overwritten')
    proof, rows = collect()
    manifest = dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(),
        git_head=git('rev-parse', 'HEAD').decode().strip(),
        git_branch=git('branch', '--show-current').decode().strip(),
        git_status=git('status', '--short').decode(),
        preflight_fingerprint=proof['fingerprint'], files=rows,
        exclusions=['Windows virtual environment, OS artifacts, credentials, Git internals',
                    'Unrelated historical outputs, GPU caches, and large final-state arrays'],
        historical_results_resume_portable=False,
        note='Historical plans keep original Windows paths and ownership receipts. Do not resume them on Linux. '
             'Prepare a fresh study after selecting and approving the intended experiments. '
             'The analytic source is implemented only in the controlled trial; production/convergence defaults '
             'still use linear interpolation. The optional broadband comparison uses matched analytic sources '
             'in both methods and leaves the original solver unchanged; see docs/broadband-comparison.md.')
    payload = (json.dumps(manifest, indent=2) + '\n').encode()
    destination.mkdir(parents=True)
    archive_path = destination / 'traxis-transfer.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for row in rows:
            path = PROJECT / row['path']
            if sha(path) != row['sha256']:
                raise ValueError(f'File changed during packaging: {path}')
            info = archive.gettarinfo(str(path), arcname='TRAXIS/' + row['path'])
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mode = 0o644
            with path.open('rb') as stream:
                archive.addfile(info, stream)
        info = tarfile.TarInfo('TRAXIS/transfer-manifest.json')
        info.size = len(payload)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(payload))
    (destination / 'transfer-manifest.json').write_bytes(payload)
    digest = sha(archive_path)
    (destination / 'SHA256SUMS').write_text(f'{digest}  {archive_path.name}\n', encoding='ascii')
    print(f'Prepared {len(rows)} files; {archive_path.stat().st_size / 1024**2:.2f} MiB')
    print(archive_path)
    print('SHA256:', digest)
    print('No upload, pod creation, or simulation performed.')


if __name__ == '__main__':
    main()
