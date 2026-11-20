"""Plate-run locations, saved status, and exclusive resumable queue ownership."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

import psutil

from traxis.storage import atomic_write, write_json
from examples.output_locations import organized_output

PROJECT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT/'outputs/simulation-runs'
CURRENT_RUN = RUNS_ROOT/'run-010-wandowski-restored-ideal-properties'
# The intermediate plate grouping is retained only as a historical path alias.
OLD_GROUPED_RUNS = {
    'wandowski-pzt-plate1': 'archive/run-000-simplified-pzt',
    'wandowski-resolved-pzt-v1': 'archive/run-001-body-setup',
    'wandowski-resolved-pzt-v2': 'archive/run-002-preparation',
    'wandowski-resolved-pzt-v3': 'archive/run-003-preparation',
    'wandowski-resolved-pzt-v4-check': 'archive/setup-checks/v4-first',
    'wandowski-resolved-pzt-v4-check2': 'archive/setup-checks/v4-second',
    'wandowski-resolved-pzt-v5': 'archive/run-005-initial-body-run',
    'wandowski-resolved-pzt-v6': 'archive/run-006-ideal-bond-50-100khz',
    'wandowski-resolved-pzt-v7': 'archive/run-007-ideal-bond-150-250khz-continuation',
    'wandowski-resolved-pzt-v8': 'archive/run-008-damping-preparation',
    'wandowski-resolved-pzt-v8-check': 'archive/setup-checks/v8-damping',
    'wandowski-resolved-pzt-v9': 'run-009-realistic-bond-damping',
    'wandowski-resolved-pzt-v9-check': 'archive/setup-checks/v9-first',
    'wandowski-resolved-pzt-v9-check-final': 'archive/setup-checks/v9-final',
}
LEGACY_RUNS = {
    old: ('setup-checks/wandowski-'+location.rsplit('/',1)[-1]
          if '/setup-checks/' in location else
          location.rsplit('/',1)[-1][:8]
          + 'wandowski-' + location.rsplit('/',1)[-1][8:])
    for old,location in OLD_GROUPED_RUNS.items()
}


def relocated(path):
    """Resolve historical paths for the analysis tools without rewriting receipts."""
    path = Path(path)
    if path.exists():
        return path
    grouped = organized_output(path)
    if grouped != path:
        return grouped
    try:
        parts = path.resolve().relative_to(PROJECT/'outputs').parts
    except ValueError:
        return path
    if parts and parts[0] in LEGACY_RUNS:
        target = RUNS_ROOT/LEGACY_RUNS[parts[0]]/Path(*parts[1:])
        if not target.exists() and 'native' in parts:
            target = RUNS_ROOT/LEGACY_RUNS[parts[0]]/Path(*(p for p in parts[1:] if p != 'native'))
        return target
    if len(parts)>2 and parts[:2]==('plate-experiments','wandowski-plate-1'):
        tail=Path(*parts[2:])
        for old,location in OLD_GROUPED_RUNS.items():
            try:
                suffix=tail.relative_to(location)
            except ValueError:
                continue
            return relocated(PROJECT/'outputs'/old/suffix)
    return path


def result_directory(trial):
    trial = Path(trial)
    return trial/'native' if (trial/'native').is_dir() else trial


def read_saved(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except FileNotFoundError:
        return None


def text_file(path, text):
    atomic_write(Path(path), lambda f: f.write(text.encode('utf-8')), overwrite=True)


def owner_alive(owner):
    if not owner:
        return False
    if owner.get('finished_unix') is not None:
        return False
    try:
        process = psutil.Process(owner['pid'])
        return process.is_running() and abs(process.create_time()-owner['started_unix']) < 10
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, KeyError):
        return None


def process_owner():
    return dict(pid=os.getpid(), started_unix=psutil.Process().create_time())


@contextmanager
def queue_lock(root):
    """OS lock is released even if the scheduler is killed or the PC reboots."""
    path = Path(root)/'queue.lock'
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('This experiment queue is already running.') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def trial_status(trial, status, **details):
    trial = Path(trial)
    write_json(trial/'status.json', dict(status=status, updated_unix=time.time(), **details), overwrite=True)
    if status in ('interrupted', 'failed'):
        text_file(trial/(status.upper()+'.txt'),
                  f'This frequency test is {status.upper()} and did not finish successfully.\n'
                  + details.get('reason', '')+'\n')


def saved_trial_status(root, trial, *, refresh=False):
    """Recognize an abrupt shutdown when no Python finally block could run."""
    trial = Path(trial)
    status = read_saved(trial/'status.json') or {}
    if status.get('status') == 'running':
        worker = read_saved(result_directory(trial)/'worker-owner.json')
        queue = read_saved(Path(root)/'queue-owner.json')
        if owner_alive(worker) is False and owner_alive(queue) is False:
            status = dict(status='interrupted', reason='Scheduler and worker stopped before saving a completed result.')
            if refresh:
                trial_status(trial, **status)
    return status


def archive_attempt(trial):
    """Retain incomplete files in the same frequency folder before retrying."""
    trial = Path(trial)
    if not trial.exists():
        trial.mkdir(parents=True)
        return
    entries = [p for p in trial.iterdir() if p.name != 'previous-attempts']
    if not entries:
        return
    status = read_saved(trial/'status.json') or {}
    if status.get('status') not in ('failed', 'interrupted'):
        trial_status(trial, 'interrupted', reason='No completed result was saved before this retry.')
    history = trial/'previous-attempts'
    history.mkdir(exist_ok=True)
    number = 1
    while (history/f'attempt-{number:03d}-unfinished').exists():
        number += 1
    target = history/f'attempt-{number:03d}-unfinished'
    target.mkdir()
    for entry in list(trial.iterdir()):
        if entry != history:
            entry.rename(target/entry.name)
