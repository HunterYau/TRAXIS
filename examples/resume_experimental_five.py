"""Resume only the three outstanding frequencies without changing frozen physics."""
import argparse
import json
import os
from pathlib import Path
import time

import psutil

from examples.experimental_five import ROOT, PROJECT, protocol, record, run_one
from examples.report_experimental_five import report
from traxis.storage import canonical_hash, sha256_file, write_json

REMAINING = (150000, 200000, 250000)


def preflight():
    if json.loads((ROOT/'protocol.json').read_text()) != protocol():
        raise ValueError('frozen physics or source identity changed')
    for process in psutil.process_iter(['name']):
        try:
            if process.pid == os.getpid() or (process.info['name'] or '').lower() != 'python.exe':
                continue
            if any(module in process.cmdline() for module in (
                    'examples.experimental_five', 'examples.run_experimental_five_queue',
                    'examples.resume_experimental_five')):
                raise RuntimeError(f'an experiment process is already active: {process.pid}')
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    for index, frequency in ((1, 50000), (2, 100000)):
        path = PROJECT/f'docs/evidence/experimental-trial-{index}-{frequency//1000}khz-20261116.json'
        evidence = json.loads(path.read_text())
        if evidence['fingerprint'] != canonical_hash({k:v for k,v in evidence.items() if k!='fingerprint'}):
            raise ValueError('completed comparison receipt changed')
        directory = ROOT/f'trial-{index}-{frequency//1000}khz'
        if (not evidence['completed']
                or sha256_file(directory/'native/result.json') != evidence['native_result_sha256']
                or sha256_file(directory/'launcher.json') != evidence['launcher_sha256']
                or sha256_file(PROJECT/'examples/report_experimental_five.py') != evidence['analysis_source_sha256']):
            raise ValueError('completed comparison or analysis source changed')
    for frequency in REMAINING:
        record(frequency)
        index = (frequency//50000)
        if (ROOT/f'trial-{index}-{frequency//1000}khz').exists():
            raise FileExistsError('preserve prior attempts before resuming; never overwrite')
    if (ROOT/'STOP_AFTER_CURRENT').exists():
        raise RuntimeError('previous stop marker must be preserved in history before this authorized resume')
    return dict(remaining_frequencies_hz=list(REMAINING), protocol_sha256=sha256_file(ROOT/'protocol.json'),
                completed_tests_preserved=[1,2], frozen_physics_unchanged=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    checked = preflight()
    if args.check_only:
        print(json.dumps(checked)); return
    owner = ROOT/'resume-v1-owner.json'
    if owner.exists():
        raise FileExistsError('preserve previous resume ownership; do not launch another queue')
    write_json(owner, dict(pid=os.getpid(), created=psutil.Process().create_time(),
                          source_sha256=sha256_file(Path(__file__)), **checked))
    with (ROOT/'resume-v1-events.jsonl').open('x', encoding='utf-8') as events:
        def event(phase, **extra):
            message = dict(phase=phase, unix_time=time.time(), **extra)
            events.write(json.dumps(message)+'\n'); events.flush()
            print(json.dumps(message), flush=True)
        try:
            event('authorized_resume_started', **checked)
            for frequency in REMAINING:
                if (ROOT/'STOP_AFTER_CURRENT').exists():
                    event('stopped_between_tests'); return
                event('starting_trial', frequency_hz=frequency, trial=frequency//50000)
                monitor = run_one(frequency)
                if not monitor['completed']:
                    raise RuntimeError(f'trial monitor did not complete: {frequency}')
                result = report(frequency)
                event('trial_saved', trial=result['trial'], frequency_hz=frequency,
                      arrival_error_us=result['arrival_error_us'],
                      flight_relative_error=result['flight_relative_error'],
                      evidence_fingerprint=result['fingerprint'])
            event('remaining_three_saved', total_completed=5)
        except BaseException as exc:
            event('resume_stopped_on_error', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    main()
