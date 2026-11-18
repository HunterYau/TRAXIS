"""Read-only terminal progress and per-experiment results for a PZT run queue.

No GUI, GPU work, or additional dependency. This can be opened or closed
independently of the simulation. --once prints a snapshot without waiting.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import psutil

DEFAULT_ROOT = Path(__file__).resolve().parents[1]/'outputs/wandowski-pzt-plate1'


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except FileNotFoundError:
        return None


def progress_state(log):
    """A partial final JSON line is expected while a worker is writing."""
    try:
        with log.open('rb') as stream:
            stream.seek(0, 2)
            start = max(0, stream.tell()-65536)
            stream.seek(start)
            lines = stream.read().decode('utf-8', errors='replace').splitlines()
    except FileNotFoundError:
        return dict(phase='waiting to start', done=0, total=65536, elapsed=None)
    state = dict(phase='initializing', done=0, total=65536, elapsed=None)
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        state['phase'] = event.get('phase', state['phase'])
        if event.get('phase') == 'propagating':
            done, total = event['steps_completed'], event['requested_steps']
            if not 0 <= done <= total or total <= 0:
                raise ValueError('invalid progress counters')
            state.update(done=done, total=total, elapsed=event.get('elapsed_seconds'))
    return state


def clock(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return '--:--:--'
    seconds = int(seconds)
    return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'


def progress_line(frequency, index, count, state):
    fraction = state['done']/state['total']
    filled = int(20*fraction)
    bar = '#'*filled + '-'*(20-filled)
    elapsed = state['elapsed']
    eta = elapsed*(1/fraction-1) if fraction and elapsed is not None else None
    return (f'Plate 1 | test {index}/{count} | {frequency//1000} kHz '
            f'[{bar}] {fraction:5.1%} | {state["done"]:,}/{state["total"]:,} steps '
            f'| elapsed {clock(elapsed)} | ETA ~{clock(eta)} | {state["phase"]}')


def result_lines(frequency, result, directory):
    status = 'COMPLETED' if result.get('completed') else 'FAILED / INCOMPLETE'
    lines = [f'{frequency//1000} kHz: {status} | time {clock(result.get("wall_seconds"))}']
    for key, label in [('cpu_force_relative_error', 'CPU/GPU force difference'),
                       ('modified_energy_relative_drift', 'Energy drift')]:
        value = result.get(key)
        if value is not None:
            lines.append(f'  {label}: {value:.3e} (relative)')
    comparison = result.get('comparison')
    if isinstance(comparison, dict):
        for name, value in comparison.items():
            lines.append(f'  {name}: {value}')
    else:
        lines.append('  Comparison with real measurements: pending analysis.')
    if result.get('error'):
        lines.append(f'  Error: {result["error"]}')
    lines.append(f'  Saved results: {directory}')
    return lines


def queue_alive(root):
    owner = read_json(root/'queue-owner.json')
    if owner is None:
        return None
    try:
        process = psutil.Process(owner['pid'])
        # Reject a reused PID when observing an older saved run.
        return process.is_running() and abs(process.create_time()-owner['started_unix']) < 10
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return None


def watch(root, *, once=False, interval=2., stream=None):
    stream = sys.stdout if stream is None else stream
    prepared = read_json(root/'prepared.json')
    if prepared is None:
        raise FileNotFoundError(f'No prepared run found: {root}')
    frequencies = prepared['configuration']['frequencies_hz']
    printed = set(); previous_line = None
    interactive = stream.isatty()
    def emit(text):
        nonlocal previous_line
        if interactive and previous_line:
            stream.write('\r'+' '*len(previous_line)+'\r')
        stream.write(text+'\n'); stream.flush(); previous_line = None
    while True:
        pending = None
        for index, frequency in enumerate(frequencies, 1):
            directory = root/f'{frequency//1000}khz'/'native'
            result = read_json(directory/'result.json')
            if result is not None:
                if frequency not in printed:
                    for line in result_lines(frequency, result, directory):
                        emit(line)
                    printed.add(frequency)
                if not result.get('completed'):
                    return 1
            elif pending is None:
                pending = (index, frequency, directory.parent)
        if pending is None:
            emit(f'All {len(frequencies)} experiments completed.'); return 0
        index, frequency, trial = pending
        launcher = read_json(trial/'launcher.json')
        if launcher is not None and launcher.get('exit_code') != 0:
            emit(f'{frequency//1000} kHz failed. Inspect {trial}/worker.log'); return 1
        state = progress_state(trial/'worker.log')
        line = progress_line(frequency, index, len(frequencies), state)
        if line != previous_line:
            if interactive:
                stream.write('\r'+line.ljust(len(previous_line or ''))); stream.flush()
            else:
                stream.write(line+'\n'); stream.flush()
            previous_line = line
        if once:
            if interactive:
                stream.write('\n')
            return 0
        if queue_alive(root) is False:
            emit('Queue has stopped; completed results are preserved. Check worker/queue logs.')
            return 2
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    try:
        return watch(args.root, once=args.once)
    except KeyboardInterrupt:
        print('\nProgress display closed. Simulation left running.')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
