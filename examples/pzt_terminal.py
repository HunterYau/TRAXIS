"""Read-only terminal progress and per-experiment results for a PZT run queue.

No GUI, GPU work, or additional dependency. This can be opened or closed
independently of the simulation. --once prints a snapshot without waiting.
"""
import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
import sys
import time

import psutil

DEFAULT_ROOT = Path(__file__).resolve().parents[1]/'outputs/wandowski-pzt-plate1'


def resource_line():
    """Whole-machine readings, not memory attributed to a simulation worker."""
    memory = psutil.virtual_memory()
    line = (f'Machine: RAM {memory.percent:.1f}% '
            f'({memory.available/1024**3:.1f} GiB free)')
    executable = shutil.which('nvidia-smi')
    if not executable:
        return line + ' | GPU readings unavailable'
    try:
        result = subprocess.run([executable,
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True,
            timeout=3, check=True,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        for index, row in enumerate(result.stdout.strip().splitlines()):
            usage, used, total, temperature = row.split(',')
            line += (f' | GPU {index}: {float(usage):.0f}% busy, '
                     f'{float(used):.0f}/{float(total):.0f} MiB device memory, '
                     f'{float(temperature):.0f} C')
        return line + ' | shared GPU memory not measured'
    except (OSError, ValueError, subprocess.SubprocessError):
        return line + ' | GPU readings unavailable'


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
        state['elapsed'] = event.get('elapsed_seconds',state['elapsed'])
        if 'requested_steps' in event and state['done']==0:
            state['total']=event['requested_steps']
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
    if result.get('smoke_test'):
        status='PASSED' if result.get('completed') else 'FAILED'
        return [f'Short setup check: {status} | time {clock(result.get("wall_seconds"))}',
            '  This is not a completed experiment or a ground-truth comparison.',
            *([f'  Error: {result["error"]}'] if result.get('error') else []),
            f'  Saved check: {directory}']
    status = 'COMPLETED' if result.get('completed') else 'FAILED / INCOMPLETE'
    lines = [f'{frequency//1000} kHz: {status} | time {clock(result.get("wall_seconds"))}']
    for key, label in [('cpu_force_relative_error', 'CPU/GPU force difference'),
                       ('modified_energy_relative_drift', 'Energy drift')]:
        value = result.get(key)
        if value is not None:
            lines.append(f'  {label}: {value:.3e} (relative)')
    comparison = result.get('comparison')
    if isinstance(comparison, dict):
        if 'fast_status' in comparison:
            for label in ('fast','slow'):
                value=comparison.get(label+'_arrival_error_percent')
                status=comparison.get(label+'_status','unknown')
                value=f'{value:.2f}%' if value is not None else f'unresolved ({status})'
                lines.append(f'  {label.capitalize()} wave arrival error: {value} | lower is better')
                shape=comparison.get(label+'_envelope_shape_error_percent')
                if shape is not None:
                    trace='top+bottom' if label=='fast' else 'top-bottom'
                    lines.append(f'  {trace} trace envelope-shape difference: {shape:.2f}% | lower is better')
            lines.append('  Shape comparison normalizes amplitude; these are not overall accuracy percentages.')
        else:
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


def watch(root, *, once=False, interval=2., stream=None, resources=False, debug=False, record_stream=None):
    stream = sys.stdout if stream is None else stream
    prepared = read_json(root/'prepared.json')
    if prepared is None:
        raise FileNotFoundError(f'No prepared run found: {root}')
    frequencies = prepared['configuration']['frequencies_hz']
    printed = set(); debug_printed = set(); previous_line = None; last_resources = -math.inf
    interactive = stream.isatty()
    def emit(text):
        nonlocal previous_line
        if interactive and previous_line:
            stream.write('\r'+' '*len(previous_line)+'\r')
        stream.write(text+'\n'); stream.flush(); previous_line = None
        if record_stream is not None:
            record_stream.write(text+'\n');record_stream.flush()
    emit('Queue order: '+' -> '.join(f'{f//1000} kHz' for f in frequencies))
    if not (root/'queue-owner.json').exists():
        emit('NOT STARTED. This display does not launch simulations.')
        for reason in prepared['configuration'].get('launch_holds',[]):
            emit('Launch hold: '+reason)
    while True:
        if resources and time.monotonic()-last_resources >= 10:
            emit(resource_line())
            last_resources = time.monotonic()
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
        if debug and frequency not in debug_printed:
            emit(f'Debug log: {trial / "worker.log"}')
            debug_printed.add(frequency)
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
            if record_stream is not None:
                record_stream.write(line+'\n');record_stream.flush()
        if once:
            if interactive:
                stream.write('\n')
            return 0
        if queue_alive(root) is False:
            emit('Queue has stopped; completed results are preserved. Check worker/queue logs.')
            return 2
        time.sleep(interval)


def watch_log(log, result_path, label, total):
    """Watch a bounded setup check without presenting it as a full experiment."""
    print(label, flush=True)
    print(f'Debug log: {log}', flush=True)
    previous=None; last_resources=-math.inf
    while True:
        if time.monotonic()-last_resources>=10:
            print(resource_line(),flush=True);last_resources=time.monotonic()
        result=read_json(result_path)
        if result is not None:
            status='PASSED' if result.get('completed') else 'FAILED'
            print(f'{label}: {status} | elapsed {clock(result.get("wall_seconds"))}',flush=True)
            if result.get('error'):print(result['error'],flush=True)
            print('Full frequency queue has NOT started. This was only the setup check.',flush=True)
            return 0 if result.get('completed') else 1
        state=progress_state(log)
        if state['done']==0:state['total']=total
        fraction=state['done']/state['total'];filled=int(20*fraction)
        line=(f'[{"#"*filled}{"-"*(20-filled)}] {fraction:.1%} | '
            f'{state["done"]}/{state["total"]} steps | elapsed {clock(state["elapsed"])} | {state["phase"]}')
        if line!=previous:print(line,flush=True);previous=line
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--resources', action='store_true', help='show whole-machine RAM/GPU readings')
    parser.add_argument('--debug', action='store_true', help='show current worker log location')
    parser.add_argument('--log', type=Path, help='watch a setup-check log instead of the frequency queue')
    parser.add_argument('--result', type=Path)
    parser.add_argument('--label', default='Setup check (not a full experiment)')
    parser.add_argument('--total', type=int, default=128)
    parser.add_argument('--transcript',type=Path,help='append displayed progress/results to a plain-text file')
    args = parser.parse_args()
    try:
        if args.log:
            if args.result is None or args.total<=0:parser.error('--log requires --result and a positive --total')
            return watch_log(args.log,args.result,args.label,args.total)
        if args.transcript:
            args.transcript.parent.mkdir(parents=True,exist_ok=True)
            with args.transcript.open('a',encoding='utf-8') as stream:
                return watch(args.root,once=args.once,resources=args.resources,debug=args.debug,record_stream=stream)
        return watch(args.root, once=args.once, resources=args.resources, debug=args.debug)
    except KeyboardInterrupt:
        print('\nProgress display closed. Simulation left running.')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
