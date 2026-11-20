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

from traxis.storage import atomic_write
from examples.plate_outputs import CURRENT_RUN, result_directory, relocated, owner_alive, saved_trial_status

DEFAULT_ROOT = CURRENT_RUN


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
    damping=result.get('damping')
    if damping:
        growth=result.get('post_drive_energy_max_relative_growth')
        if growth is not None:
            expectation='remain constant' if damping.get('model')=='none' else 'decrease'
            lines.append(f'  Post-pulse energy growth: {growth:.3e} (energy is expected to {expectation})')
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
                mode='S0' if label=='fast' else 'A0'
                lines.append(f'  {mode} wave arrival error: {value} | lower is better')
                method=comparison.get(label+'_arrival_method')
                if method:
                    lines.append(f'  Arrival search: {method}')
                shape=comparison.get(label+'_envelope_shape_error_percent')
                if shape is not None:
                    trace='top+bottom' if label=='fast' else 'top-bottom'
                    lines.append(f'  {trace} trace envelope-shape difference: {shape:.2f}% | lower is better')
                if label+'_envelope_pearson_r' in comparison:
                    pearson=comparison[label+'_envelope_pearson_r']
                    text=f'{pearson:.4f}' if pearson is not None else 'undefined (constant or zero envelope)'
                    lines.append(f'  {mode}-related trace envelope Pearson r: {text} | +1 best')
                wiggles=comparison.get(label+'_waveform')
                if wiggles:
                    difference=wiggles['normalized_wiggle_difference_percent']
                    text=f'{difference:.2f}%' if difference is not None else 'undefined (zero signal)'
                    correlation=wiggles['correlation']
                    correlation=f'{correlation:.4f}' if correlation is not None else 'undefined'
                    lines.append(f'  {mode}-related trace individual-wiggle difference: {text} | lower better, strength normalized')
                    lines.append(f'  Signed wiggle correlation: {correlation} | +1 best, -1 opposite polarity')
            for channel,metrics in comparison.get('waveform_channels',{}).items():
                raw=metrics['raw_voltage_difference_percent']
                text=f'{raw:.2f}%' if raw is not None else 'undefined (zero measured signal)'
                lines.append(f'  {channel} receiver raw voltage difference: {text}; RMSE {metrics["rmse_v"]:.6g} V')
            if comparison.get('waveform_channels'):
                lines.append('  Raw voltage differences include unmatched drive/load. Wiggles use the full shared interval without fitted alignment.')
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
    return owner_alive(owner)


def export_queue_results(root, frequencies=None):
    """Save a results-only snapshot, including partial and failed queues.

    Rebuild from saved records so exporting again never duplicates experiments.
    This does not launch workers or change their comparison results.
    """
    root = Path(root)
    root = relocated(root)
    if frequencies is None:
        prepared = read_json(root/'prepared.json')
        if prepared is None:
            raise FileNotFoundError(f'No prepared run found: {root}')
        frequencies = prepared['configuration']['frequencies_hz']
    lines = ['Collective experiment results',
             'Queue order: '+' -> '.join(f'{f//1000} kHz' for f in frequencies), '']
    completed = failed = 0
    elapsed = 0.
    for frequency in frequencies:
        summary_start = len(lines)
        trial = root/f'{frequency//1000}khz'
        directory = result_directory(trial)
        result = read_json(directory/'result.json')
        if result is None:
            launcher = read_json(trial/'launcher.json')
            status = saved_trial_status(root,trial,refresh=True)
            if status.get('status') in ('interrupted', 'failed'):
                failed += 1
                lines.append(f'{frequency//1000} kHz: {status["status"].upper()} / UNFINISHED')
                if status.get('reason'):
                    lines.append('  '+status['reason'])
            elif launcher is not None and launcher.get('exit_code') != 0:
                failed += 1
                lines.append(f'{frequency//1000} kHz: FAILED | worker exit code {launcher.get("exit_code")}')
                lines.append(f'  Debug log: {trial / "worker.log"}')
            else:
                lines.append(f'{frequency//1000} kHz: No saved result yet.')
        else:
            completed += bool(result.get('completed'))
            failed += not bool(result.get('completed'))
            elapsed += result.get('wall_seconds') or 0.
            lines.extend(result_lines(frequency, result, directory))
        summary_text = '\n'.join(lines[summary_start:])+'\n'
        atomic_write(root/'summaries'/f'{frequency//1000}khz.txt',
                     lambda f: f.write(summary_text.encode('utf-8')), overwrite=True)
        lines.append('')
    lines.extend([f'Completed: {completed}/{len(frequencies)} | failed/incomplete: {failed} '
                  f'| awaiting results: {len(frequencies)-completed-failed}',
                  f'Total recorded experiment time: {clock(elapsed)}'])
    path = root/'all-results.txt'
    payload = ('\n'.join(lines)+'\n').encode('utf-8')
    atomic_write(path, lambda stream: stream.write(payload), overwrite=True)
    return path


def watch(root, *, once=False, interval=2., stream=None, resources=False, debug=False, record_stream=None,
          follow_retries=False):
    stream = sys.stdout if stream is None else stream
    root = relocated(root)
    prepared = read_json(root/'prepared.json')
    if prepared is None:
        raise FileNotFoundError(f'No prepared run found: {root}')
    frequencies = prepared['configuration']['frequencies_hz']
    first_experiment = prepared['configuration'].get('start_experiment',1)
    total_experiments = first_experiment+len(frequencies)-1
    printed = set(); debug_printed = set(); previous_line = None; last_resources = -math.inf
    stopped_printed = False
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
        for index, frequency in enumerate(frequencies, first_experiment):
            trial = root/f'{frequency//1000}khz'
            directory = result_directory(trial)
            result = read_json(directory/'result.json')
            if result is not None:
                signature=(frequency,result.get('completed'),result.get('wall_seconds'),result.get('error'))
                if signature not in printed:
                    for line in result_lines(frequency, result, directory):
                        emit(line)
                    printed.add(signature)
                if not result.get('completed'):
                    if not follow_retries:
                        return 1
                    if pending is None:
                        pending = (index, frequency, trial)
            elif pending is None:
                pending = (index, frequency, trial)
        if pending is None:
            emit(f'All {len(frequencies)} experiments completed.'); return 0
        index, frequency, trial = pending
        status = saved_trial_status(root,trial)
        if status.get('status') in ('interrupted', 'failed'):
            if not follow_retries:
                emit(f'{frequency//1000} kHz: {status["status"].upper()} / UNFINISHED. Restart the scheduler to retry.')
                return 2
        if debug and frequency not in debug_printed:
            emit(f'Debug log: {trial / "worker.log"}')
            debug_printed.add(frequency)
        launcher = read_json(trial/'launcher.json')
        if launcher is not None and launcher.get('exit_code') != 0:
            if not follow_retries:
                emit(f'{frequency//1000} kHz failed. Inspect {trial}/worker.log'); return 1
        state = progress_state(trial/'worker.log')
        if status.get('status') in ('interrupted','failed'):
            state['phase']=status['status'].upper()+' / UNFINISHED; waiting for retry'
        line = progress_line(frequency, index, total_experiments, state)
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
            if not stopped_printed:
                emit('Queue has stopped; completed results are preserved. Check worker/queue logs.')
                stopped_printed = True
            if not follow_retries:
                return 2
        else:
            stopped_printed = False
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
    parser.add_argument('--export-results',action='store_true',
        help='write all-results.txt from saved results and exit without watching or running simulations')
    parser.add_argument('--follow-retries',action='store_true',help='keep the display open across stopped queues and retries')
    args = parser.parse_args()
    try:
        if args.export_results:
            print(f'Saved collective results: {export_queue_results(args.root)}')
            return 0
        if args.log:
            if args.result is None or args.total<=0:parser.error('--log requires --result and a positive --total')
            return watch_log(args.log,args.result,args.label,args.total)
        if args.transcript:
            args.transcript.parent.mkdir(parents=True,exist_ok=True)
            with args.transcript.open('a',encoding='utf-8') as stream:
                return watch(args.root,once=args.once,resources=args.resources,debug=args.debug,record_stream=stream,
                             follow_retries=args.follow_retries)
        return watch(args.root, once=args.once, resources=args.resources, debug=args.debug,follow_retries=args.follow_retries)
    except KeyboardInterrupt:
        print('\nProgress display closed. Simulation left running.')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
