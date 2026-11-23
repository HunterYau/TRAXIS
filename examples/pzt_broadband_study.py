"""Prepare or explicitly run matched direct/broadband PZT studies. No autostart."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
import psutil

from examples import resolved_pzt_queue as queue
from examples.embedded_wave import ThermalPause
from examples.experimental_five import PROJECT, PeakGuard, record
from examples.plate_outputs import queue_lock, archive_attempt
from examples.pzt_analytic_source import analytic_source
from examples.pzt_broadband import (interior_impulse, reconstruct, paired_metrics,
                                    within_target, common_grid)
from examples.pzt_convergence import cases as refinement_cases, worker_alive
from examples.pzt_sampling import antialiased_samples
from examples.pzt_terminal import result_lines
from traxis.gpu_enriched_verlet import gpu_explicit_simulate
from traxis.resources import ResourceLimits
from traxis.storage import write_json, atomic_write, sha256_file

FREQUENCIES = [50000, 100000, 150000, 200000, 250000]
REFINEMENT_FREQUENCIES = [100000, 250000]
DURATION_S = .00035
STUDY_ROOT = Path('outputs/simulation-runs')
PAIRS = [('temporal first', 'baseline', 'time-2'),
         ('temporal second', 'time-2', 'time-4'),
         ('spatial coarse', 'plate-200', 'plate-230'),
         ('spatial fine', 'plate-230', 'plate-260'),
         ('production spatial', 'time-2', 'plate-260'),
         ('plate thickness', 'plate-thickness-1', 'plate-thickness-2'),
         *[('PZT ' + name, 'time-2', 'body-' + name) for name in ('radial', 'angular', 'thickness')],
         ('coupled', 'baseline', 'coupled-fine')]


def identity():
    proof = queue.preflight()
    paths = ('examples/pzt_broadband.py', 'examples/pzt_broadband_study.py',
             'examples/pzt_analytic_source.py', 'examples/pzt_convergence.py')
    return dict(production_preflight=proof,
                study_sources={name: sha256_file(PROJECT / name) for name in paths})


def source(frequency, steps):
    row = record(frequency)
    drive = np.asarray(row['nifgen_avg_waveform_ch1'])
    return analytic_source(frequency, 5, np.arange(len(drive)) / row['sampleRate'], drive)


def study_design(suite):
    if suite not in ('benchmark', 'convergence'):
        raise ValueError('Unknown study suite')
    resolutions = refinement_cases() if suite == 'convergence' else refinement_cases()[:1]
    tasks = []
    for index, case in enumerate(resolutions):
        frequencies = FREQUENCIES if case['name'] == 'baseline' else REFINEMENT_FREQUENCIES
        # Alternate order between resolutions to expose warm-up/order effects.
        methods = ['direct', 'broadband'] if index % 2 == 0 else ['broadband', 'direct']
        for method in methods:
            for frequency in frequencies if method == 'direct' else [None]:
                tasks.append(dict(id=case['name'] + '/' + method +
                                  (f'/{frequency // 1000}khz' if frequency else ''),
                                  case=case['name'], method=method, frequency_hz=frequency))
    return resolutions, tasks


def prepare(root, suite):
    if root.exists():
        raise FileExistsError('Choose a new study root; --run resumes a prepared study')
    config = queue.configuration()
    if config.get('setup_profile') or config.get('damping_profile') or config['body']['bond'] != 'perfect':
        raise ValueError('This first study supports only the unchanged, linear, ideal-open, lossless configuration')
    resolutions, tasks = study_design(suite)
    plan = dict(schema=1, suite=suite, configuration=config, cases=resolutions, tasks=tasks,
                identity=identity(), frequencies_hz=FREQUENCIES,
                refinement_frequencies_hz=REFINEMENT_FREQUENCIES, duration_s=DURATION_S,
                output_stride=1, precision='float64', source='verified analytic five-cycle Hann sine',
                target_percent=1., sources={str(f): source(f, 65536)[1] for f in FREQUENCIES},
                source_or_response_fitted=False, simulation_started=False,
                timing_policy='Separate process per task; record build, solver, reconstruction, comparison, and total times. '
                    'Total five-frequency speedup is measured only where all five direct runs exist. '
                    'Single execution per task; no repeated timing-confidence claim.',
                limitations=['Impulse is a numerical identification input, not the experimental excitation.',
                    'Current benchmark and convergence results have not been computed.',
                    'Convergence studies check 100 and 250 kHz; all five frequencies are checked at baseline.',
                    'Method agreement does not establish accuracy against real life.',
                    'Each geometry/material/mesh/time step and driven actuator needs its own kernel.',
                    'A0/S0-related sums and differences are not proven pure separated modes.'])
    root.mkdir(parents=True)
    write_json(root / 'plan.json', plan)
    with zipfile.ZipFile(root / 'frozen-source.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        names = {**plan['identity']['production_preflight']['source_sha256'],
                 **plan['identity']['study_sources']}
        for name in names:
            archive.write(PROJECT / name, name)
        archive.write(root / 'plan.json', 'plan.json')
    write_json(root / 'status.json', dict(status='prepared', simulation_running=False))
    assess(root)
    print(f'PREPARED ONLY: {suite}, {len(tasks)} solver runs. No simulation started. {root}', flush=True)


def load_plan(root):
    plan = json.loads((root / 'plan.json').read_text())
    if plan['identity'] != identity():
        raise ValueError('Prepared scientific inputs or study sources changed')
    if (plan['configuration'] != plan['identity']['production_preflight']['configuration']
            or plan['duration_s'] != DURATION_S or plan['output_stride'] != 1
            or plan['precision'] != 'float64'):
        raise ValueError('Prepared physics or recording settings were altered')
    resolutions, tasks = study_design(plan['suite'])
    if plan['cases'] != resolutions or plan['tasks'] != tasks:
        raise ValueError('Prepared resolution schedule was altered')
    # Task paths become directories and subprocess arguments; allow only our plan.
    if any(Path(t['id']).is_absolute() or '..' in Path(t['id']).parts or '\\' in t['id']
           for t in plan['tasks']):
        raise ValueError('Invalid task path')
    return plan


def task_receipt(folder, plan_sha256):
    path = folder / 'result.json'
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    if not receipt.get('completed'):
        return None
    if receipt['plan_sha256'] != plan_sha256:
        raise ValueError('Result belongs to a different prepared study')
    launcher = folder / 'launcher.json'
    if launcher.exists():
        launch = json.loads(launcher.read_text())
        if launch['plan_sha256'] != plan_sha256:
            raise ValueError('Launcher belongs to a different study')
        if launch['exit_code'] != 0:
            return None
        receipt['process_wall_seconds'] = launch['process_wall_seconds']
    expected = ({'waveforms.npz', 'experimental-comparison.json', 'experimental-results.txt'}
                if receipt['task']['method'] == 'direct' else
                {'impulse.npz', *[f'{f // 1000}khz/{name}' for f in FREQUENCIES
                    for name in ('waveforms.npz', 'experimental-comparison.json',
                                 'experimental-results.txt', 'reconstruction.json')]})
    if set(receipt['artifacts_sha256']) != expected:
        raise ValueError('Incomplete result artifact list')
    for name, digest in receipt['artifacts_sha256'].items():
        path = (folder / name).resolve()
        path.relative_to(folder.resolve())
        if sha256_file(path) != digest:
            raise ValueError('Saved artifact changed: ' + str(path))
    return receipt


def save_trace(folder, time_s, voltage, frequency, config, **extra):
    folder.mkdir(parents=True, exist_ok=True)
    atomic_write(folder / 'waveforms.npz', lambda stream: np.savez_compressed(stream,
                  time_s=time_s, voltage_v=voltage, **extra))
    start = time.perf_counter()
    row = record(frequency)
    measured = np.asarray(row['niscope_avg_waveform'])[:, [2, 3]]
    # Retain the original queue's reporting cadence for its experimental scores.
    # Raw method agreement always uses the full saved internal-grid traces.
    stride = (len(time_s) - 1) // 8192
    if stride < 1 or (len(time_s) - 1) % 8192:
        raise ValueError('Expected nested original recording cadence')
    comparison = queue.compare(time_s[::stride], voltage[::stride], measured,
                               frequency, row['sampleRate'], config=config)
    comparison['study_reporting'] = dict(cadence='Original 42.724609375 ns point-sampled cadence',
        raw_internal_grid_retained=True, unchanged_queue_comparison=True)
    write_json(folder / 'experimental-comparison.json', comparison)
    lines = [str(folder.relative_to(folder.parents[2])) + ' | comparison with measured plate',
             *result_lines(frequency, dict(completed=True, comparison=comparison), folder)[1:]]
    text = '\n'.join(lines) + '\n'
    atomic_write(folder / 'experimental-results.txt', lambda stream: stream.write(text.encode('utf-8')))
    print(text, end='', flush=True)
    return time.perf_counter() - start


def worker(root, task_id):
    plan = load_plan(root)
    task = next(t for t in plan['tasks'] if t['id'] == task_id)
    case = next(c for c in plan['cases'] if c['name'] == task['case'])
    folder = root / task_id
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists():
        raise FileExistsError('Scheduler must archive an interrupted attempt before retry')
    os.environ.setdefault('CUPY_CACHE_DIR', str(root / 'cache'))
    started = time.perf_counter()
    result = dict(completed=False, task=task, case=case, artifacts_sha256={},
                  plan_sha256=sha256_file(root / 'plan.json'))
    last = -math.inf

    def event(phase, **details):
        elapsed = time.perf_counter() - started
        state = dict(status='running', task=task_id, phase=phase, elapsed_seconds=elapsed, **details)
        if phase == 'propagating':
            done, total = details['steps_completed'], details['requested_steps']
            fraction = done / total
            state.update(percent=100 * fraction, eta_seconds=elapsed * (1 / fraction - 1))
            print(f'{task_id} | {100 * fraction:6.2f}% | {done:,}/{total:,} steps | '
                  f'elapsed {elapsed / 60:.1f} min | ETA {state["eta_seconds"] / 60:.1f} min', flush=True)
        else:
            print(f'{task_id} | {phase} | elapsed {elapsed:.1f} s', flush=True)
        write_json(folder / 'status.json', state, overwrite=True)

    try:
        config = copy.deepcopy(plan['configuration'])
        config['mesh']['cells'] = [case['cells'], case['cells'], case.get('nz', 1)]
        config['body'].update(zip(('sectors', 'rings', 'layers'), case['body']))
        config['mesh']['minimum_steps'] = case['steps']
        guard = PeakGuard(ResourceLimits(**config['resource_limits']), telemetry_errors_fatal=False)
        guard.check(force=True)
        cooling = ThermalPause(guard, config['cooling']['upper_c'], config['cooling']['lower_c'], allow_missing=True)
        build_start = time.perf_counter()
        system, load, probes, required = queue.build(config, guard, event)
        result['build_seconds'] = time.perf_counter() - build_start
        if case['steps'] < required:
            raise ValueError('Planned common time step is not stable; do not silently change one method')
        dt = DURATION_S / case['steps']
        force = interior_impulse(dt) if task['method'] == 'broadband' else source(task['frequency_hz'], case['steps'])[0]

        def progress(state):
            nonlocal last
            cooling.check(state['steps_completed'])
            if time.perf_counter() - last >= 15 or state['steps_completed'] == case['steps']:
                event('propagating', **state)
                last = time.perf_counter()

        solver_start = time.perf_counter()
        simulation = gpu_explicit_simulate(system, case['steps'], dt, probes, load,
            force=force, output_stride=1, energy_stride=256, guard=guard,
            progress=progress, max_workspace_mib=4096)
        result['solver_seconds'] = time.perf_counter() - solver_start
        end = dt if task['method'] == 'broadband' else 5 / task['frequency_hz']
        tail = simulation.modified_energy_j[simulation.energy_time_s > end + 2 * dt]
        drift = float(np.ptp(tail) / abs(tail[0])) if len(tail) > 1 and tail[0] != 0 else None
        if drift is None or drift > 1e-8:
            raise ArithmeticError('Unforced lossless energy check failed')
        result.update(energy_relative_drift=drift, stable_dt_s=system.stable_dt_s,
                      statistics=simulation.statistics, resource_peaks=guard.peaks,
                      body_reports=[b.report for b in system.body_attachments],
                      internal_dt_s=dt, configuration=config, reconstruction_seconds=0., comparison_seconds=0.)
        if task['method'] == 'direct':
            result['comparison_seconds'] = save_trace(folder, simulation.time_s, simulation.displacement_m,
                task['frequency_hz'], config, energy_time_s=simulation.energy_time_s,
                modified_energy_j=simulation.modified_energy_j)
            for name in ('waveforms.npz', 'experimental-comparison.json', 'experimental-results.txt'):
                result['artifacts_sha256'][name] = sha256_file(folder / name)
        else:
            atomic_write(folder / 'impulse.npz', lambda stream: np.savez_compressed(stream,
                time_s=simulation.time_s, voltage_v=simulation.displacement_m,
                energy_time_s=simulation.energy_time_s, modified_energy_j=simulation.modified_energy_j))
            result['artifacts_sha256']['impulse.npz'] = sha256_file(folder / 'impulse.npz')
            for frequency in FREQUENCIES:
                event('reconstructing', frequency_hz=frequency)
                tick = time.perf_counter()
                target, metadata = source(frequency, case['steps'])
                voltage = reconstruct(simulation.displacement_m, target(simulation.time_s))
                result['reconstruction_seconds'] += time.perf_counter() - tick
                child = folder / f'{frequency // 1000}khz'
                result['comparison_seconds'] += save_trace(child, simulation.time_s, voltage, frequency, config)
                write_json(child / 'reconstruction.json', dict(source=metadata,
                    method='causal discrete impulse convolution; internal grid retained',
                    impulse_step=1, impulse_amplitude_v=1., time_or_gain_fit=False,
                    frequency_cutoff_hz=None, deconvolution_or_regularization=False))
                for name in ('waveforms.npz', 'experimental-comparison.json', 'experimental-results.txt', 'reconstruction.json'):
                    key = f'{frequency // 1000}khz/{name}'
                    result['artifacts_sha256'][key] = sha256_file(child / name)
        result['completed'] = True
        event('completed')
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['total_seconds'] = time.perf_counter() - started
        write_json(folder / 'result.json', result)
        write_json(folder / 'status.json', dict(status='completed' if result['completed'] else 'interrupted',
                   error=result.get('error'), elapsed_seconds=result['total_seconds']), overwrite=True)


def assess(root):
    plan = json.loads((root / 'plan.json').read_text())
    digest = sha256_file(root / 'plan.json')
    receipts = {t['id']: task_receipt(root / t['id'], digest) for t in plan['tasks']}
    cache = {}

    def trace(case, method, frequency):
        task_id = f'{case}/{method}' + (f'/{frequency // 1000}khz' if method == 'direct' else '')
        if not receipts.get(task_id):
            return None
        path = root / case / method / f'{frequency // 1000}khz' / 'waveforms.npz'
        if path not in cache:
            with np.load(path) as data:
                cache[path] = (data['time_s'].copy(), data['voltage_v'].copy())
        return cache[path]

    agreement, convergence, timings, temporal_order = [], [], [], []
    for case in plan['cases']:
        frequencies = FREQUENCIES if case['name'] == 'baseline' else REFINEMENT_FREQUENCIES
        for frequency in frequencies:
            direct = trace(case['name'], 'direct', frequency)
            broad = trace(case['name'], 'broadband', frequency)
            if direct is None or broad is None:
                continue
            a, b = common_grid(*direct, *broad)
            changes = paired_metrics(a, b)
            agreement.append(dict(case=case['name'], frequency_hz=frequency, channels=changes,
                                  within_target=within_target(changes)))
        case_tasks = [t for t in plan['tasks'] if t['case'] == case['name']]
        if all(receipts[t['id']] for t in case_tasks):
            direct = [receipts[t['id']] for t in case_tasks if t['method'] == 'direct']
            broad = next(receipts[t['id']] for t in case_tasks if t['method'] == 'broadband')
            timed = [*direct, broad]
            complete_process_timing = all('process_wall_seconds' in r for r in timed)
            timing_key = 'process_wall_seconds' if complete_process_timing else 'total_seconds'
            direct_total = sum(r[timing_key] for r in direct)
            broad_total = broad[timing_key]
            timings.append(dict(case=case['name'], direct_frequencies_hz=frequencies,
                broadband_reconstructed_frequencies_hz=FREQUENCIES,
                direct_total_seconds=direct_total, broadband_total_seconds=broad_total,
                timing_basis='whole worker process' if complete_process_timing else 'worker function only',
                total_speedup=direct_total / broad_total if broad_total else None,
                five_frequency_benchmark=(len(direct) == 5),
                direct_solver_seconds=sum(r['solver_seconds'] for r in direct),
                broadband_solver_seconds=broad['solver_seconds'],
                broadband_reconstruction_seconds=broad['reconstruction_seconds']))
    for label, first, second in PAIRS:
        if not {first, second}.issubset({c['name'] for c in plan['cases']}):
            continue
        for frequency in REFINEMENT_FREQUENCIES:
            for method in ('direct', 'broadband'):
                a, b = trace(first, method, frequency), trace(second, method, frequency)
                if a is None or b is None:
                    continue
                coarse, fine = common_grid(*a, *b)
                changes = paired_metrics(fine, coarse)  # fine solution is denominator
                grid = np.arange(1750) / 5000000
                # Match the existing output cadence before the separately
                # labelled existing 5 MHz anti-alias comparison.
                sa, sb = (len(a[0]) - 1) // 8192, (len(b[0]) - 1) // 8192
                sampled = paired_metrics(antialiased_samples(b[0][::sb], b[1][::sb], grid),
                                         antialiased_samples(a[0][::sa], a[1][::sa], grid))
                convergence.append(dict(study=label, first=first, second=second, method=method,
                    frequency_hz=frequency, channels=changes, within_target=within_target(changes),
                    anti_aliased_5mhz_channels=sampled))
    for frequency in REFINEMENT_FREQUENCIES:
        for method in ('direct', 'broadband'):
            levels = [trace(case, method, frequency) for case in ('baseline', 'time-2', 'time-4')]
            if any(level is None for level in levels):
                continue
            a, b = common_grid(*levels[0], *levels[1])
            c, d = common_grid(*levels[1], *levels[2])
            first = paired_metrics(b, a)
            second = paired_metrics(d, c)
            rows = {}
            for channel in first:
                # RMS accounts for the different numbers of samples at each dt.
                e1, e2 = first[channel]['rmse_v'], second[channel]['rmse_v']
                ratio = e1 / e2 if min(e1, e2) > 0 else None
                rows[channel] = dict(first_rms_difference_v=e1, second_rms_difference_v=e2,
                    difference_ratio=ratio, observed_order=math.log2(ratio) if ratio else None)
            temporal_order.append(dict(method=method, frequency_hz=frequency, channels=rows))
    complete = all(receipts.values())
    agreement_pass = complete and bool(agreement) and all(r['within_target'] for r in agreement)
    # A small finest-pair change alone is not an absolute error bound. Keep all
    # coarse and fine comparisons visible; no automatic acceptance/mesh edits.
    answer = dict(completed=complete, suite=plan['suite'], tasks_completed=sum(r is not None for r in receipts.values()),
        tasks_total=len(receipts), method_agreement=agreement, convergence=convergence,
        temporal_order=temporal_order, timings=timings,
        all_method_agreements_within_target=agreement_pass,
        convergence_evaluated=complete and plan['suite'] == 'convergence' and bool(convergence),
        ready_for_training_data=False, target_percent=plan['target_percent'], limitations=plan['limitations'])
    write_json(root / 'findings.json', answer, overwrite=True)
    experimental_text = ['Collective comparisons with the measured plate; raw and normalized scores are labelled.']
    for task in plan['tasks']:
        if not receipts[task['id']]:
            continue
        folder = root / task['id']
        sources = ([folder / 'experimental-results.txt'] if task['method'] == 'direct' else
                   [folder / f'{f // 1000}khz/experimental-results.txt' for f in FREQUENCIES])
        for path in sources:
            experimental_text.append(path.read_text(encoding='utf-8'))
    atomic_write(root / 'experimental-results.txt',
        lambda stream: stream.write(('\n'.join(experimental_text) + '\n').encode('utf-8')), overwrite=True)
    lines = [f'Direct versus broadband | {plan["suite"]} | {answer["tasks_completed"]}/{len(receipts)} runs completed',
             'Raw voltage difference: lower is better; target <1% in every channel. No gain/time/polarity fit.']
    for row in agreement:
        values = '; '.join(f'{k} {v["raw_l2_change_percent"]:.6f}%' if v['raw_l2_change_percent'] is not None
                           else k + ' undefined' for k, v in row['channels'].items())
        lines.append(f'{row["case"]} | {row["frequency_hz"] // 1000} kHz | method difference | {values}')
    for row in convergence:
        values = '; '.join(f'{k} {v["raw_l2_change_percent"]:.6f}%' if v['raw_l2_change_percent'] is not None
                           else k + ' undefined' for k, v in row['channels'].items())
        lines.append(f'{row["method"]} | {row["frequency_hz"] // 1000} kHz | {row["study"]} | {values}')
    for row in timings:
        lines.append(f'{row["case"]} | speedup {row["total_speedup"]:.3f}x | '
                     f'{len(row["direct_frequencies_hz"])} direct frequencies versus 5 reconstructed | '
                     f'direct {row["direct_total_seconds"]:.1f}s | broadband {row["broadband_total_seconds"]:.1f}s')
    lines.append('Convergence and full-plate speedup remain unverified until those simulations finish. '
                 'No automatic adoption or training-data approval.')
    atomic_write(root / 'findings.txt', lambda stream: stream.write(('\n'.join(lines) + '\n').encode()), overwrite=True)
    return answer


def run(root):
    plan = load_plan(root)
    with queue_lock(root):
        write_json(root / 'status.json', dict(status='running', simulation_running=True), overwrite=True)
        try:
            for index, task in enumerate(plan['tasks'], 1):
                folder = root / task['id']
                if worker_alive(folder):
                    raise RuntimeError('Worker still running: ' + task['id'])
                if task_receipt(folder, sha256_file(root / 'plan.json')):
                    continue
                previous = folder / 'result.json'
                if previous.exists():
                    error = json.loads(previous.read_text()).get('error', '')
                    if error and not error.startswith(('KeyboardInterrupt', 'SystemExit')):
                        raise RuntimeError('Investigate failed task before retry: ' + error)
                archive_attempt(folder)
                print(f'EXPERIMENT {index}/{len(plan["tasks"])} | {task["id"]}', flush=True)
                write_json(root / 'status.json', dict(status='running', simulation_running=True,
                    task=task['id'], experiment=index, experiments_total=len(plan['tasks'])), overwrite=True)
                with (folder / 'worker.log').open('a', encoding='utf-8') as log:
                    launch_start = time.perf_counter()
                    process = subprocess.Popen([sys.executable, '-u', '-m', 'examples.pzt_broadband_study',
                        '--root', str(root), '--worker', task['id']], cwd=PROJECT, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                    try:
                        created = psutil.Process(process.pid).create_time()
                    except psutil.NoSuchProcess:
                        created = 0.
                    write_json(folder / 'owner.json', dict(pid=process.pid, started_unix=created), overwrite=True)
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end='', flush=True)
                    code = process.wait()
                write_json(folder / 'launcher.json', dict(exit_code=code,
                    process_wall_seconds=time.perf_counter() - launch_start,
                    plan_sha256=sha256_file(root / 'plan.json')))
                assess(root)
                print((root / 'findings.txt').read_text(), flush=True)
                if code:
                    raise RuntimeError(f'Task failed (exit {code}): {task["id"]}')
            write_json(root / 'status.json', dict(status='completed', simulation_running=False), overwrite=True)
        except BaseException as exc:
            live = any(worker_alive(root / task['id']) for task in plan['tasks'])
            write_json(root / 'status.json', dict(status='interrupted', error=str(exc), simulation_running=live), overwrite=True)
            raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--suite', choices=('benchmark', 'convergence'), default='benchmark')
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--run', action='store_true')
    actions.add_argument('--assess', action='store_true')
    actions.add_argument('--worker')
    args = parser.parse_args()
    root = args.root.resolve()
    root.relative_to((PROJECT / STUDY_ROOT).resolve())
    if args.prepare:
        prepare(root, args.suite)
    elif args.run:
        run(root)
    elif args.worker:
        worker(root, args.worker)
    else:
        assess(root)
        print((root / 'findings.txt').read_text())


if __name__ == '__main__':
    main()
