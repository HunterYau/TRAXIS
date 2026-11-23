"""Prepare-only by default intent: explicit --run is required for MOR evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile

import numpy as np
import psutil

from examples import pzt_broadband_study as broadband
from examples import resolved_pzt_queue as queue
from examples.experimental_five import PROJECT
from examples.plate_outputs import archive_attempt, queue_lock
from examples.pzt_convergence import worker_alive
from examples.pzt_analytic_source import analytic_source
from examples.pzt_broadband import reconstruct, paired_metrics, within_target
from examples.pzt_era import kernel_data, reduced_impulse
from examples.pzt_era_matrix_free import (fit_era, memory_estimate, SVD_TOLERANCE,
                                         RESIDUAL_TOLERANCE, BLOCK, SEED)
from traxis.storage import write_json, atomic_write, sha256_file

REFERENCE = Path('outputs/simulation-runs/runpod-h200-nvl-benchmark-001')
STUDY = Path('outputs/simulation-runs/runpod-mor-comparison-matrix-free-v3')
FREQUENCIES = [50000, 100000, 150000, 200000, 250000, 300000]
ORDERS = [32, 64, 128, 256, 512, 1024, 2048]
DT = .00035 / 65536
TARGET = .1
BUILD_BACKEND = 'matrix_free_fft_propack_float64'
BUILD_OPTIONS = dict(svd_tolerance=SVD_TOLERANCE, residual_tolerance=RESIDUAL_TOLERANCE,
                     block_columns=BLOCK, seed=SEED, fft_backend='cupy', krylov_limit='min(m+1,n+1,2*k+32)')


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(PROJECT).as_posix()


def sources():
    names = ('examples/pzt_mor_study.py', 'examples/pzt_era.py', 'examples/pzt_era_cuda.py',
             'examples/pzt_era_matrix_free.py', 'examples/pzt_broadband.py',
             'examples/pzt_broadband_study.py', 'examples/pzt_analytic_source.py',
             'examples/pzt_convergence.py')
    return {name: sha256_file(PROJECT / name) for name in names}


def reference_identity():
    root = PROJECT / REFERENCE
    manifest = json.loads((root / 'download-manifest.json').read_text())
    for name, digest in manifest.items():
        path = (root / name).resolve()
        path.relative_to(root)
        if sha256_file(path) != digest:
            raise ValueError('Saved H200 result changed: ' + name)
    saved = json.loads((root / 'plan.json').read_text())
    if saved['identity'] != broadband.identity():
        raise ValueError('Saved full-order physics/source differs from the current baseline')
    if saved['cases'][0] != dict(name='baseline', cells=250, body=[32, 6, 3], steps=65536):
        raise ValueError('Wrong baseline mesh or time-step count')
    receipt = broadband.task_receipt(root / 'baseline/broadband', sha256_file(root / 'plan.json'))
    if not receipt:
        raise ValueError('Saved full-order broadband run is incomplete')
    for frequency in FREQUENCIES[:-1]:
        if not broadband.task_receipt(root / f'baseline/direct/{frequency // 1000}khz', sha256_file(root / 'plan.json')):
            raise ValueError('Missing completed independent direct reference')
    time_s, voltage = saved_kernel()
    _, dt = kernel_data(time_s, voltage)
    if len(time_s) != 65537 or dt != DT:
        raise ValueError('This study requires the saved 5.340576171875 ns internal grid')
    return dict(reference_directory=REFERENCE.as_posix(), manifest=manifest,
        manifest_sha256=sha256_file(root / 'download-manifest.json'),
        full_order_plan_sha256=sha256_file(root / 'plan.json'), internal_dt_s=dt,
        duration_s=float(time_s[-1]), samples=len(time_s), original_broadband_seconds=receipt['total_seconds'],
        original_broadband_solver_seconds=receipt['solver_seconds'],
        original_five_reconstruction_seconds=receipt['reconstruction_seconds'])


def saved_kernel():
    with np.load(PROJECT / REFERENCE / 'baseline/broadband/impulse.npz') as data:
        return data['time_s'].copy(), data['voltage_v'].copy()


def target_source(frequency):
    if frequency not in FREQUENCIES:
        raise ValueError('Unsupported comparison frequency')
    if frequency != 300000:
        pulse, info = broadband.source(frequency, 65536)
        return pulse, dict(**info, provenance='Measured generator record verified against analytic formula')
    # Plate 1 has NO measured 300 kHz drive or receiving record. This ideal
    # pulse is declared explicitly and is identical on both sides of the test.
    t = np.arange(1750) / 5000000
    end = 5 / frequency
    drive = np.where((t > 0) & (t < end), np.sin(2 * np.pi * frequency * t) * np.sin(np.pi * t / end)**2, 0.)
    pulse, _ = analytic_source(frequency, 5, t, drive)
    return pulse, dict(method='synthetic_five_cycle_Hann_sine', frequency_hz=frequency, cycles=5,
        pulse_end_s=end, source_sample_rate_hz=5000000,
        normalization='Unit peak of the declared synthetic 5 MHz source samples',
        synthetic_sample_peak_v=float(np.max(abs(drive))),
        provenance='Constructed numerical input; no plate-1 300 kHz experimental record exists',
        experimental_generator_verified=False, experimental_receiver_comparison_available=False,
        received_signal_fitted=False)


def stages():
    return ['full-order-300khz', 'build-mor', *[f'mor-order-{order:04d}' for order in ORDERS]]


def prepare(root):
    if root.exists():
        raise FileExistsError('Choose a new root; use --run only to resume a frozen study')
    reference = reference_identity()
    screen = memory_estimate(reference['samples'], 2, max(ORDERS))
    plan = dict(schema=1, study='ERA MOR plus broadband at the saved 5.340576171875 ns grid',
        stages=stages(), orders=ORDERS, frequencies_hz=FREQUENCIES, target_percent=TARGET,
        reference=reference, source_sha256=sources(), production_preflight=queue.preflight(),
        build_backend=BUILD_BACKEND, build_options=BUILD_OPTIONS, memory_estimate=screen,
        source_metadata={str(f): target_source(f)[1] for f in FREQUENCIES},
        selection_policy='Choose the lowest-error eligible model using 50-250 kHz only. '
            'Report 300 kHz as an additional drive-shape validation, without using it to select order.',
        relative_error_denominator='Full-order reference voltage L2 norm; no amplitude/time/polarity alignment',
        simulation_started=False, ready_for_training_data=False,
        limitations=['ERA is input-output model reduction, not second-order M/K Galerkin projection.',
            'The ROM does not reconstruct full plate fields, strain energy, or internal PZT stresses.',
            'A kernel is valid only for its unchanged plate, PZT layout, materials, and electrical conditions.',
            'No mechanical passivity or infinite-time stability bound is asserted for truncated ERA models.',
            'Matrix-free FFT products retain the full recording; no dense Hankel matrix is allocated.',
            'PROPACK computes leading singular triplets iteratively; residual/orthogonality checks must pass.',
            'The partial singular spectrum is not the full numerical rank or an infinite-time error bound.',
            'The host Lanczos basis still uses memory; GPU FFT acceleration does not move the whole solver to GPU.',
            'Build cost and repeated-use speedup remain unverified on the full-size recording.',
            'The full-order 300 kHz result is reconstructed from the saved broadband response; no new full-plate run is required.',
            '300 kHz is a synthetic excitation comparison, not experimental validation.',
            'A0/S0-related traces are receiver differences/sums, not proven pure modes.'])
    root.mkdir(parents=True)
    write_json(root / 'plan.json', plan)
    with zipfile.ZipFile(root / 'frozen-source.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        names = {**plan['production_preflight']['source_sha256'], **plan['source_sha256']}
        for name in names:
            archive.write(PROJECT / name, name)
        archive.write(root / 'plan.json', 'plan.json')
    write_json(root / 'status.json', dict(status='prepared', computation_started=False, pod_started=False))
    assess(root)
    print('PREPARED ONLY:', root, flush=True)
    print('Full-order 300 kHz -> matrix-free ERA build -> seven MOR size checks, each reconstructing 50-300 kHz.', flush=True)
    print('No SVD, reduced propagation, full-plate simulation, or Runpod action was started.', flush=True)


def load_plan(root):
    plan = json.loads((root / 'plan.json').read_text())
    if (plan['source_sha256'] != sources() or plan['production_preflight'] != queue.preflight()
            or plan['reference'] != reference_identity() or plan['stages'] != stages()
            or plan['orders'] != ORDERS or plan['frequencies_hz'] != FREQUENCIES
            or plan['target_percent'] != TARGET or plan['build_backend'] != BUILD_BACKEND
            or plan.get('build_options') != BUILD_OPTIONS):
        raise ValueError('Prepared model, comparison settings, or scientific inputs changed')
    return plan


def receipt(folder, plan_digest):
    path = folder / 'result.json'
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if not result.get('completed'):
        return None
    if result['plan_sha256'] != plan_digest:
        raise ValueError('Result belongs to another prepared study')
    if not result.get('artifacts_sha256'):
        raise ValueError('Missing stage artifact checksums')
    if result.get('stage') != folder.name:
        raise ValueError('Result belongs to another stage')
    if folder.name == 'full-order-300khz':
        required = {'waveforms.npz', 'source.json'}
    elif folder.name == 'build-mor':
        required = {'model.npz', 'build.json'}
    elif result.get('rejected'):
        required = {'rejected.json'}
    else:
        required = {'impulse.npz', 'model-diagnostics.json'} | {
            f'{f // 1000}khz/{name}' for f in FREQUENCIES
            for name in ('waveforms.npz', 'source.json', 'comparison.json')}
    if not required.issubset(result['artifacts_sha256']):
        raise ValueError('Completed stage is missing required scientific artifacts')
    for name, digest in result['artifacts_sha256'].items():
        path = (folder / name).resolve()
        path.relative_to(folder.resolve())
        if sha256_file(path) != digest:
            raise ValueError('Saved stage artifact changed: ' + name)
    return result


def write_trace(folder, time_s, voltage, metadata):
    folder.mkdir(parents=True, exist_ok=True)
    atomic_write(folder / 'waveforms.npz', lambda stream: np.savez_compressed(stream, time_s=time_s, voltage_v=voltage))
    write_json(folder / 'source.json', metadata)


def reference_trace(root, frequency):
    folder = (root / 'full-order-300khz' if frequency == 300000 else
              PROJECT / REFERENCE / f'baseline/direct/{frequency // 1000}khz')
    with np.load(folder / 'waveforms.npz') as data:
        return data['time_s'].copy(), data['voltage_v'].copy()


def worker(root, stage):
    plan = load_plan(root)
    if stage not in plan['stages']:
        raise ValueError('Stage not in frozen schedule')
    folder = root / stage
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists():
        raise FileExistsError('A stage attempt already exists')
    result = dict(completed=False, stage=stage, plan_sha256=sha256_file(root / 'plan.json'), artifacts_sha256={})
    started = time.perf_counter()
    progress_lock, stopped = threading.Lock(), threading.Event()
    current = {'phase': 'starting'}

    def event(phase, **details):
        with progress_lock:
            state = dict(status='running', stage=stage, phase=phase,
                         wall_seconds=time.perf_counter() - started, **details)
            if 'steps_completed' in details:
                state['percent'] = 100 * details['steps_completed'] / details['requested_steps']
            current.clear()
            current.update(state)
            print(json.dumps(state), flush=True)
            write_json(folder / 'status.json', state, overwrite=True)

    def heartbeat():
        while not stopped.wait(15):
            with progress_lock:
                state = dict(current, heartbeat=True, wall_seconds=time.perf_counter() - started)
                # cuSOLVER supplies no trustworthy percentage during its SVD.
                # Report the phase and elapsed time instead of inventing one.
                print(json.dumps(state), flush=True)
                write_json(folder / 'status.json', state, overwrite=True)

    reporter = threading.Thread(target=heartbeat, daemon=True)
    reporter.start()

    try:
        t, h = saved_kernel()
        if stage == 'full-order-300khz':
            pulse, metadata = target_source(300000)
            event('reconstructing_full_order_300khz', internal_dt_s=DT)
            write_trace(folder, t, reconstruct(h, pulse(t)), metadata)
        elif stage == 'build-mor':
            if not receipt(root / 'full-order-300khz', result['plan_sha256']):
                raise ValueError('The first scheduled reference stage has not completed')
            model, diagnostic = fit_era(t, h, max(ORDERS), backend='cupy', event=event)
            atomic_write(folder / 'model.npz', lambda stream: np.savez_compressed(stream, **model))
            write_json(folder / 'build.json', diagnostic)
        else:
            if not receipt(root / 'build-mor', result['plan_sha256']):
                raise ValueError('The reduced model build is missing or changed')
            with np.load(root / 'build-mor/model.npz') as data:
                model = {key: data[key].copy() for key in data.files}
            requested = int(stage.rsplit('-', 1)[1])
            effective = min(requested, len(model['a']))
            result.update(requested_order=requested, effective_order=effective)
            event('checking_reduced_impulse', requested_order=requested, effective_order=effective)
            try:
                impulse, diagnostic = reduced_impulse(model, effective, len(t), event=event)
            except ArithmeticError as error:
                # A rejected model is a reported scientific outcome; try the
                # other predeclared orders without clipping unstable poles.
                write_json(folder / 'rejected.json', dict(reason=str(error), poles_clipped=False))
                result.update(rejected=True, eligible=False, rejection_reason=str(error))
            else:
                atomic_write(folder / 'impulse.npz', lambda stream: np.savez_compressed(stream, time_s=t, voltage_v=impulse))
                write_json(folder / 'model-diagnostics.json', diagnostic)
                records = []
                for frequency in FREQUENCIES:
                    event('reconstructing_mor_signal', frequency_hz=frequency)
                    pulse, metadata = target_source(frequency)
                    tick = time.perf_counter()
                    voltage = reconstruct(impulse, pulse(t))
                    reconstruction_seconds = time.perf_counter() - tick
                    child = folder / f'{frequency // 1000}khz'
                    write_trace(child, t, voltage, metadata)
                    reference_t, full = reference_trace(root, frequency)
                    if not np.array_equal(reference_t, t):
                        raise ValueError('Reference and MOR use different internal recording times')
                    changes = paired_metrics(full, voltage)
                    write_json(child / 'comparison.json', dict(frequency_hz=frequency, channels=changes,
                        target_percent=TARGET, within_target=within_target(changes, TARGET),
                        real_experiment_comparison=False, fitted_alignment=False))
                    errors = [row['raw_l2_change_percent'] for row in changes.values()]
                    maximum = max(errors) if all(value is not None for value in errors) else None
                    records.append(dict(frequency_hz=frequency, channels=changes, maximum_error_percent=maximum,
                        within_target=within_target(changes, TARGET), reconstruction_seconds=reconstruction_seconds))
                    display = f'{maximum:.6g}%' if maximum is not None else 'unresolved (zero reference norm)'
                    print(f'{stage} | {frequency // 1000} kHz | maximum raw waveform difference {display} | '
                          f'target <{TARGET}% | no amplitude or time fitting', flush=True)
                # Select order using only the five original direct reference drives.
                training = [row for row in records if row['frequency_hz'] != 300000]
                result.update(rejected=False, comparisons=records,
                    eligible=all(row['within_target'] for row in training),
                    selection_error_percent=max(row['maximum_error_percent'] for row in training)
                        if all(row['maximum_error_percent'] is not None for row in training) else None,
                    additional_300khz_within_target=records[-1]['within_target'],
                    impulse_seconds=diagnostic['impulse_seconds'],
                    online_impulse_and_six_reconstructions_seconds=diagnostic['impulse_seconds'] +
                        sum(row['reconstruction_seconds'] for row in records))
        for path in folder.rglob('*'):
            if (path.is_file() and 'previous-attempts' not in path.relative_to(folder).parts
                    and path.name not in ('result.json', 'status.json', 'owner.json', 'worker.log')):
                result['artifacts_sha256'][path.relative_to(folder).as_posix()] = sha256_file(path)
        result['completed'] = True
        event('completed')
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        stopped.set()
        reporter.join()
        result['wall_seconds'] = time.perf_counter() - started
        write_json(folder / 'result.json', result)
        write_json(folder / 'status.json', dict(status='completed' if result['completed'] else 'interrupted',
                   error=result.get('error'), wall_seconds=result['wall_seconds']), overwrite=True)


def assess(root):
    plan = json.loads((root / 'plan.json').read_text())
    digest = sha256_file(root / 'plan.json')
    results = [receipt(root / stage, digest) for stage in plan['stages']]
    candidates = [row for row in results[2:] if row]
    eligible = [row for row in candidates if row.get('eligible')]
    selected = min(eligible, key=lambda row: (row['selection_error_percent'], -row['effective_order'])) if eligible else None
    complete = all(row is not None for row in results)
    answer = dict(completed=complete, stages_completed=sum(row is not None for row in results),
        stages_total=len(results), target_percent=TARGET, internal_dt_s=DT,
        selected_order=selected['effective_order'] if selected else None,
        selected_candidate=selected['stage'] if selected else None,
        selected_comparisons=selected.get('comparisons') if selected else None,
        numerical_method_pass=bool(complete and selected and selected['additional_300khz_within_target']),
        ready_for_training_data=False, candidate_results=candidates,
        build_seconds=results[1]['wall_seconds'] if results[1] else None,
        reference_runtime_note='The saved H200 full-order broadband solver took about 232 s. '
            'The new full-order 300 kHz stage reuses its saved kernel, so its short reconstruction '
            'time is NOT a fresh full-order simulation speed benchmark.',
        limitations=plan['limitations'])
    if selected and 'reference' in plan:
        online = selected['online_impulse_and_six_reconstructions_seconds']
        answer['timing_comparison'] = dict(mor_online_seconds=online,
            era_build_worker_seconds=answer['build_seconds'],
            era_build_plus_selected_online_seconds=answer['build_seconds'] + online,
            saved_full_order_solver_seconds=plan['reference']['original_broadband_solver_seconds'],
            saved_five_broadband_reconstructions_seconds=plan['reference']['original_five_reconstruction_seconds'],
            saved_solver_to_mor_online_ratio=plan['reference']['original_broadband_solver_seconds'] / online,
            all_order_validation_worker_seconds=sum(row['wall_seconds'] for row in candidates),
            same_hardware_and_fresh_full_order_rerun=False,
            note='Historical H200 solver timing versus current ROM; not a controlled fresh hardware speed benchmark. '
                 'Reusing the saved full-order kernel already avoids solving again. '
                 'ERA has a build cost and does not accelerate the first response of a new plate.')
    write_json(root / 'findings.json', answer, overwrite=True)
    lines = [f'ERA MOR + broadband | {answer["stages_completed"]}/{len(results)} stages complete',
        'Reference internal step: 5.340576171875 ns; 350 us recording; FP64.',
        f'Raw voltage differences: lower is better; target <{TARGET}% in top, bottom, S0-related and A0-related traces.',
        'No fitted gain, time shift, polarity correction, damping or pole clipping.']
    for candidate in candidates:
        error = candidate.get('selection_error_percent')
        display = f'{error:.6g}%' if error is not None else 'unresolved'
        lines.append(candidate['stage'] + ' | ' + ('REJECTED: ' + candidate['rejection_reason'] if candidate.get('rejected') else
            f'actual order {candidate["effective_order"]}; worst 50-250 kHz error {display}'))
    if selected:
        lines.append(f'Selected order: {selected["effective_order"]}; 300 kHz pass: {selected["additional_300khz_within_target"]}')
        lines.append(f'MOR impulse plus six reconstructions: {selected["online_impulse_and_six_reconstructions_seconds"]:.6g} s; '
                     f'one-time ERA build worker: {answer["build_seconds"]:.6g} s.')
        for row in selected['comparisons']:
            values = '; '.join(f'{name} {values["raw_l2_change_percent"]:.6g}%' for name, values in row['channels'].items())
            lines.append(f'{row["frequency_hz"] // 1000} kHz | {values}')
    else:
        lines.append('No acceptable MOR model selected. This is pending until execution, or a failed accuracy test after completion.')
    lines.append('300 kHz has no plate-1 experimental ground truth. Mesh convergence and physical validation are separate.')
    atomic_write(root / 'findings.txt', lambda stream: stream.write(('\n'.join(lines) + '\n').encode()), overwrite=True)
    return answer


def run(root):
    plan = load_plan(root)
    with queue_lock(root):
        write_json(root / 'status.json', dict(status='running', computation_started=True), overwrite=True)
        try:
            for index, stage in enumerate(plan['stages'], 1):
                folder = root / stage
                if worker_alive(folder):
                    raise RuntimeError('An earlier stage worker is still running')
                if receipt(folder, sha256_file(root / 'plan.json')):
                    continue
                if (folder / 'result.json').exists():
                    error = json.loads((folder / 'result.json').read_text()).get('error', '')
                    if error and not error.startswith(('KeyboardInterrupt', 'SystemExit')):
                        raise RuntimeError('Investigate the failed stage before retrying: ' + error)
                archive_attempt(folder)
                print(f'STAGE {index}/{len(plan["stages"])} | {stage}', flush=True)
                with (folder / 'worker.log').open('a', encoding='utf-8') as log:
                    process = subprocess.Popen([sys.executable, '-u', '-m', 'examples.pzt_mor_study',
                        '--root', str(root), '--worker', stage], cwd=PROJECT, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                    try:
                        born = psutil.Process(process.pid).create_time()
                    except psutil.NoSuchProcess:
                        born = 0.
                    write_json(folder / 'owner.json', dict(pid=process.pid, started_unix=born), overwrite=True)
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end='', flush=True)
                    code = process.wait()
                assess(root)
                print((root / 'findings.txt').read_text(), flush=True)
                if code:
                    raise RuntimeError(f'Stage failed (exit {code}): {stage}')
            write_json(root / 'status.json', dict(status='completed', computation_running=False), overwrite=True)
        except BaseException as error:
            write_json(root / 'status.json', dict(status='interrupted', error=str(error),
                computation_running=any(worker_alive(root / stage) for stage in plan['stages'])), overwrite=True)
            raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT / STUDY)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--run', action='store_true')
    actions.add_argument('--assess', action='store_true')
    actions.add_argument('--worker')
    args = parser.parse_args()
    root = args.root.resolve()
    root.relative_to((PROJECT / 'outputs/simulation-runs').resolve())
    if args.prepare:
        prepare(root)
    elif args.run:
        run(root)
    elif args.worker:
        worker(root, args.worker)
    else:
        load_plan(root)
        assess(root)
        print((root / 'findings.txt').read_text())


if __name__ == '__main__':
    main()
