"""Prepare/run serial delay-aware ERA and mechanical congruence comparisons.

Preparation is read-only scientific checking plus a fresh schedule. --run is
required for computation. No Runpod lifecycle API or automatic stop is used.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np

from examples import pzt_mor_study as baseline
from examples import pzt_era_delay as delay
from examples import pzt_mechanical_mor as mechanical
from examples.pzt_broadband import reconstruct, paired_metrics, within_target
from examples.pzt_era import ReducedModelGrowthError
from examples.plate_outputs import queue_lock, archive_attempt
from examples.pzt_convergence import worker_alive
from traxis.storage import write_json, atomic_write, sha256_file

ROOT = Path('outputs/simulation-runs/runpod-mor-remedies-v2')
DELAY_ORDERS = [32, 64, 128, 256, 512, 1024, 2048]
MECHANICAL_ORDERS = [16, 32, 64, 128]
PREFIX_NORM = 1e-8


def stages():
    return ['full-order-300khz', 'delay-build',
            *[f'delay-order-{r:04d}' for r in DELAY_ORDERS], 'mechanical-build',
            *[f'mechanical-order-{r:04d}' for r in MECHANICAL_ORDERS]]


def sources():
    names = ['examples/pzt_era_delay.py', 'examples/pzt_mechanical_mor.py',
             'examples/pzt_mor_remedies.py']
    return {**baseline.sources(), **{n: sha256_file(baseline.PROJECT / n) for n in names}}


def prepare(root):
    if root.exists():
        raise FileExistsError('Use a fresh root; --run resumes a frozen plan')
    reference = baseline.reference_identity()
    t, h = baseline.saved_kernel()
    dynamic_t, _, _, split = delay.split_record(t, h, PREFIX_NORM)
    from traxis.gpu_tensor import make_layout
    from traxis.config import Plate
    config = copy.deepcopy(baseline.queue.configuration())
    layout = make_layout(Plate(*config['plate_m']), tuple(config['mesh']['cells']), config['mesh']['order'],
                         max_dofs=32000000, max_device_mib=6144, max_host_mib=4096)
    ndof_bound = int(layout.screen['vector_dofs']) + 3 * 1024 * 3
    plan = dict(schema=1, stages=stages(), source_sha256=sources(),
        preflight=baseline.queue.preflight(), reference=reference, configuration=config,
        delay_orders=DELAY_ORDERS, mechanical_orders=MECHANICAL_ORDERS,
        prefix_relative_norm=PREFIX_NORM, delay_split=split,
        delay_memory=baseline.memory_estimate(len(dynamic_t), h.shape[1], max(DELAY_ORDERS)),
        mechanical_memory=mechanical.memory_estimate(ndof_bound, max(MECHANICAL_ORDERS)),
        frequencies_hz=baseline.FREQUENCIES, target_percent=baseline.TARGET,
        timestep_s=baseline.DT, simulation_started=False,
        historical_control='Original ERA results retained in runpod-h100-nvl-mor-005; never overwritten',
        limitations=['Neither remedy has passed full-plate accuracy validation.',
            'Mechanical builds require full physical operators and a large cloud GPU basis.',
            'Mechanical orders count displacement coordinates (twice as many first-order states).',
            'Delay branch preserves the exact early prefix; no waveform time/gain fitting.',
            '300 kHz is synthetic and reuses the full-order kernel, not real measured truth.',
            'Reduced positive stiffness and original Verlet timestep are checked; no pole clipping.',
            'Krylov order is fixed explicitly; no mesh/time/precision reductions to fit hardware.',
            'Runtime and full-plate speedup are unverified; a stable model can still be inaccurate.'])
    root.mkdir(parents=True)
    write_json(root / 'plan.json', plan)
    write_json(root / 'status.json', dict(status='prepared', computation_started=False))
    assess(root)
    print(f'PREPARED ONLY: {len(stages())} serial stages; no computation started. {root}', flush=True)
    return plan


def load_plan(root):
    p = json.loads((root / 'plan.json').read_text())
    if (p['source_sha256'] != sources() or p['preflight'] != baseline.queue.preflight()
            or p['reference'] != baseline.reference_identity() or p['stages'] != stages()
            or p['delay_orders'] != DELAY_ORDERS or p['mechanical_orders'] != MECHANICAL_ORDERS
            or p['prefix_relative_norm'] != PREFIX_NORM or p['target_percent'] != baseline.TARGET
            or p['frequencies_hz'] != baseline.FREQUENCIES or p['timestep_s'] != baseline.DT
            or p['configuration'] != baseline.queue.configuration()):
        raise ValueError('Scientific settings/source changed; prepare a new plan')
    return p


def receipt(folder, digest):
    p = folder / 'result.json'
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    if not r.get('completed'):
        return None
    if r['plan_sha256'] != digest or r['stage'] != folder.name:
        raise ValueError('Stage/plan identity mismatch')
    hashes = r['artifacts_sha256']
    if folder.name.endswith('build'):
        required = {'model.npz', 'build.json'}
    elif folder.name == 'full-order-300khz':
        required = {'waveforms.npz', 'source.json'}
    else:
        required = {'impulse.npz', 'model-diagnostics.json',
            *[f'{f // 1000}khz/{n}' for f in baseline.FREQUENCIES
              for n in ('waveforms.npz', 'comparison.json')]}
    if not required.issubset(hashes):
        raise ValueError('Incomplete stage artifact manifest')
    for name, expected in hashes.items():
        path = (folder / name).resolve()
        path.relative_to(folder.resolve())
        if sha256_file(path) != expected:
            raise ValueError('Stage artifact changed: ' + name)
    return r


def worker(root, stage):
    plan = load_plan(root)
    if stage not in plan['stages']:
        raise ValueError('Unknown remedy stage')
    folder = root / stage
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists():
        raise FileExistsError('Preserve earlier attempt before retrying')
    digest = sha256_file(root / 'plan.json')
    result = dict(completed=False, stage=stage, plan_sha256=digest)
    started, stopped, lock = time.perf_counter(), threading.Event(), threading.Lock()
    current = {'phase': 'starting'}
    def event(phase, **details):
        with lock:
            current.clear()
            current.update(stage=stage, phase=phase, wall_seconds=time.perf_counter() - started)
            current.update(details)
            if 'steps_completed' in details:
                current['percent'] = 100 * details['steps_completed'] / details['requested_steps']
            print(json.dumps(current), flush=True)
            write_json(folder / 'status.json', current, overwrite=True)
    def heartbeat():
        while not stopped.wait(15):
            with lock:
                print(json.dumps(dict(current, heartbeat=True,
                                      wall_seconds=time.perf_counter() - started)), flush=True)
    reporter = threading.Thread(target=heartbeat, daemon=True); reporter.start()
    try:
        t, h = baseline.saved_kernel()
        if stage == 'full-order-300khz':
            pulse, info = baseline.target_source(300000)
            baseline.write_trace(folder, t, reconstruct(h, pulse(t)), info)
        elif stage.endswith('build'):
            if not receipt(root / 'full-order-300khz', digest):
                raise ValueError('Missing reference stage')
            if stage == 'delay-build':
                model, info = delay.fit_delayed(t, h, max(DELAY_ORDERS), event=event,
                                               prefix_relative_norm=PREFIX_NORM)
            else:
                model, info = mechanical.build_plate(plan['configuration'], max(MECHANICAL_ORDERS), event=event)
                if len(t) - 1 < info['required_full_steps']:
                    raise ArithmeticError('Physical build requires a different stable timestep')
            atomic_write(folder / 'model.npz', lambda f: np.savez_compressed(f, **model))
            write_json(folder / 'build.json', info)
        else:
            method = stage.split('-')[0]
            parent = root / f'{method}-build'
            if not receipt(parent, digest):
                raise ValueError('Missing verified model build')
            with np.load(parent / 'model.npz') as z:
                model = {k: z[k].copy() for k in z.files}
            requested = int(stage.rsplit('-', 1)[1])
            available = len(model['a']) if method == 'delay' else len(model['load'])
            order = min(requested, available)
            result.update(requested_order=requested, effective_order=order, rejected=False)
            if method == 'delay':
                try:
                    kernel, info = delay.delayed_impulse(model, order, len(t), event=event)
                except ReducedModelGrowthError as error:
                    result.update(rejected=True, rejection_reason=str(error))
                    kernel, info = delay.delayed_impulse(model, order, len(t), event=event, diagnostic_only=True)
            else:
                kernel, info = mechanical.impulse(model, order, len(t), baseline.DT, event=event)
            atomic_write(folder / 'impulse.npz', lambda f: np.savez_compressed(f, time_s=t, voltage_v=kernel))
            write_json(folder / 'model-diagnostics.json', info)
            rows = []
            for frequency in baseline.FREQUENCIES:
                event('comparing_waveforms', frequency_hz=frequency)
                pulse, metadata = baseline.target_source(frequency)
                voltage = reconstruct(kernel, pulse(t))
                child = folder / f'{frequency // 1000}khz'
                baseline.write_trace(child, t, voltage, metadata)
                if frequency == 300000:
                    with np.load(root / 'full-order-300khz/waveforms.npz') as z:
                        reference_t, full = z['time_s'], z['voltage_v']
                else:
                    reference_t, full = baseline.reference_trace(root, frequency)
                if not np.array_equal(reference_t, t):
                    raise ValueError('Reference time grid differs')
                metrics = paired_metrics(full, voltage)
                row = dict(frequency_hz=frequency, channels=metrics,
                    within_target=within_target(metrics, baseline.TARGET),
                    diagnostic_only=result['rejected'], real_experiment_comparison=False,
                    fitted_alignment=False)
                errors = [m['raw_l2_change_percent'] for m in metrics.values()]
                row['maximum_error_percent'] = max(errors) if all(x is not None for x in errors) else None
                write_json(child / 'comparison.json', row)
                rows.append(row)
                print(f'{stage} | {frequency // 1000} kHz | raw error {row["maximum_error_percent"]}% | '
                      f'target <{baseline.TARGET}% | rejected={result["rejected"]}', flush=True)
            result.update(comparisons=rows, eligible=not result['rejected'] and all(r['within_target'] for r in rows))
        result['artifacts_sha256'] = {p.relative_to(folder).as_posix(): sha256_file(p)
            for p in folder.rglob('*') if p.is_file()
            and 'previous-attempts' not in p.parts and p.name not in ('status.json', 'worker.log', 'owner.json')}
        result['completed'] = True
        event('completed')
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        stopped.set(); reporter.join()
        result['wall_seconds'] = time.perf_counter() - started
        write_json(folder / 'result.json', result)


def assess(root):
    p = json.loads((root / 'plan.json').read_text())
    rows = [receipt(root / s, sha256_file(root / 'plan.json')) for s in p['stages']]
    candidates = [r for r in rows if r and 'comparisons' in r]
    result = dict(completed=all(r is not None for r in rows), stages_completed=sum(r is not None for r in rows),
        stages_total=len(rows), target_percent=baseline.TARGET, candidate_results=candidates,
        accepted_candidates=[r['stage'] for r in candidates if r['eligible']], ready_for_training_data=False)
    write_json(root / 'findings.json', result, overwrite=True)
    lines = [f'MOR remedies | {result["stages_completed"]}/{len(rows)} stages complete',
             f'Raw waveform differences; lower is better; target <{baseline.TARGET}%.',
             'Original timing, amplitude, polarity and physical parameters retained.']
    for r in candidates:
        lines.append(f'{r["stage"]}: ' + ('PASS' if r['eligible'] else 'FAIL'))
        for row in r['comparisons']:
            lines.append(f'  {row["frequency_hz"] // 1000} kHz: {row["maximum_error_percent"]}%'
                         + (' (unstable diagnostic only)' if r['rejected'] else ''))
    lines.append('No full-plate remedy is validated until its comparisons pass. Pod stopping is never automatic.')
    atomic_write(root / 'all-results.txt', lambda f: f.write(('\n'.join(lines) + '\n').encode()), overwrite=True)
    return result


def run(root):
    plan = load_plan(root)
    with queue_lock(root):
        write_json(root / 'status.json', dict(status='running', computation_started=True), overwrite=True)
        try:
            for i, stage in enumerate(plan['stages'], 1):
                folder = root / stage
                if worker_alive(folder):
                    raise RuntimeError('An earlier worker remains active')
                if receipt(folder, sha256_file(root / 'plan.json')):
                    continue
                if (folder / 'result.json').exists():
                    prior = json.loads((folder / 'result.json').read_text())
                    if not prior.get('error', '').startswith(('KeyboardInterrupt', 'SystemExit')):
                        raise RuntimeError('Investigate failed stage before retrying: ' + stage)
                archive_attempt(folder)
                print(f'STAGE {i}/{len(plan["stages"])} | {stage}', flush=True)
                # Each stage is isolated; worker logs and owner identity support
                # interruption detection and serial resume on the same run folder.
                with (folder / 'worker.log').open('a', encoding='utf-8') as log:
                    process = subprocess.Popen([sys.executable, '-u', '-m', 'examples.pzt_mor_remedies',
                        '--root', str(root), '--worker', stage], stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, encoding='utf-8', cwd=baseline.PROJECT)
                    import psutil
                    write_json(folder / 'owner.json', dict(pid=process.pid,
                               started_unix=psutil.Process(process.pid).create_time()), overwrite=True)
                    for line in process.stdout:
                        print(line, end='', flush=True); log.write(line); log.flush()
                    code = process.wait()
                    if code:
                        raise RuntimeError(f'{stage} exited with code {code}')
                assess(root)
            write_json(root / 'status.json', dict(status='completed', computation_running=False), overwrite=True)
        except BaseException as error:
            write_json(root / 'status.json', dict(status='interrupted', error=str(error)), overwrite=True)
            raise


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--worker')
    a = p.parse_args()
    if a.prepare:
        prepare(a.root.resolve())
    elif a.run:
        run(a.root.resolve())
    else:
        worker(a.root.resolve(), a.worker)


if __name__ == '__main__':
    main()
