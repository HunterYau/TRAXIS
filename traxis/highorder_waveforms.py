"""Bounded, isolated higher-order mechanical pulses on exact intact boxes."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile

import numpy as np

from traxis.acquisition import action_catalog
from traxis.config import positive
from traxis.elmer_operators import save_operators
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.elmer_reference import run_monitored
from traxis.fem import tone_burst
from traxis.geometry import estimate_mesh, structured_mesh
from traxis.highorder import assemble_highorder
from traxis.highorder_patches import highorder_surface_patch, patch_polynomial_moments
from traxis.highorder_verification import distribution_identity
from traxis.implicit_fem import implicit_simulate
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.scenarios import Scenario
from traxis.storage import atomic_write, canonical_hash, sha256_file, write_json


@dataclass(frozen=True)
class HighOrderWaveSettings:
    order: int = 4
    cells: tuple[int, int, int] = (5, 5, 3)
    substeps_per_output: int = 4
    max_dofs: int = 20000
    max_assembly_mib: float = 256
    max_internal_steps: int = 5000

    def __post_init__(self):
        object.__setattr__(self, 'cells', tuple(self.cells))
        if type(self.order) is not int or not 1 <= self.order <= 4:
            raise ValueError('reference degree must lie in [1,4]')
        estimate_mesh(self.cells)
        for name, low, high in (('substeps_per_output', 1, 64), ('max_dofs', 12, 60000), ('max_internal_steps', 1, 5000)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name} must lie in [{low},{high}]')
        positive('max assembly MiB', self.max_assembly_mib)
        if self.max_assembly_mib > 512:
            raise ValueError('high-order assembly screen must not exceed 512 MiB')


def screen_highorder_reference(settings):
    estimate = estimate_mesh(settings.cells)
    dofs = 3*math.prod(settings.order*n+1 for n in settings.cells)
    local = 3*math.comb(settings.order+3, 3)
    assembly_mib = estimate.tetrahedra_upper*local**2*32/1024**2
    if (dofs > settings.max_dofs or estimate.nodes_upper > 20000 or estimate.tetrahedra_upper > 40000
            or assembly_mib > settings.max_assembly_mib or estimate.assembly_bytes_upper > settings.max_assembly_mib*1024**2):
        raise MemoryError('structured high-order degree/element/assembly screen exceeds its limits')
    return {'vector_dofs': dofs, 'vertices': estimate.nodes_upper, 'tetrahedra': estimate.tetrahedra_upper,
        'assembly_screen_mib': assembly_mib}


def reference_actions(scenario, settings, transmitter=0, frequency_hz=None):
    if scenario.nuisance.support != 'free' or scenario.nuisance.relative_noise_std or scenario.nuisance.absolute_noise_std_m:
        raise ValueError('high-order numerical references require a free plate and zero stochastic noise')
    if type(transmitter) is not int or not 0 <= transmitter < scenario.acquisition.sensor_count:
        raise ValueError('invalid high-order transmitter')
    frequency = scenario.acquisition.frequencies_hz[0] if frequency_hz is None else frequency_hz
    actions = tuple(a for a in action_catalog(scenario.acquisition.sensor_count, scenario.acquisition.frequencies_hz,
        scenario.acquisition.reciprocal) if a.transmitter == transmitter and a.frequency_hz == frequency)
    if not actions:
        raise ValueError('requested transmitter/frequency has no eligible receivers')
    if (scenario.acquisition.samples-1)*settings.substeps_per_output > settings.max_internal_steps:
        raise ValueError('high-order reference exceeds its internal-step limit before meshing')
    screen_highorder_reference(settings)
    return actions


def _load_traces(path, samples, records, dt):
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) != 2 or set(e.filename for e in archive.infolist()) != {'time_s.npy', 'displacement_m.npy'} or sum(e.file_size for e in archive.infolist()) > 16*1024**2:
            raise MemoryError('invalid or oversized high-order trace archive')
    with np.load(path, allow_pickle=False) as saved:
        times, values = saved['time_s'], saved['displacement_m']
        if (times.shape != (samples,) or values.shape != (records, samples) or values.dtype != np.float64
                or not np.isfinite(values).all() or not np.array_equal(times, np.arange(samples)*dt)):
            raise ValueError('high-order trace shape/values/time grid changed')
        return values.copy()


def run_native_job(path, output, guard):
    path, output = Path(path).resolve(), Path(output).resolve()
    if path.stat().st_size > 1024**2:
        raise MemoryError('high-order waveform job exceeds 1 MiB')
    job = json.loads(path.read_text())
    if canonical_hash({k: v for k, v in job.items() if k != 'fingerprint'}) != job['fingerprint']:
        raise ValueError('high-order waveform job changed')
    if (set(job) != {'kind', 'scenario', 'settings', 'transmitter', 'frequency_hz', 'source_sha256', 'runtime', 'fingerprint'}
            or job['kind'] != 'highorder_intact_unit_force_job' or reference_source_hashes() != job['source_sha256']):
        raise ValueError('high-order waveform job is missing bindings or its source changed')
    scenario, settings = Scenario.from_dict(job['scenario']), HighOrderWaveSettings(**job['settings'])
    actions = reference_actions(scenario, settings, job['transmitter'], job['frequency_hz'])
    if not output.is_relative_to(path.parent) or (output.exists() and any(output.iterdir())):
        raise ValueError('high-order waveform output must be new and within its job directory')
    if distribution_identity(guard) != job['runtime']:
        raise ValueError('high-order native runtime changed')
    output.mkdir(parents=True, exist_ok=True)
    report = {'kind': 'highorder_intact_mechanical_worker', 'completed': False, 'ready_for_training_data': False,
        'training_updates': 0, 'job_fingerprint': job['fingerprint']}
    started = time.perf_counter()
    try:
        mesh = structured_mesh(scenario.plate, settings.cells, max_assembly_bytes=settings.max_assembly_mib*1024**2)
        system = assemble_highorder(mesh, scenario.nuisance.material, settings.order, guard,
            max_dofs=settings.max_dofs, max_assembly_mib=settings.max_assembly_mib)
        atomic_write(output / 'mesh.npz', lambda stream: np.savez_compressed(stream, points_m=mesh.points_m, tetrahedra=mesh.tetrahedra))
        sensors = scenario.acquisition.sensors(scenario.plate)
        ids = (job['transmitter'], *(a.receiver for a in actions))
        vectors, patches = [], []
        for sensor in ids:
            vector, details = highorder_surface_patch(system, scenario.plate, sensors[sensor], scenario.acquisition.patch_radius_m, guard=guard)
            moments = patch_polynomial_moments(system, vector, sensors[sensor], scenario.acquisition.patch_radius_m,
                (0., 0., 1.), 'full', settings.order)
            details['maximum_monomial_absolute_error'] = moments['maximum_absolute_error']
            if details['edge_clipped'] or details['full_disk_area_relative_error'] > 1e-10 or moments['maximum_absolute_error'] > 1e-9:
                raise ArithmeticError('high-order physical footprint moment check failed before propagation')
            vectors.append(vector)
            patches.append(details)
        load, probes = vectors[0], np.array(vectors[1:])
        atomic_write(output / 'loads-probes.npz', lambda stream: np.savez_compressed(stream, force_n=load, probes=probes))
        save_operators(output / 'operators.npz', system.stiffness, system.mass)
        geometry = {name: sha256_file(output / name) for name in ('mesh.npz', 'loads-probes.npz', 'operators.npz')}
        corners = mesh.points_m[mesh.tetrahedra]
        volume = float(abs(np.linalg.det(corners[:,1:]-corners[:,:1])).sum()/6)
        expected_volume = scenario.plate.width_m*scenario.plate.height_m*scenario.plate.thickness_m
        if abs(volume/expected_volume-1) > 1e-12:
            raise ArithmeticError('high-order tetrahedra lost the intact box volume')
        maximum_edge = max(float(np.linalg.norm(corners[:,i]-corners[:,j], axis=1).max())
            for i in range(4) for j in range(i))
        assembly_seconds = time.perf_counter()-started
        stride = settings.substeps_per_output
        dt, steps = scenario.acquisition.sample_dt_s/stride, (scenario.acquisition.samples-1)*stride
        solved = implicit_simulate(system.stiffness, system.mass, steps, dt, probes,
            force=lambda t: load*tone_burst(float(t), job['frequency_hz'], scenario.acquisition.cycles), output_stride=stride, guard=guard)
        times = np.arange(scenario.acquisition.samples)*scenario.acquisition.sample_dt_s
        if not np.allclose(solved.time_s, times, rtol=1e-13, atol=1e-18):
            raise ArithmeticError('high-order integrator output grid changed')
        atomic_write(output / 'traces.npz', lambda stream: np.savez_compressed(stream, time_s=times, displacement_m=solved.displacement_m.T))
        atomic_write(output / 'energy.npz', lambda stream: np.savez_compressed(stream, time_s=solved.energy_time_s, energy_j=solved.energy_j))
        after = solved.energy_j[solved.energy_time_s > scenario.acquisition.cycles/job['frequency_hz']]
        energy_range = float(np.ptp(after)/abs(after.mean())) if len(after) > 1 and after.mean() else None
        if energy_range is not None and energy_range > 1e-8:
            raise ArithmeticError('high-order postburst energy drift exceeds tolerance')
        report.update({'completed': True, 'order': settings.order, 'cells': settings.cells, 'dofs': system.space.ndof,
            'dt_internal_s': dt, 'internal_steps': steps, 'assembly_seconds': assembly_seconds,
            'stiffness_nonzeros': system.stiffness.nnz, 'mass_nonzeros': system.mass.nnz, 'lu_factor_nonzeros': solved.factor_nonzeros,
            'maximum_relative_linear_residual': solved.maximum_relative_residual, 'postburst_relative_energy_range': energy_range,
            'geometry_load_sha256': geometry, 'geometry_load_fingerprint': canonical_hash(geometry), 'patches': patches,
            'linear_mesh': {'cells': settings.cells, 'nodes': len(mesh.points_m), 'tetrahedra': len(mesh.tetrahedra),
                'maximum_edge_m': maximum_edge, 'maximum_element_z_span_m': float(np.ptp(corners[:,:,2], axis=1).max()),
                'retained_volume_relative_error': abs(volume/expected_volume-1)},
            'artifact_sha256': {name: sha256_file(output / name) for name in ('mesh.npz', 'loads-probes.npz', 'operators.npz', 'traces.npz', 'energy.npz')}})
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output / 'worker-report.json', report)
    return report


def _checked_worker(directory, fingerprint):
    job_path = directory.parent / 'worker-job.json'
    if job_path.stat().st_size > 1024**2:
        raise MemoryError('cached high-order job exceeds 1 MiB')
    job = json.loads(job_path.read_text())
    if job.get('fingerprint') != fingerprint or canonical_hash({k: v for k, v in job.items() if k != 'fingerprint'}) != fingerprint:
        raise ValueError('cached high-order job changed')
    path = directory / 'worker-report.json'
    if path.stat().st_size > 4*1024**2:
        raise MemoryError('high-order worker report exceeds 4 MiB')
    report = json.loads(path.read_text())
    if (canonical_hash({k: v for k, v in report.items() if k != 'fingerprint'}) != report['fingerprint']
            or not report['completed'] or report['job_fingerprint'] != fingerprint
            or set(report['artifact_sha256']) != {'mesh.npz', 'loads-probes.npz', 'operators.npz', 'traces.npz', 'energy.npz'}):
        raise ValueError('high-order worker receipt is incomplete or changed')
    for name, digest in report['artifact_sha256'].items():
        if sha256_file(directory / name) != digest:
            raise ValueError('high-order worker artifact changed')
    if (report['geometry_load_sha256'] != {name: report['artifact_sha256'][name] for name in ('mesh.npz', 'loads-probes.npz', 'operators.npz')}
            or report['geometry_load_fingerprint'] != canonical_hash(report['geometry_load_sha256'])):
        raise ValueError('high-order mesh/load/operator identity changed')
    return report


def run_highorder_waveform_case(scenario, settings, output, guard, *, transmitter=0, frequency_hz=None, cache_dir=None):
    actions = reference_actions(scenario, settings, transmitter, frequency_hz)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('high-order waveform case needs a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    source, runtime = reference_source_hashes(), distribution_identity(guard)
    job = {'kind': 'highorder_intact_unit_force_job', 'scenario': asdict(scenario), 'settings': asdict(settings),
        'transmitter': transmitter, 'frequency_hz': actions[0].frequency_hz, 'source_sha256': source, 'runtime': runtime}
    job['fingerprint'] = canonical_hash(job)
    def snapshot(stream):
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in source:
                info = zipfile.ZipInfo('traxis/'+name)
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, (Path(__file__).parent / name).read_bytes())
    atomic_write(output / 'source-snapshot.zip', snapshot)
    plan = {**job, 'kind': 'highorder_intact_waveform_case', 'actions': [asdict(a) for a in actions], 'state': 'intact',
        'source_snapshot_sha256': sha256_file(output / 'source-snapshot.zip'), 'ready_for_training_data': False,
        'scope': 'Intact straight-tetrahedron mechanical reference; no PZT voltage drive or physical acceptance.'}
    plan['fingerprint'] = canonical_hash({k: v for k, v in plan.items() if k != 'fingerprint'})
    write_json(output / 'case-plan.json', plan)
    write_json(output / 'worker-job.json', job)
    report = {'kind': 'highorder_mechanical_waveforms', 'completed': False, 'ready_for_training_data': False,
        'training_updates': 0, 'plan_fingerprint': plan['fingerprint'], 'raw_cache_key': job['fingerprint']}
    started = time.perf_counter()
    try:
        base = Path(cache_dir).resolve() / job['fingerprint'] if cache_dir else output / 'group'
        base.mkdir(parents=True, exist_ok=True)
        pointer = base / 'completed.json'
        hit = pointer.exists()
        if hit:
            if pointer.stat().st_size > 4096:
                raise ValueError('oversized high-order cache receipt')
            receipt = json.loads(pointer.read_text())
            attempt = (base / receipt['attempt']).resolve()
            if not attempt.is_relative_to(base) or attempt.parent != base or sha256_file(attempt / 'native/worker-report.json') != receipt['worker_report_sha256']:
                raise ValueError('high-order cache receipt changed or escaped its group')
        else:
            attempt = Path(tempfile.mkdtemp(prefix='attempt-', dir=base))
            write_json(attempt / 'worker-job.json', job)
            remaining = max(1., guard.limits.max_seconds-(time.monotonic()-guard.started))
            env = os.environ.copy()
            env.update({'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
            process = run_monitored([sys.executable, '-m', 'traxis.highorder_waveforms', '--job', str(attempt / 'worker-job.json'),
                '--output', str(attempt / 'native'), '--max-seconds', str(remaining), '--max-rss-mib', str(guard.limits.max_rss_mib)],
                Path(__file__).resolve().parents[1], env, guard, attempt / 'worker.log')
            if process['exit_code'] != 0:
                raise RuntimeError('high-order waveform worker failed; inspect retained attempt log/report')
            _checked_worker(attempt / 'native', job['fingerprint'])
            _load_traces(attempt / 'native/traces.npz', scenario.acquisition.samples, len(actions), scenario.acquisition.sample_dt_s)
            write_json(pointer, {'attempt': attempt.name, 'worker_report_sha256': sha256_file(attempt / 'native/worker-report.json')})
            report['worker_run'] = process
        measured = _checked_worker(attempt / 'native', job['fingerprint'])
        raw = _load_traces(attempt / 'native/traces.npz', scenario.acquisition.samples, len(actions), scenario.acquisition.sample_dt_s)
        gain = np.array([scenario.nuisance.peak_force_n*scenario.nuisance.sensor_gains[transmitter]*scenario.nuisance.sensor_gains[a.receiver] for a in actions])
        values = raw*gain[:,None]
        atomic_write(output / 'solver-traces.npz', lambda stream: np.savez_compressed(stream,
            time_s=np.arange(scenario.acquisition.samples)*scenario.acquisition.sample_dt_s, displacement_m=values))
        if reference_source_hashes() != source or distribution_identity(guard) != runtime:
            raise ValueError('high-order source/runtime changed during the frozen case')
        report.update({'completed': True, 'cache_hit': hit, 'forward_solves': int(not hit), 'native_directory': str(attempt / 'native'),
            'worker_report_sha256': sha256_file(attempt / 'native/worker-report.json'),
            'trace_sha256': sha256_file(output / 'solver-traces.npz'), 'gain_application': 'Unit-force raw probes multiplied by declared drive/coupling gains; intact state, zero noise.',
            **{k: measured[k] for k in ('order', 'cells', 'dofs', 'dt_internal_s', 'internal_steps', 'linear_mesh', 'patches',
                'geometry_load_sha256', 'geometry_load_fingerprint', 'stiffness_nonzeros', 'mass_nonzeros', 'lu_factor_nonzeros',
                'maximum_relative_linear_residual', 'postburst_relative_energy_range')}})
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output / 'waveform-case.json', report)
    return values, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=60)
    parser.add_argument('--max-rss-mib', type=float, default=512)
    args = parser.parse_args()
    run_native_job(args.job, args.output, ResourceGuard(ResourceLimits(max_seconds=args.max_seconds, max_rss_mib=args.max_rss_mib)))
