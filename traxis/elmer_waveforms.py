"""Bounded P2 finite-plate mechanical references using independent Elmer assembly.

Elmer exports stiffness and consistent mass once. A monitored Python worker
uses checked sparse Newmark integration, avoiding the release's failing bulk
cache path. Every trace remains provisional until spatial/temporal/physical
acceptance is established.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import scipy

from traxis.acquisition import action_catalog
from traxis.cad_mesh import cad_mesh
from traxis.config import positive
from traxis.elmer_reference import elmer_environment, operator_sif, run_monitored, write_elmer_mesh
from traxis.forward import _cached_group
from traxis.quadratic_patches import TETRAHEDRON_EDGES, load_quadratic_elmer_mesh, quadratic_surface_patch
from traxis.scenarios import Scenario
from traxis.storage import atomic_write, canonical_hash, sha256_file, write_json


@dataclass(frozen=True)
class ElmerWaveSettings:
    mesh_size_m: float = .003
    substeps_per_output: int = 4
    mesh_algorithm: int = 10
    mesh_optimization: str = 'Netgen'
    max_cad_assembly_mib: float = 256
    max_matrix_screen_mib: float = 256
    max_quadratic_nodes: int = 20000
    max_tetrahedra: int = 40000
    max_internal_steps: int = 5000
    max_volume_error: float = .05
    max_operator_entries: int = 4000000
    max_operator_file_mib: float = 256

    def __post_init__(self):
        for key in ('mesh_size_m', 'max_cad_assembly_mib', 'max_matrix_screen_mib', 'max_volume_error', 'max_operator_file_mib'):
            positive(key, getattr(self, key))
        if self.max_volume_error > 1 or self.max_operator_file_mib > 256 or self.max_matrix_screen_mib > 512:
            raise ValueError('invalid geometry or sparse matrix screen limit')
        for key, low, high in (('substeps_per_output', 1, 64), ('max_quadratic_nodes', 10, 20000),
                               ('max_tetrahedra', 1, 40000), ('max_internal_steps', 1, 5000),
                               ('max_operator_entries', 1, 4000000)):
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{key} must lie in [{low},{high}]')
        if type(self.mesh_algorithm) is not int or self.mesh_algorithm not in (1, 4, 10) or self.mesh_optimization not in ('default', 'Netgen'):
            raise ValueError('invalid CAD meshing algorithm/optimizer')


def run_elmer_waveform_case(scenario: Scenario, settings: ElmerWaveSettings, home: Path, output: Path,
                           guard, *, transmitter=0, frequency_hz=None, state='intact', cache_dir=None):
    if scenario.nuisance.support != 'free' or scenario.nuisance.relative_noise_std or scenario.nuisance.absolute_noise_std_m:
        raise ValueError('numerical reference cases require a free plate and zero stochastic noise')
    if state not in ('intact', 'current'):
        raise ValueError('choose intact or current geometry/channel')
    if type(transmitter) is not int or not 0 <= transmitter < scenario.acquisition.sensor_count:
        raise ValueError('transmitter must be an available integer sensor index')
    frequency = scenario.acquisition.frequencies_hz[0] if frequency_hz is None else frequency_hz
    actions = tuple(a for a in action_catalog(scenario.acquisition.sensor_count, scenario.acquisition.frequencies_hz,
                                             scenario.acquisition.reciprocal)
                    if a.transmitter == transmitter and a.frequency_hz == frequency)
    if not actions:
        raise ValueError('requested transmitter/frequency has no eligible receive records')
    steps = (scenario.acquisition.samples - 1) * settings.substeps_per_output
    if steps > settings.max_internal_steps:
        raise ValueError('reference case exceeds its internal-step limit before meshing')
    solver, env = elmer_environment(home)
    grid = solver.with_name('ElmerGrid' + solver.suffix)
    if not grid.is_file():
        raise FileNotFoundError('ElmerGrid is required for P2 reference meshes')
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('P2 waveform reference needs a new or empty directory')
    output.mkdir(parents=True, exist_ok=True)
    source = {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    def snapshot(stream):
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in source:
                info = zipfile.ZipInfo('traxis/' + name)
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, (Path(__file__).parent / name).read_bytes())
    atomic_write(output / 'source-snapshot.zip', snapshot)
    runtime = {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__}
    identity = {'solver_executable': sha256_file(solver), 'grid_executable': sha256_file(grid)}
    for path in (solver.parent / 'libelmersolver.dll', Path(env['ELMER_LIB']) / 'StressSolve.dll'):
        if path.exists():
            identity[path.name] = sha256_file(path)
    plan = {'kind': 'bounded_elmer_p2_waveform_case', 'scenario': asdict(scenario), 'settings': asdict(settings),
            'actions': [asdict(a) for a in actions], 'state': state, 'source_sha256': source,
            'kernel_sha256': identity, 'runtime': runtime, 'source_snapshot_sha256': sha256_file(output / 'source-snapshot.zip'),
            'ready_for_training_data': False,
            'scope': 'Mechanical normal-force/displacement reference; no PZT/adhesive/electrical domains or physical acceptance.'}
    plan['fingerprint'] = canonical_hash(plan)
    write_json(output / 'case-plan.json', plan)
    report = {'kind': 'elmer_p2_mechanical_waveforms', 'completed': False, 'plan_fingerprint': plan['fingerprint'],
              'ready_for_training_data': False, 'training_updates': 0}
    started = time.perf_counter()
    try:
        recesses = scenario.recesses if state == 'current' else ()
        linear, mesh_report = cad_mesh(scenario.plate, recesses, settings.mesh_size_m,
            max_assembly_mib=settings.max_cad_assembly_mib, max_tetrahedra=settings.max_tetrahedra,
            seed=scenario.geometry_seed, optimization=settings.mesh_optimization, algorithm=settings.mesh_algorithm, guard=guard)
        corners = linear.points_m[linear.tetrahedra]
        lengths = np.linalg.norm(corners[:, TETRAHEDRON_EDGES[:, 0]] - corners[:, TETRAHEDRON_EDGES[:, 1]], axis=2)
        mesh_report.update({'minimum_edge_m': float(lengths.min()), 'maximum_edge_m': float(lengths.max()),
                            'maximum_element_z_span_m': float(np.ptp(corners[:, :, 2], axis=1).max())})
        volume_error = abs(linear.removed_volume_m3 / linear.requested_removed_volume_m3 - 1) if linear.requested_removed_volume_m3 else 0.
        if volume_error > settings.max_volume_error:
            raise ValueError('CAD-facet recess volume error exceeds the reference limit')
        edge_keys = np.sort(linear.tetrahedra[:, TETRAHEDRON_EDGES], axis=2).reshape(-1, 2)
        quadratic_nodes = len(linear.points_m) + len(np.unique(edge_keys, axis=0))
        screen_mib = len(linear.tetrahedra) * 900 * 32 / 1024**2
        if quadratic_nodes > settings.max_quadratic_nodes or screen_mib > settings.max_matrix_screen_mib:
            raise MemoryError('P2 nodes/matrix assembly estimate exceed the local reference bounds')
        write_elmer_mesh(linear, output / 'linear-mesh', max_nodes=settings.max_quadratic_nodes, max_tetrahedra=settings.max_tetrahedra)
        conversion = run_monitored([str(grid), '2', '2', 'linear-mesh', '-increase', '-decimals', '17', '-out', 'mesh'],
                                    output, env, guard, output / 'grid.log')
        if conversion['exit_code'] != 0:
            raise RuntimeError('P2 reference mesh conversion failed')
        mesh = load_quadratic_elmer_mesh(output / 'mesh', scenario.plate)
        if len(mesh.points_m) != quadratic_nodes or len(mesh.tetrahedra) != len(linear.tetrahedra):
            raise ArithmeticError('quadratic conversion changed the screened node/element counts')
        sensors = scenario.acquisition.sensors(scenario.plate)
        ids = (transmitter, *(a.receiver for a in actions))
        weights, details = zip(*(quadratic_surface_patch(mesh, sensors[i], scenario.acquisition.patch_radius_m) for i in ids))
        if any(p['edge_clipped'] or p['full_disk_area_relative_error'] > 1e-8 or p['centroid_error_m'] > 1e-12
               or p['full_disk_second_moment_error_m2'] > 1e-14 for p in details):
            raise ArithmeticError('P2 physical aperture moment checks failed before solving')
        load, probes = weights[0], np.stack(weights[1:])
        atomic_write(output / 'loads-probes.npz', lambda stream: np.savez_compressed(stream, force_n=load, probes=probes))
        (output / 'operator.sif').write_text(operator_sif(scenario.nuisance.material), encoding='ascii')
        mesh_hashes = {str(p.relative_to(output)).replace('\\', '/'): sha256_file(p) for p in sorted((output / 'mesh').iterdir()) if p.is_file()}
        dt = scenario.acquisition.sample_dt_s / settings.substeps_per_output
        recipe = {'material': asdict(scenario.nuisance.material), 'mesh_sha256': mesh_hashes,
                  'loads_probes_sha256': sha256_file(output / 'loads-probes.npz'), 'frequency_hz': frequency,
                  'cycles': scenario.acquisition.cycles, 'dt_s': dt, 'steps': steps, 'stride': settings.substeps_per_output,
                  'source_sha256': source, 'kernel_sha256': identity, 'runtime': runtime}
        key = canonical_hash(recipe)
        computation = {}
        def compute():
            operator_run = run_monitored([str(solver), 'operator.sif'], output, env, guard, output / 'operator.log')
            log = (output / 'operator.log').read_text(encoding='utf-8', errors='replace')
            if operator_run['exit_code'] != 0 or 'Just saved matrix and stopped!' not in log:
                raise RuntimeError('native sparse reference assembly did not finish')
            inputs = ['operator_sizes.dat', 'operator_a.dat', 'operator_perm.dat', 'loads-probes.npz']
            job = {'kind': 'sparse_p2_mechanical_time_job', 'nodes': len(mesh.points_m), 'steps': steps, 'dt_s': dt,
                   'output_stride': settings.substeps_per_output, 'frequency_hz': frequency, 'cycles': scenario.acquisition.cycles,
                   'max_operator_entries': settings.max_operator_entries, 'max_operator_file_mib': settings.max_operator_file_mib,
                   'input_sha256': {name: sha256_file(output / name) for name in inputs}, 'source_sha256': source}
            job['fingerprint'] = canonical_hash(job)
            write_json(output / 'worker-job.json', job)
            remaining = max(1., guard.limits.max_seconds - (time.monotonic() - guard.started))
            worker_run = run_monitored([sys.executable, '-m', 'traxis.implicit_worker', '--job', str(output / 'worker-job.json'),
                '--output', str(output / 'implicit'), '--max-seconds', str(remaining), '--max-rss-mib', str(guard.limits.max_rss_mib)],
                Path(__file__).resolve().parents[1], env, guard, output / 'worker.log')
            if worker_run['exit_code'] != 0:
                raise RuntimeError('implicit reference worker did not finish; inspect worker.log/report')
            summary = json.loads((output / 'implicit/worker-report.json').read_text())
            if (canonical_hash({k: v for k, v in summary.items() if k != 'fingerprint'}) != summary['fingerprint']
                    or not summary['completed'] or sha256_file(output / 'implicit/traces.npz') != summary['traces_sha256']):
                raise ValueError('implicit reference output integrity check failed')
            with np.load(output / 'implicit/traces.npz', allow_pickle=False) as saved:
                if not np.allclose(saved['time_s'], np.arange(scenario.acquisition.samples) * scenario.acquisition.sample_dt_s, rtol=1e-13, atol=1e-18):
                    raise ValueError('implicit reference output time grid changed')
                result = saved['displacement_m'].copy()
            computation.update({'operator_run': operator_run, 'worker_run': worker_run,
                'worker_report_sha256': sha256_file(output / 'implicit/worker-report.json'),
                'maximum_relative_linear_residual': summary['maximum_relative_linear_residual'],
                'postburst_relative_energy_range': summary['postburst_relative_energy_range'],
                'stiffness_nonzeros': summary['stiffness_nonzeros'], 'mass_nonzeros': summary['mass_nonzeros'],
                'lu_factor_nonzeros': summary['lu_factor_nonzeros']})
            return result
        raw, hit = _cached_group(Path(cache_dir) if cache_dir else None, key, (len(actions), scenario.acquisition.samples), compute)
        gain = np.array([scenario.nuisance.peak_force_n * scenario.nuisance.sensor_gains[transmitter]
                         * scenario.nuisance.sensor_gains[a.receiver] for a in actions])
        if state == 'current':
            gain *= scenario.nuisance.current_gain_drift[transmitter] * np.array([scenario.nuisance.current_gain_drift[a.receiver] for a in actions])
        values = raw * gain[:, None]
        atomic_write(output / 'solver-traces.npz', lambda stream: np.savez_compressed(stream,
            time_s=np.arange(scenario.acquisition.samples) * scenario.acquisition.sample_dt_s, displacement_m=values))
        report.update({'completed': True, 'cache_hit': hit, 'raw_cache_key': key, 'forward_solves': int(not hit),
            'dt_internal_s': dt, 'internal_steps': steps, 'linear_mesh': mesh_report, 'quadratic_nodes': len(mesh.points_m),
            'tetrahedra': len(mesh.tetrahedra), 'matrix_assembly_screen_mib': screen_mib, 'patches': list(details),
            'conversion': conversion, 'computation': computation, 'trace_sha256': sha256_file(output / 'solver-traces.npz'),
            'gain_application': 'Unit-force raw probes multiplied by declared drive/coupling and current-only drift; zero noise.'})
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter() - started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output / 'waveform-case.json', report)
    return values, report
