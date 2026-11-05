"""Isolated H1 aperture moments and independent Elmer transient comparisons."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

from traxis.config import Material, Plate
from traxis.elmer_operators import save_operators
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.elmer_reference import elmer_environment, run_monitored
from traxis.fem import tone_burst
from traxis.geometry import structured_mesh
from traxis.highorder import assemble_highorder, nodal_evaluation_transform
from traxis.highorder_patches import highorder_surface_patch
from traxis.highorder_verification import distribution_identity
from traxis.implicit_fem import implicit_simulate
from traxis.polynomial_patches import triangle_powers
from traxis.quadratic_patch_verification import verify_quadratic_patches
from traxis.quadratic_patches import load_quadratic_elmer_mesh, quadratic_surface_patch
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import atomic_write, canonical_hash, sha256_file, write_json


def _disk_average(a, b, region):
    quarter = math.gamma((a+1)/2)*math.gamma((b+1)/2)/(2*(a+b+2)*math.gamma((a+b+2)/2))
    if region == 'quarter':
        return quarter/(math.pi/4)
    if region == 'half':
        return 0. if b % 2 else 2*quarter/(math.pi/2)
    return 0. if a % 2 or b % 2 else 4*quarter/math.pi


def _moments(system, vector, center, radius, direction, region, order):
    import ngsolve as ng
    direction = np.array(direction, float)/np.linalg.norm(direction)
    field = ng.GridFunction(system.scalar_space)
    rows = []
    for a, b in triangle_powers(order):
        field.Set(((ng.x-center[0])/radius)**a*((ng.y-center[1])/radius)**b)
        measured = vector.reshape(3, -1) @ field.vec.FV().NumPy()
        expected = direction*_disk_average(a, b, region)
        rows.append({'powers': [a, b], 'expected_component_averages': expected.tolist(),
            'measured_component_averages': measured.tolist(), 'absolute_error': float(np.max(abs(measured-expected)))})
    return {'monomials': rows, 'maximum_absolute_error': max(row['absolute_error'] for row in rows)}


def _validate_job(job_path, output, guard):
    job_path, output = Path(job_path).resolve(), Path(output).resolve()
    if job_path.stat().st_size > 1024**2:
        raise MemoryError('aperture job exceeds 1 MiB')
    job = json.loads(job_path.read_text())
    if canonical_hash({k: v for k, v in job.items() if k != 'fingerprint'}) != job['fingerprint']:
        raise ValueError('aperture worker job changed')
    if 'highorder-aperture-plan.json' not in job['input_sha256']:
        raise ValueError('aperture job must bind its frozen plan and native inputs')
    if not output.is_relative_to(job_path.parent) or (output.exists() and any(output.iterdir())):
        raise ValueError('aperture worker output must be new and within its job directory')
    expected = {'highorder-aperture-plan.json', 'source-snapshot.zip'}
    expected.update(str(p.relative_to(job_path.parent)).replace('\\', '/')
                    for p in (job_path.parent / 'elmer').rglob('*') if p.is_file())
    if set(job['input_sha256']) != expected:
        raise ValueError('aperture job must bind every native input')
    for name, digest in job['input_sha256'].items():
        path = (job_path.parent / name).resolve()
        if not path.is_relative_to(job_path.parent) or sha256_file(path) != digest:
            raise ValueError('aperture input changed or escaped the job directory')
    plan = json.loads((job_path.parent / 'highorder-aperture-plan.json').read_text())
    if (canonical_hash({k: v for k, v in plan.items() if k != 'fingerprint'}) != plan['fingerprint']
            or plan['fingerprint'] != job['plan_fingerprint'] or reference_source_hashes() != plan['source_sha256']
            or sha256_file(job_path.parent / 'source-snapshot.zip') != plan['source_snapshot_sha256']
            or distribution_identity(guard) != plan['runtime'] or plan['orders'] != [1, 2, 3, 4]
            or plan['meshes'] != ['baseline', 'renumbered', 'refined']):
        raise ValueError('aperture plan/source/runtime changed')
    return job_path.parent, output, plan


def run_native_job(job_path, output, guard):
    root, output, plan = _validate_job(job_path, output, guard)
    output.mkdir(parents=True, exist_ok=True)
    result = {'completed': False, 'passed': False, 'ready_for_training_data': False, 'training_updates': 0,
        'plan_fingerprint': plan['fingerprint'], 'cases': [], 'waves': []}
    try:
        native_plan = json.loads((root / 'elmer/aperture-plan.json').read_text())
        plate, material = Plate(**native_plan['plate']), Material(**native_plan['material'])
        quadratic = load_quadratic_elmer_mesh(root / 'elmer/mesh', plate)
        patches = [(center, (0., 0., 1.), 'full') for center in native_plan['centers_xy_m']]
        patches += [((0., plate.height_m/2), (1., 2., 3.), 'half'), ((0., 0.), (1., 2., 3.), 'quarter')]
        radius = native_plan['radius_m']
        traces = []
        all_passed = True
        with np.load(root / 'elmer/aperture-traces.npz', allow_pickle=False) as saved:
            elmer_traces, times = saved['displacement_m'].copy(), saved['time_s'].copy()
        if times.shape != (128,) or elmer_traces.shape != (3, 128) or not np.isfinite(elmer_traces).all():
            raise ValueError('invalid bounded native Elmer trace archive')
        for name in plan['meshes']:
            mesh = structured_mesh(plate, (3, 3, 2) if name == 'refined' else (2, 2, 1))
            if name == 'renumbered':
                permutation = np.random.default_rng(7351).permutation(len(mesh.points_m))
                mesh.points_m = mesh.points_m[permutation]
                mesh.tetrahedra = np.argsort(permutation)[mesh.tetrahedra[::-1, ::-1]]
            for order in plan['orders']:
                system = assemble_highorder(mesh, material, order, guard, max_dofs=5000, max_assembly_mib=64)
                row = {'mesh': name, 'order': order, 'dofs': system.space.ndof, 'patches': []}
                vectors = []
                transform = nodal_evaluation_transform(system, quadratic.points_m) if name != 'refined' and order == 2 else None
                for center, direction, region in patches:
                    vector, details = highorder_surface_patch(system, plate, center, radius, direction, guard)
                    moments = _moments(system, vector, center, radius, direction, region, order)
                    expected_area = math.pi*radius**2/({'full': 1, 'half': 2, 'quarter': 4}[region])
                    passed = moments['maximum_absolute_error'] < plan['moment_absolute_tolerance']
                    details.update({'region': region, 'moments': moments, 'expected_area_relative_error': abs(details['area_m2']/expected_area-1)})
                    passed = passed and details['expected_area_relative_error'] < plan['area_relative_tolerance']
                    if transform is not None:
                        nodal, _ = quadratic_surface_patch(quadratic, center, radius, direction)
                        expected = transform.T @ nodal
                        error = float(np.linalg.norm(vector-expected)/np.linalg.norm(expected))
                        details['p2_covector_relative_error'] = error
                        passed = passed and error < plan['covector_relative_tolerance']
                    details['passed'] = bool(passed)
                    all_passed = all_passed and passed
                    row['patches'].append(details)
                    vectors.append(vector)
                archive = output / f'{name}-p{order}-patches.npz'
                atomic_write(archive, lambda stream: np.savez_compressed(stream, covectors=np.array(vectors),
                    centers_xy_m=np.array([p[0] for p in patches]), radius_m=np.array(radius),
                    unit_directions=np.array([p[1]/np.linalg.norm(p[1]) for p in patches]),
                    coefficient_layout=np.array('component_major_hierarchical_H1')))
                row['patch_vectors_sha256'] = sha256_file(archive)
                row['passed'] = all(patch['passed'] for patch in row['patches'])
                result['cases'].append(row)
                if name != 'baseline':
                    continue
                operators = output / f'baseline-p{order}-operators.npz'
                save_operators(operators, system.stiffness, system.mass)
                row['operators_sha256'] = sha256_file(operators)
                pair, residuals, energy_ranges = [], [], []
                for transmitter in (0, 1):
                    solved = implicit_simulate(system.stiffness, system.mass, 127, native_plan['sample_dt_s'],
                        np.array([vectors[1-transmitter]]), force=lambda t: vectors[transmitter]*tone_burst(
                            float(t), native_plan['frequency_hz'], native_plan['burst_cycles']), guard=guard)
                    pair.append(solved.displacement_m[:, 0])
                    residuals.append(solved.maximum_relative_residual)
                    after = solved.energy_j[solved.energy_time_s > native_plan['burst_cycles']/native_plan['frequency_hz']]
                    energy_ranges.append(float(np.ptp(after)/abs(after.mean())))
                pair = np.array(pair)
                reciprocity = float(np.linalg.norm(pair[0]-pair[1])/np.linalg.norm(pair[1]))
                wave = {'order': order, 'reciprocity_relative_error': reciprocity, 'maximum_linear_residual': max(residuals),
                    'postburst_relative_energy_ranges': energy_ranges, 'scope': 'fixed coarse mesh; no spatial or time convergence claim'}
                passed = reciprocity < plan['wave_relative_tolerance'] and max(energy_ranges) < 1e-8
                if order == 2:
                    errors = [float(np.linalg.norm(pair[i]-elmer_traces[i])/np.linalg.norm(elmer_traces[i])) for i in (0, 1)]
                    wave['native_elmer_trace_relative_errors'] = errors
                    passed = passed and max(errors) < plan['wave_relative_tolerance']
                wave['passed'] = bool(passed)
                result['waves'].append(wave)
                all_passed = all_passed and passed
                traces.append(pair)
        atomic_write(output / 'highorder-aperture-traces.npz', lambda stream: np.savez_compressed(stream,
            time_s=times, displacement_m=np.array(traces), elmer_displacement_m=elmer_traces[:2]))
        result.update({'completed': True, 'passed': bool(all_passed),
            'traces_sha256': sha256_file(output / 'highorder-aperture-traces.npz')})
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['resources'] = guard.report()
        result['fingerprint'] = canonical_hash(result)
        write_json(output / 'native-report.json', result)
    return result


def verify_highorder_patches(home, output, guard):
    solver, elmer_env = elmer_environment(home)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('high-order aperture verification needs a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    source = reference_source_hashes()
    def snapshot(stream):
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in source:
                archive.writestr('traxis/'+name, (Path(__file__).parent / name).read_bytes())
    atomic_write(output / 'source-snapshot.zip', snapshot)
    plan = {'kind': 'highorder_mechanical_aperture_check', 'orders': [1, 2, 3, 4],
        'meshes': ['baseline', 'renumbered', 'refined'], 'runtime': distribution_identity(guard), 'source_sha256': source,
        'source_snapshot_sha256': sha256_file(output / 'source-snapshot.zip'), 'moment_absolute_tolerance': 1e-9,
        'area_relative_tolerance': 1e-10, 'covector_relative_tolerance': 1e-10, 'wave_relative_tolerance': 1e-8,
        'elmer_relative_integration_order': 1, 'ready_for_training_data': False,
        'elmer_kernel_sha256': {str(p.relative_to(Path(home).resolve())).replace('\\', '/'): sha256_file(p) for p in
            (solver, solver.with_name('ElmerGrid'+solver.suffix), solver.with_name('libelmersolver.dll'),
             Path(elmer_env['ELMER_LIB']) / 'StressSolve.dll', Path(elmer_env['ELMER_LIB']) / 'elements.def') if p.exists()},
        'scope': 'Represented surface fields and coarse discrete mechanical transients; no continuum convergence, PZT or physical acceptance.'}
    plan['fingerprint'] = canonical_hash(plan)
    write_json(output / 'highorder-aperture-plan.json', plan)
    result = {'kind': 'highorder_aperture_verification', 'completed': False, 'passed': False,
        'ready_for_training_data': False, 'training_updates': 0, 'plan_fingerprint': plan['fingerprint']}
    started = time.perf_counter()
    try:
        native = verify_quadratic_patches(home, output / 'elmer', guard)
        paths = [output / 'highorder-aperture-plan.json', output / 'source-snapshot.zip']
        paths += [p for p in (output / 'elmer').rglob('*') if p.is_file()]
        job = {'plan_fingerprint': plan['fingerprint'], 'input_sha256': {
            str(p.relative_to(output)).replace('\\', '/'): sha256_file(p) for p in paths}}
        job['fingerprint'] = canonical_hash(job)
        write_json(output / 'worker-job.json', job)
        env = os.environ.copy()
        env.update({'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
        remaining = max(1., guard.limits.max_seconds-(time.monotonic()-guard.started))
        process = run_monitored([sys.executable, '-m', 'traxis.highorder_patch_verification', '--job', str(output / 'worker-job.json'),
            '--output', str(output / 'ngsolve'), '--max-seconds', str(remaining), '--max-rss-mib', str(guard.limits.max_rss_mib)],
            Path(__file__).resolve().parents[1], env, guard, output / 'worker.log')
        if process['exit_code'] != 0:
            raise RuntimeError('high-order aperture worker failed; inspect worker.log and native-report.json')
        report = json.loads((output / 'ngsolve/native-report.json').read_text())
        if (canonical_hash({k: v for k, v in report.items() if k != 'fingerprint'}) != report['fingerprint']
                or report['plan_fingerprint'] != plan['fingerprint'] or not report['completed']
                or sha256_file(output / 'ngsolve/highorder-aperture-traces.npz') != report['traces_sha256']
                or reference_source_hashes() != source):
            raise ValueError('high-order aperture worker/source integrity check failed')
        for row in report['cases']:
            archive = output / 'ngsolve' / f"{row['mesh']}-p{row['order']}-patches.npz"
            if sha256_file(archive) != row['patch_vectors_sha256']:
                raise ValueError('saved high-order aperture archive changed')
            if row['mesh'] == 'baseline' and sha256_file(output / 'ngsolve' / f"baseline-p{row['order']}-operators.npz") != row['operators_sha256']:
                raise ValueError('saved high-order aperture operator archive changed')
        result.update({'completed': True, 'passed': native['passed'] and report['passed'], 'cases': report['cases'],
            'waves': report['waves'], 'elmer_report_sha256': sha256_file(output / 'elmer/aperture-report.json'),
            'native_report_sha256': sha256_file(output / 'ngsolve/native-report.json'), 'worker_run': process})
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['wall_seconds'] = time.perf_counter()-started
        result['resources'] = guard.report()
        result['fingerprint'] = canonical_hash(result)
        write_json(output / 'highorder-aperture-report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=60)
    parser.add_argument('--max-rss-mib', type=float, default=512)
    args = parser.parse_args()
    run_native_job(args.job, args.output, ResourceGuard(ResourceLimits(max_seconds=args.max_seconds, max_rss_mib=args.max_rss_mib)))
