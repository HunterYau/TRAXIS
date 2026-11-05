"""Isolated independent P1/P2 and exact-energy checks for the P1-P4 reference."""

import argparse
from dataclasses import asdict
import importlib.metadata as metadata
import json
from pathlib import Path
import os
import sys
import time
import zipfile

import numpy as np
from scipy.linalg import eigh

from traxis.config import Material, Plate
from traxis.elmer_operators import save_operators
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.elmer_reference import compare_operators, elmer_environment, operator_sif, read_elmer_operators, run_monitored, write_elmer_mesh
from traxis.geometry import structured_mesh
from traxis.highorder import assemble_highorder, nodal_evaluation_transform, polynomial_energy_checks
from traxis.quadratic_mass import exact_quadratic_mass
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import atomic_write, canonical_hash, sha256_file, write_json


def distribution_identity(guard=None):
    result = {}
    names = ('ngsolve', 'netgen-mesher', 'netgen-occt') + (('ngsolve-openblas',) if sys.platform != 'darwin' else ())
    for name in names:
        distribution = metadata.distribution(name)
        files = {}
        for entry in distribution.files or ():
            if entry.suffix.lower() in ('.pyd', '.dll', '.so', '.dylib') or entry.name == '__init__.py':
                if guard:
                    guard.check()
                files[str(entry).replace('\\', '/')] = sha256_file(Path(distribution.locate_file(entry)))
        result[name] = {'version': distribution.version, 'kernel_sha256': files}
    result['numpy'] = metadata.version('numpy')
    result['scipy'] = metadata.version('scipy')
    result['python'] = sys.version
    return result


def reference_mesh(recipe):
    mesh = structured_mesh(Plate(**recipe['plate']), tuple(recipe['cells']))
    mesh.points_m = mesh.points_m @ np.asarray(recipe['transform'], float).T
    return mesh


def run_native_job(job_path, output, guard):
    job_path, output = Path(job_path).resolve(), Path(output).resolve()
    if job_path.stat().st_size > 1024**2:
        raise MemoryError('high-order job exceeds 1 MiB')
    job = json.loads(job_path.read_text())
    if canonical_hash({k: v for k, v in job.items() if k != 'fingerprint'}) != job['fingerprint']:
        raise ValueError('high-order worker job changed')
    if 'highorder-plan.json' not in job['input_sha256']:
        raise ValueError('high-order job must bind its frozen plan and native inputs')
    if not output.is_relative_to(job_path.parent) or (output.exists() and any(output.iterdir())):
        raise ValueError('high-order worker output must be new and within the job directory')
    for name, expected in job['input_sha256'].items():
        path = (job_path.parent / name).resolve()
        if not path.is_relative_to(job_path.parent) or sha256_file(path) != expected:
            raise ValueError('high-order job input changed or escaped the job directory')
    plan = json.loads((job_path.parent / 'highorder-plan.json').read_text())
    expected = {'highorder-plan.json'}
    for name in ('rectangular', 'affine-skewed'):
        expected.update(f'{name}/{file}' for file in ('operator_sizes.dat', 'operator_a.dat', 'operator_perm.dat'))
        expected.update(f'{name}/mesh/{file.name}' for file in (job_path.parent / name / 'mesh').iterdir() if file.is_file())
    if set(job['input_sha256']) != expected:
        raise ValueError('high-order job must bind every native operator and mesh input')
    if (canonical_hash({k: v for k, v in plan.items() if k != 'fingerprint'}) != plan['fingerprint']
            or plan['fingerprint'] != job['plan_fingerprint'] or reference_source_hashes() != plan['source_sha256']
            or distribution_identity(guard) != plan['runtime'] or plan['orders'] != [1, 2, 3, 4] or len(plan['meshes']) != 2):
        raise ValueError('high-order plan/source/runtime changed or has invalid case counts')
    output.mkdir(parents=True, exist_ok=True)
    result = {'completed': False, 'passed': False, 'ready_for_training_data': False, 'cases': [], 'plan_fingerprint': plan['fingerprint']}
    try:
        for recipe in plan['meshes']:
            if recipe['name'] not in ('rectangular', 'affine-skewed') or recipe['cells'] != [2, 2, 1]:
                raise ValueError('invalid bounded high-order reference geometry')
            mesh, material = reference_mesh(recipe), Material(**recipe['material'])
            case = output / recipe['name']
            case.mkdir()
            orders = []
            previous = None
            for order in plan['orders']:
                guard.check(force=True)
                system = assemble_highorder(mesh, material, order, guard, max_dofs=1536, max_assembly_mib=32)
                energy = polynomial_energy_checks(system, material, (mesh.plate.width_m, mesh.plate.height_m, mesh.plate.thickness_m),
                                                 recipe['transform'], order)
                k, m = system.stiffness.toarray(), system.mass.toarray()
                eigenvalues = eigh(k, m, eigvals_only=True)
                normalized = eigenvalues / eigenvalues[-1]
                rigid = bool(np.max(abs(normalized[:6])) < plan['rigid_tolerance'] and normalized[6] > plan['rigid_tolerance'])
                frequencies = np.sqrt(np.maximum(eigenvalues[6:12], 0)) / (2*np.pi)
                decrease = previous is None or bool(np.all(frequencies <= previous * (1+plan['ritz_relative_tolerance'])))
                previous = frequencies
                row = {'order': order, 'dofs': system.space.ndof, 'topology_screen': system.screen, 'energies': energy,
                    'smallest_six_eigenvalues_over_largest': normalized[:6].tolist(), 'seventh_eigenvalue_over_largest': float(normalized[6]),
                    'six_rigid_modes': rigid, 'first_six_elastic_frequencies_hz': frequencies.tolist(),
                    'ritz_nonincrease_passed': decrease}
                passed = rigid and decrease and max(energy['maximum_mass_relative_error'], energy['maximum_stiffness_relative_error'],
                                                     energy['maximum_rigid_relative_force']) < plan['relative_tolerance']
                if order == 1:
                    transform = nodal_evaluation_transform(system, mesh.points_m)
                    inverse = np.linalg.inv(transform)
                    row['p1_independent_comparison'] = compare_operators(mesh, material, inverse.T @ k @ inverse, inverse.T @ m @ inverse)
                    passed = passed and row['p1_independent_comparison']['passed']
                if order == 2:
                    native = job_path.parent / recipe['name']
                    nodes = np.loadtxt(native / 'mesh/mesh.nodes', ndmin=2)[:,2:5]
                    ek, em = read_elmer_operators(native, len(nodes))
                    transform = nodal_evaluation_transform(system, nodes)
                    tets = np.loadtxt(native / 'mesh/mesh.elements', dtype=int, ndmin=2)[:,3:] - 1
                    exact_mass = exact_quadratic_mass(nodes, tets, material.density_kg_m3)
                    inverse = np.linalg.inv(transform)
                    elmer_exact_error = float(np.linalg.norm(em-exact_mass) / np.linalg.norm(exact_mass))
                    ngsolve_exact_error = float(np.linalg.norm(inverse.T@m@inverse-exact_mass) / np.linalg.norm(exact_mass))
                    stiffness_error = float(np.linalg.norm(k-transform.T@ek@transform) / np.linalg.norm(k))
                    mass_error = float(np.linalg.norm(m-transform.T@em@transform) / np.linalg.norm(m))
                    row['p2_elmer_basis_comparison'] = {'stiffness_relative_frobenius': stiffness_error,
                        'mass_relative_frobenius': mass_error, 'basis_condition_number': float(np.linalg.cond(transform)),
                        'basis_rank': int(np.linalg.matrix_rank(transform)), 'dimension': len(transform),
                        'elmer_exact_mass_relative_error': elmer_exact_error, 'ngsolve_exact_mass_relative_error': ngsolve_exact_error}
                    passed = passed and max(stiffness_error, mass_error, elmer_exact_error, ngsolve_exact_error) < plan['relative_tolerance'] and row['p2_elmer_basis_comparison']['basis_rank'] == len(transform)
                    atomic_write(case / 'p2-basis-transform.npz', lambda stream: np.savez_compressed(stream, transform=transform, points_m=nodes))
                path = case / f'order-{order}-operators.npz'
                save_operators(path, system.stiffness, system.mass)
                row.update({'passed': bool(passed), 'operators_sha256': sha256_file(path)})
                orders.append(row)
            result['cases'].append({'name': recipe['name'], 'orders': orders, 'passed': all(row['passed'] for row in orders)})
        guard.check(force=True)
        result.update({'completed': True, 'passed': all(row['passed'] for row in result['cases'])})
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['resources'] = guard.report()
        result['fingerprint'] = canonical_hash(result)
        write_json(output / 'native-report.json', result)
    return result


def verify_highorder(home, output, guard):
    solver, elmer_env = elmer_environment(home)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('high-order reference needs a new or empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    runtime = distribution_identity(guard)
    source = reference_source_hashes()
    def snapshot(stream):
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in source:
                info = zipfile.ZipInfo('traxis/' + name)
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, (Path(__file__).parent / name).read_bytes())
    atomic_write(output / 'source-snapshot.zip', snapshot)
    recipes = []
    for name, transform, material in (('rectangular', np.eye(3), Material()),
        ('affine-skewed', np.array([[1., .12, .04], [.05, .9, .03], [.01, .02, 1.1]]), Material(71e9, .29, 2800.))):
        recipes.append({'name': name, 'plate': asdict(Plate(.04, .03, .003)), 'cells': [2, 2, 1],
                        'transform': transform.tolist(), 'material': asdict(material)})
    plan = {'kind': 'independent_highorder_assembly_check', 'orders': [1, 2, 3, 4], 'meshes': recipes,
            'relative_tolerance': 1e-10, 'rigid_tolerance': 1e-10, 'ritz_relative_tolerance': 1e-9, 'runtime': runtime, 'source_sha256': source,
            'source_snapshot_sha256': sha256_file(output / 'source-snapshot.zip'), 'ready_for_training_data': False,
            'elmer_kernel_sha256': {str(p.relative_to(Path(home).resolve())).replace('\\', '/'): sha256_file(p) for p in
                (solver, solver.with_name('ElmerGrid' + solver.suffix), solver.with_name('libelmersolver.dll'), Path(elmer_env['ELMER_LIB']) / 'StressSolve.dll', Path(elmer_env['ELMER_LIB']) / 'elements.def') if p.exists()},
            'elmer_relative_integration_order': 1,
            'scope': 'Independent discrete operators and exact represented fields on fixed blocks; no finite-plate wave, aperture, PZT or physical acceptance.'}
    plan['fingerprint'] = canonical_hash(plan)
    write_json(output / 'highorder-plan.json', plan)
    result = {'kind': 'highorder_reference_verification', 'completed': False, 'passed': False,
              'ready_for_training_data': False, 'training_updates': 0, 'plan_fingerprint': plan['fingerprint'], 'native_runs': []}
    started = time.perf_counter()
    try:
        for recipe in recipes:
            case = output / recipe['name']
            case.mkdir()
            write_elmer_mesh(reference_mesh(recipe), case / 'linear-mesh')
            conversion = run_monitored([str(solver.with_name('ElmerGrid' + solver.suffix)), '2', '2', 'linear-mesh', '-increase', '-decimals', '17', '-out', 'mesh'],
                                       case, elmer_env, guard, case / 'grid.log')
            (case / 'operator.sif').write_text(operator_sif(Material(**recipe['material'])), encoding='ascii')
            assembly = run_monitored([str(solver), 'operator.sif'], case, elmer_env, guard, case / 'operator.log')
            if conversion['exit_code'] != 0 or assembly['exit_code'] != 0 or 'Just saved matrix and stopped!' not in (case / 'operator.log').read_text(errors='replace'):
                raise RuntimeError('independent high-order reference export did not finish')
            result['native_runs'].append({'case': recipe['name'], 'conversion': conversion, 'assembly': assembly})
        paths = [output / 'highorder-plan.json']
        for recipe in recipes:
            case = output / recipe['name']
            paths += [case / name for name in ('operator_sizes.dat', 'operator_a.dat', 'operator_perm.dat')]
            paths += [p for p in (case / 'mesh').iterdir() if p.is_file()]
        job = {'kind': 'highorder_reference_worker', 'plan_fingerprint': plan['fingerprint'],
                'input_sha256': {str(p.relative_to(output)).replace('\\', '/'): sha256_file(p) for p in paths}}
        job['fingerprint'] = canonical_hash(job)
        write_json(output / 'worker-job.json', job)
        remaining = max(1., guard.limits.max_seconds - (time.monotonic() - guard.started))
        env = os.environ.copy()
        env.update({'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
        worker = run_monitored([sys.executable, '-m', 'traxis.highorder_verification', '--job', str(output / 'worker-job.json'),
            '--output', str(output / 'ngsolve'), '--max-seconds', str(remaining), '--max-rss-mib', str(guard.limits.max_rss_mib)],
            Path(__file__).resolve().parents[1], env, guard, output / 'worker.log')
        if worker['exit_code'] != 0:
            raise RuntimeError('high-order worker failed; inspect worker.log and native-report.json')
        native = json.loads((output / 'ngsolve/native-report.json').read_text())
        if (canonical_hash({k: v for k, v in native.items() if k != 'fingerprint'}) != native['fingerprint']
                or native['plan_fingerprint'] != plan['fingerprint'] or not native['completed'] or reference_source_hashes() != source):
            raise ValueError('high-order worker/source integrity check failed')
        for case in native['cases']:
            for row in case['orders']:
                if sha256_file(output / 'ngsolve' / case['name'] / f"order-{row['order']}-operators.npz") != row['operators_sha256']:
                    raise ValueError('saved high-order operator archive changed')
        result.update({'completed': True, 'passed': native['passed'], 'cases': native['cases'], 'worker_run': worker,
                       'native_report_sha256': sha256_file(output / 'ngsolve/native-report.json')})
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['wall_seconds'] = time.perf_counter() - started
        result['resources'] = guard.report()
        result['fingerprint'] = canonical_hash(result)
        write_json(output / 'highorder-report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=60)
    parser.add_argument('--max-rss-mib', type=float, default=512)
    args = parser.parse_args()
    run_native_job(args.job, args.output, ResourceGuard(ResourceLimits(max_seconds=args.max_seconds, max_rss_mib=args.max_rss_mib)))
