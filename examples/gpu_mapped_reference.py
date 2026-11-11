"""Bounded mapped CUDA/native action and time-integration references at 3.175 mm."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy import sparse
from scipy.linalg import eigh

from examples import recess_geometry_reference as geometry_reference
from examples import recess_operator_reference as operator_reference
from examples.tensor_wave import PROJECT, checked_receipt, identities, read_json, sealed, snapshot_sources
from traxis.config import CircularRecess, Material, Plate
from traxis.elmer_reference import run_monitored
from traxis.gpu_enriched_verlet import gpu_explicit_simulate
from traxis.gpu_mapped_tensor import build_gpu_mapped
from traxis.gpu_verification import gpu_package_identity
from traxis.highorder_verification import distribution_identity
from traxis.mapped_tensor import build_system
from traxis.recess_hex import make_recess_mesh
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import sha256_file, write_json


def compare_gpu(cpu, gpu, native, output, guard):
    with np.load(output/'global-actions.npz') as proof:
        fields, mapped, native_actions = (proof[name] for name in
            ('nodal_states', 'mapped_actions', 'native_actions'))
        ns = len(cpu.mesh.points_m)
        transform = sparse.csr_matrix((proof['transform_data'], proof['transform_indices'], proof['transform_indptr']),
                                      shape=(ns, ns))
    measured = np.array([gpu.stiffness@row for row in fields])
    dual = np.array([[transform.T@c for c in row.reshape(3, ns)] for row in measured]).reshape(measured.shape)
    cpu_errors = [float(np.linalg.norm(a-b)/np.linalg.norm(b)) for a, b in zip(measured[:3], mapped[:3])]
    native_errors = [float(np.linalg.norm(a-b)/np.linalg.norm(b)) for a, b in zip(dual[:3], native_actions[:3])]
    rigid_error = float(np.linalg.norm(measured[3:])/
        (np.linalg.norm(cpu.stiffness_majorant_kg_s2)*np.linalg.norm(fields[3:])))
    scaled = (-.5*gpu.stiffness)@fields[0]
    scale_error = float(np.linalg.norm(scaled+.5*mapped[0])/np.linalg.norm(mapped[0]))
    arrays = dict(nodal_states=fields, gpu_actions=measured, gpu_actions_native_dual=dual,
                  negative_scaled_action=scaled)
    columns = mode = None
    if native['bounded_full_matrix'] is not None:
        with np.load(output/'global-matrices.npz') as proof:
            matrix = proof['native_stiffness']
        n = len(matrix); measured_matrix = np.empty_like(matrix); basis = np.zeros(n)
        for column in range(n):
            guard.check(); basis[column] = 1.
            measured_matrix[:, column] = gpu.stiffness@basis
            basis[column] = 0.
        error = float(np.linalg.norm(measured_matrix-matrix)/np.linalg.norm(matrix))
        columns = dict(columns=n, native_relative_l2=error, passed=error < 1e-10)
        arrays['gpu_stiffness'] = measured_matrix
        eigenvalues, eigenvectors = eigh(matrix, np.diag(cpu.mass_kg), subset_by_index=(n-1, n-1))
        omega = math.sqrt(eigenvalues[0]); initial = eigenvectors[:, 0]
        initial *= 1e-9/np.max(abs(initial)); duration = 2/omega
        probe = np.zeros((1, n)); probe[0, np.argmax(abs(initial))] = 1.
        levels = []; forced_levels = []
        norm = math.sqrt(float(np.sum(cpu.mass_kg*initial**2)))
        drive_omega = .37*omega
        for steps in (32, 64, 128):
            result = gpu_explicit_simulate(gpu, steps, duration/steps, probe, np.zeros(n),
                initial_displacement=initial, guard=guard, energy_stride=1)
            exact_u = initial*np.cos(omega*duration)
            exact_v = -omega*initial*np.sin(omega*duration)
            error_u = math.sqrt(float(np.sum(cpu.mass_kg*(result.final_displacement_m-exact_u)**2)))/norm
            error_v = math.sqrt(float(np.sum(cpu.mass_kg*(result.final_velocity_m_s-exact_v)**2)))/(omega*norm)
            energy_range = float(np.ptp(result.modified_energy_j)/abs(result.modified_energy_j.mean()))
            levels.append(dict(steps=steps, dt_s=duration/steps, displacement_error=error_u,
                velocity_error=error_v, modified_energy_relative_range=energy_range, statistics=result.statistics))
            for name, value in dict(time_s=result.time_s, trace=result.displacement_m,
                final_u=result.final_displacement_m, final_v=result.final_velocity_m_s,
                exact_u=exact_u, exact_v=exact_v, modified_energy=result.modified_energy_j).items():
                arrays[f'mode_{steps}_{name}'] = value
            forced = gpu_explicit_simulate(gpu, steps, duration/steps, probe, cpu.mass_kg*initial*omega**2,
                force=lambda t: math.cos(drive_omega*t), guard=guard)
            factor = omega**2/(omega**2-drive_omega**2)
            forced_u = initial*factor*(np.cos(drive_omega*duration)-np.cos(omega*duration))
            forced_v = initial*factor*(-drive_omega*np.sin(drive_omega*duration)+omega*np.sin(omega*duration))
            forced_error_u = math.sqrt(float(np.sum(cpu.mass_kg*(forced.final_displacement_m-forced_u)**2)))/norm
            forced_error_v = math.sqrt(float(np.sum(cpu.mass_kg*(forced.final_velocity_m_s-forced_v)**2)))/(omega*norm)
            forced_levels.append(dict(steps=steps, displacement_error=forced_error_u, velocity_error=forced_error_v,
                                      statistics=forced.statistics))
            for name, value in dict(final_u=forced.final_displacement_m, final_v=forced.final_velocity_m_s,
                                   exact_u=forced_u, exact_v=forced_v).items():
                arrays[f'forced_{steps}_{name}'] = value
        ratios = {label+'_'+component:[rows[i][field]/rows[i+1][field] for i in (0, 1)]
            for label, rows in (('unforced', levels), ('forced', forced_levels))
            for component, field in (('u', 'displacement_error'), ('v', 'velocity_error'))}
        mode = dict(duration_s=duration, omega_rad_s=omega, drive_omega_rad_s=drive_omega,
            levels=levels, forced_levels=forced_levels, displacement_error_ratios=ratios,
            passed=bool(all(3.8 < value < 4.2 for values in ratios.values() for value in values)
                and max(row['modified_energy_relative_range'] for row in levels) < 1e-8),
            scope='Native small-mesh eigenmode and analytic modal drive; no physical port, waveform-window or mesh convergence claim.')
    np.savez_compressed(output/'gpu-comparisons.npz', **arrays)
    errors = cpu_errors+native_errors+[scale_error]
    return dict(passed=bool(max(errors) < 1e-10 and rigid_error < 1e-12
            and (columns is None or columns['passed']) and (mode is None or mode['passed'])),
        cpu_field_relative_l2=cpu_errors, native_dual_field_relative_l2=native_errors,
        rigid_motion_relative_residual=rigid_error, negative_scale_relative_l2=scale_error,
        all_columns=columns, mode_check=mode, proof_sha256=sha256_file(output/'gpu-comparisons.npz'))


def checked_settings(plan):
    expected = dict(kind='gpu_mapped_recess_reference', plate=asdict(Plate(.04, .03, .003175)),
        recess=asdict(CircularRecess(.021, .014, .008, .0015)), material=asdict(Material()), geometry_order=3,
        operator_tolerance=1e-10, rigid_tolerance=1e-12, modal_error_ratio_interval=[3.8, 4.2], energy_tolerance=1e-8)
    if (any(plan.get(k) != v for k, v in expected.items()) or type(plan.get('order')) is not int
            or plan['order'] not in (3, 4, 5) or type(plan.get('angular_cells')) is not int
            or plan['angular_cells'] not in (1, 4) or type(plan.get('defective')) is not bool):
        raise ValueError('mapped GPU reference settings or acceptance criteria changed')


def source_check(plan, guard):
    checked_settings(plan)
    if (any(plan[k] != v for k, v in identities().items())
            or any(sha256_file(PROJECT/name) != digest for name, digest in plan['analysis_sources'].items())
            or plan['runtime'] != distribution_identity(guard) or plan['gpu_packages'] != gpu_package_identity(guard)):
        raise ValueError('mapped GPU source/runtime changed')


def worker(plan_path, output, guard):
    plan = checked_receipt(read_json(plan_path)); output.mkdir()
    report = dict(completed=False, passed=False, ready_for_training_data=False, training_updates=0,
                  plan_fingerprint=plan['fingerprint'])
    try:
        source_check(plan, guard)
        mesh = make_recess_mesh(Plate(**plan['plate']), CircularRecess(**plan['recess']), plan['order'],
            angular_cells=plan['angular_cells'], geometry_order=plan['geometry_order'], defective=plan['defective'])
        cpu = build_system(mesh, Material(**plan['material']), guard)
        geometry, proof, model = geometry_reference.native_geometry(mesh, guard, return_model=True)
        native = operator_reference.compare_native(cpu, model, geometry, output, guard)
        gpu = build_gpu_mapped(cpu, guard=guard)
        comparison = compare_gpu(cpu, gpu, native, output, guard)
        source_check(plan, guard)
        report.update(completed=True, passed=bool(native['passed'] and comparison['passed']),
            geometry=geometry, system=gpu.screen, device=gpu.device, native=native, gpu=comparison,
            scope='Bounded curved-mesh CUDA actions and small modal time integration at 3.175 mm; no full-size recess waves, physical clamp calibration or accepted training data.')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'; raise
    finally:
        report['resources'] = guard.report(); write_json(output/'gpu-mapped-report.json', sealed(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--order', type=int, choices=(3, 4, 5), default=3)
    parser.add_argument('--angular-cells', type=int, choices=(1, 4), default=4)
    parser.add_argument('--intact', action='store_true'); parser.add_argument('--worker', action='store_true')
    parser.add_argument('--plan', type=Path); args = parser.parse_args(); output = args.output.resolve()
    guard = ResourceGuard(ResourceLimits(max_seconds=300, max_rss_mib=1024,
        max_system_memory_percent=85, min_available_mib=2048))
    if args.worker:
        return 0 if worker(args.plan, output, guard)['passed'] else 2
    if not output.is_relative_to(PROJECT/'outputs'):
        raise ValueError('mapped GPU evidence must remain in project outputs')
    output.mkdir(parents=True, exist_ok=False)
    drivers = (__file__, geometry_reference.__file__, operator_reference.__file__)
    analysis = {str(Path(name).resolve().relative_to(PROJECT)).replace('\\', '/'):sha256_file(name) for name in drivers}
    plan = dict(kind='gpu_mapped_recess_reference', plate=asdict(Plate(.04, .03, .003175)),
        recess=asdict(CircularRecess(.021, .014, .008, .0015)), material=asdict(Material()),
        order=args.order, angular_cells=args.angular_cells, defective=not args.intact, geometry_order=3,
        **identities(), analysis_sources=analysis, runtime=distribution_identity(guard),
        gpu_packages=gpu_package_identity(guard), operator_tolerance=1e-10, rigid_tolerance=1e-12,
        modal_error_ratio_interval=[3.8, 4.2], energy_tolerance=1e-8)
    snapshot_sources(output/'source-snapshot.zip', plan)
    with zipfile.ZipFile(output/'source-snapshot.zip', 'a', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in analysis:
            archive.writestr(name, (PROJECT/name).read_bytes())
    plan['source_snapshot_sha256'] = sha256_file(output/'source-snapshot.zip')
    write_json(output/'gpu-mapped-plan.json', sealed(plan))
    temporary = output/'temporary'; temporary.mkdir()
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
        'CUPY_CACHE_DIR':str(output/'kernel-cache'), 'CUPY_CACHE_SAVE_CUDA_SOURCE':'1',
        **{name:str(temporary) for name in ('TEMP', 'TMP', 'TMPDIR')}}
    result = dict(completed=False, passed=False, ready_for_training_data=False, training_updates=0,
                  plan_fingerprint=plan['fingerprint'])
    try:
        command = [sys.executable, '-m', 'examples.gpu_mapped_reference', '--worker',
            '--plan', str(output/'gpu-mapped-plan.json'), '--output', str(output/'native')]
        result['worker_run'] = run_monitored(command, PROJECT, env, guard, output/'worker.log')
        if result['worker_run']['exit_code'] not in (0, 2):
            raise RuntimeError('mapped GPU worker failed; inspect worker.log')
        report = checked_receipt(read_json(output/'native/gpu-mapped-report.json'))
        if not report['completed'] or report['plan_fingerprint'] != plan['fingerprint']:
            raise ValueError('mapped GPU evidence is incomplete or changed')
        paths = {item['path']:item['sha256'] for item in report['native']['element_files']}
        paths.update({'global-actions.npz':report['native']['global_actions_sha256'],
                      'gpu-comparisons.npz':report['gpu']['proof_sha256']})
        full = report['native']['bounded_full_matrix']
        if full is not None:
            paths['global-matrices.npz'] = full['matrices_sha256']
        if any(sha256_file(output/'native'/name) != digest for name, digest in paths.items()):
            raise ValueError('mapped GPU proof changed')
        source_check(plan, guard)
        if sha256_file(output/'source-snapshot.zip') != plan['source_snapshot_sha256']:
            raise ValueError('mapped GPU source snapshot changed')
        result.update(completed=True, passed=bool(report['passed'] and result['worker_run']['exit_code'] == 0),
            report_sha256=sha256_file(output/'native/gpu-mapped-report.json'))
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        result['resources'] = guard.report(); write_json(output/'monitor-result.json', sealed(result))
    print(json.dumps({k:v for k,v in result.items() if k != 'resources'}))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
