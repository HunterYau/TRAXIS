"""Bounded Elmer force/probe reciprocity and Hann-source interface checks."""

from dataclasses import asdict
import math
from pathlib import Path
import shutil
import time

import numpy as np

from traxis.config import Material, Plate
from traxis.elmer_reference import elmer_environment, run_monitored, write_elmer_mesh
from traxis.elmer_transient import read_scalar_probe, transient_sif
from traxis.fem import tone_burst
from traxis.geometry import structured_mesh
from traxis.quadratic_patches import load_quadratic_elmer_mesh, quadratic_surface_patch
from traxis.storage import atomic_write, canonical_hash, sha256_file, write_json


def verify_quadratic_patches(home: Path, output: Path, guard) -> dict:
    solver, env = elmer_environment(home)
    grid = solver.with_name('ElmerGrid' + solver.suffix)
    if not grid.is_file():
        raise FileNotFoundError('ElmerGrid is required for the quadratic aperture check')
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('quadratic aperture check needs a new or empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    plate, material = Plate(.04, .03, .003), Material()
    plan = {'kind': 'predeclared_quadratic_aperture_check', 'plate': asdict(plate), 'material': asdict(material),
            'cells': [2, 2, 1], 'centers_xy_m': [[.0103, .0091], [.0301, .0207]], 'radius_m': .0032,
            'frequency_hz': 100000., 'burst_cycles': 2, 'sample_dt_s': 5e-7, 'samples': 128,
            'peak_patch_force_n': 1., 'reference_volume_force_n_m3': 1e6,
            'relative_reciprocity_tolerance': 1e-8, 'relative_source_interface_tolerance': 1e-8,
            'full_disk_area_relative_tolerance': 1e-10, 'centroid_tolerance_m': 1e-13,
            'second_moment_tolerance_m2': 1e-16, 'reference_rms_floor_m': 1e-18,
            'solver_executable_sha256': sha256_file(solver), 'grid_executable_sha256': sha256_file(grid),
            'source_sha256': {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            'scope': 'Finite-area P2 mechanical load/probe and windowed-source interfaces on a coarse fixed mesh; no continuum, PZT or physical validation.',
            'ready_for_training_data': False}
    plan['fingerprint'] = canonical_hash(plan)
    write_json(output / 'aperture-plan.json', plan)
    report = {'kind': 'quadratic_mechanical_aperture_verification', 'completed': False, 'passed': False,
              'plan_fingerprint': plan['fingerprint'], 'patches': [], 'runs': [], 'ready_for_training_data': False,
              'training_updates': 0, 'scope': plan['scope']}
    started = time.perf_counter()
    try:
        write_elmer_mesh(structured_mesh(plate, tuple(plan['cells'])), output / 'linear-mesh')
        conversion = run_monitored([str(grid), '2', '2', 'linear-mesh', '-increase', '-decimals', '17', '-out', 'mesh'],
                                    output, env, guard, output / 'grid.log')
        if conversion['exit_code'] != 0:
            raise RuntimeError('quadratic aperture mesh conversion failed')
        mesh = load_quadratic_elmer_mesh(output / 'mesh', plate)
        weights = []
        for center in plan['centers_xy_m']:
            vector, details = quadratic_surface_patch(mesh, center, plan['radius_m'])
            report['patches'].append(details)
            weights.append(vector.reshape(-1, 3)[:, 2])
        weights = np.array(weights)
        if any(patch['full_disk_area_relative_error'] > plan['full_disk_area_relative_tolerance']
               or patch['centroid_error_m'] > plan['centroid_tolerance_m']
               or patch['full_disk_second_moment_error_m2'] > plan['second_moment_tolerance_m2']
               for patch in report['patches']):
            raise ArithmeticError('quadratic aperture moment checks failed before solving')
        support = np.flatnonzero(np.any(weights != 0, axis=0))
        points = tuple(int(node + 1) for node in support)
        dt, steps = plan['sample_dt_s'], plan['samples'] - 1
        times = np.arange(plan['samples']) * dt
        names = [f'value: displacement 3 at node {node}' for node in points]
        traces = []
        for index, label in enumerate(('a-to-b', 'b-to-a', 'uniform-burst')):
            directory = output / label
            directory.mkdir()
            shutil.copytree(output / 'mesh', directory / 'mesh')
            force = None
            if index < 2:
                force = np.zeros((len(mesh.points_m), 3))
                force[:, 2] = weights[index] * plan['peak_patch_force_n']
                force = force.ravel()
            (directory / 'transient.sif').write_text(transient_sif(material, dt_s=dt, steps=steps,
                drive_omega=2 * math.pi * plan['frequency_hz'], probe_node=points[0], probe_nodes=points,
                nodal_force_n=force, volume_force_n_m3=plan['reference_volume_force_n_m3'] if index == 2 else None,
                burst_cycles=plan['burst_cycles']), encoding='ascii')
            process = run_monitored([str(solver), 'transient.sif'], directory, env, guard, directory / 'solver.log')
            log = (directory / 'solver.log').read_text(encoding='utf-8', errors='replace')
            if process['exit_code'] != 0 or 'ELMER SOLVER FINISHED AT:' not in log:
                raise RuntimeError('quadratic aperture transient failed; inspect solver.log')
            columns = read_scalar_probe(directory, expected_times=times[1:], required_names=names + ['mean: displacement 3'])
            if index < 2:
                values = np.column_stack([columns[name] for name in names]) @ weights[1 - index, support]
            else:
                values = columns['mean: displacement 3']
            traces.append(np.r_[0., values])
            report['runs'].append({'label': label, 'process': process,
                'artifact_sha256': {str(p.relative_to(directory)).replace('\\', '/'): sha256_file(p)
                                    for p in sorted(directory.rglob('*')) if p.is_file()}})
        raw = np.array(traces)
        reference_rms = float(np.sqrt(np.mean(raw[1]**2)))
        if reference_rms <= plan['reference_rms_floor_m']:
            raise ArithmeticError('reciprocity reference signal is unresolved')
        reciprocity = float(np.linalg.norm(raw[0] - raw[1]) / np.linalg.norm(raw[1]))
        # Independent scalar Newmark kinematics with the existing Python
        # Hann burst checks native MATC source values and truncation on every
        # requested time. It is a source-interface check, not a new time-order
        # or continuum validation claim.
        acceleration = np.array([tone_burst(float(t), plan['frequency_hz'], plan['burst_cycles']) for t in times])
        acceleration *= plan['reference_volume_force_n_m3'] / material.density_kg_m3
        displacement, velocity = np.zeros(len(times)), 0.
        for i in range(steps):
            displacement[i + 1] = displacement[i] + dt * velocity + dt**2 / 4 * (acceleration[i] + acceleration[i + 1])
            velocity += dt / 2 * (acceleration[i] + acceleration[i + 1])
        source_error = float(np.linalg.norm(raw[2] - displacement) / np.linalg.norm(displacement))
        atomic_write(output / 'aperture-traces.npz', lambda stream: np.savez_compressed(stream, time_s=times,
            displacement_m=raw, reference_uniform_displacement_m=displacement, patch_weights=weights, support_nodes=support))
        guard.check(force=True)
        report.update({'completed': True, 'relative_reciprocity_error': reciprocity,
            'relative_source_interface_error': source_error, 'reference_rms_m': reference_rms,
            'observed_probe_nodes': len(support), 'nodes': len(mesh.points_m), 'tetrahedra': len(mesh.tetrahedra),
            'raw_traces_sha256': sha256_file(output / 'aperture-traces.npz'),
            'passed': reciprocity < plan['relative_reciprocity_tolerance'] and source_error < plan['relative_source_interface_tolerance']})
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter() - started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output / 'aperture-report.json', report)
    return report
