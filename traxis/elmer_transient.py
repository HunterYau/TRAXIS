"""Bounded independent transient/load checks; no continuum/PZT validation."""

from dataclasses import asdict
import math
from pathlib import Path
import re
import shutil
import time

import numpy as np
from scipy.linalg import eigh

from traxis.config import Material, Plate, positive
from traxis.elmer_reference import elmer_environment, operator_sif, read_elmer_operators, run_monitored, write_elmer_mesh
from traxis.elmer_waveguide import read_quadratic_nodes
from traxis.geometry import structured_mesh
from traxis.storage import canonical_hash, sha256_file, write_json


def sinusoidal_displacement(times, acceleration, drive_omega, natural_omega=0.):
    """Zero-start solution of q'' + omega0²*q = a*sin(Omega*t), in meters.

    The sinc form remains finite at resonance. A short-time series avoids
    cancellation near zero, including the rigid-body natural frequency zero.
    """
    positive('drive angular frequency', drive_omega)
    if not math.isfinite(acceleration) or not math.isfinite(natural_omega) or natural_omega < 0:
        raise ValueError('need finite acceleration and nonnegative natural angular frequency')
    times = np.asarray(times, dtype=float)
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError('need finite nonnegative times')
    if natural_omega == 0:
        theta = drive_omega * times
        result = acceleration / drive_omega**2 * (theta - np.sin(theta))
    else:
        result = acceleration / (natural_omega + drive_omega) * (
            np.sin(natural_omega * times) / natural_omega
            - times * np.cos((natural_omega + drive_omega) * times / 2)
            * np.sinc((drive_omega - natural_omega) * times / (2 * np.pi)))
    short = np.maximum(natural_omega, drive_omega) * times < 1e-3
    series = acceleration * drive_omega * times**3 * (
        1 / 6 - (drive_omega**2 + natural_omega**2) * times**2 / 120
        + (drive_omega**4 + drive_omega**2 * natural_omega**2 + natural_omega**4) * times**4 / 5040)
    return np.where(short, series, result)


def transient_sif(material: Material, *, dt_s, steps, drive_omega, probe_node,
                  nodal_force_n=None, volume_force_n_m3=None):
    """Bossak alpha=0 is average-acceleration Newmark (gamma=.5, beta=.25).

    Stress Bodyforce is force/volume. Displacement i Load is a direct nodal
    force. Negative P2 equivalent nodal coefficients are retained.
    """
    positive('time step', dt_s)
    positive('drive angular frequency', drive_omega)
    if type(steps) is not int or not 1 <= steps <= 5000 or type(probe_node) is not int or not 1 <= probe_node <= 128:
        raise ValueError('invalid bounded step count or probe node')
    if (nodal_force_n is None) == (volume_force_n_m3 is None):
        raise ValueError('choose exactly one volume or nodal force definition')
    if volume_force_n_m3 is not None:
        if not math.isfinite(volume_force_n_m3):
            raise ValueError('volume force must be finite')
        forcing = f'''Body Force 1
  Stress Bodyforce 3 = Variable Time
    Real MATC "{volume_force_n_m3:.17g}*sin({drive_omega:.17g}*tx)"
End
'''
    else:
        force = np.asarray(nodal_force_n, dtype=float)
        if force.ndim != 1 or not 12 <= len(force) <= 384 or len(force) % 3 or not np.isfinite(force).all():
            raise ValueError('need a finite bounded node-major force vector')
        if probe_node > len(force) // 3:
            raise ValueError('probe node exceeds the nodal force mesh')
        forcing = ''
        for node, components in enumerate(force.reshape(-1, 3), 1):
            if not np.any(components):
                continue
            forcing += f'Boundary Condition {node}\n  Target Nodes(1) = {node}\n'
            for component, value in enumerate(components, 1):
                if value != 0:
                    forcing += f'  Displacement {component} Load = Variable Time\n    Real MATC "{value:.17g}*sin({drive_omega:.17g}*tx)"\n'
            forcing += 'End\n'
    return f'''Header
  CHECK KEYWORDS Warn
  Mesh DB "." "mesh"
End
Simulation
  Max Output Level = 1
  Coordinate System = Cartesian 3D
  Simulation Type = Transient
  Steady State Max Iterations = 1
  Timestepping Method = Bossak
  Bossak Alpha = 0.0
  Timestep Sizes = {dt_s:.17g}
  Timestep Intervals = {steps}
  Output Intervals = 0
End
Body 1
  Equation = 1
  Material = 1
  Initial Condition = 1
{('  Body Force = 1' + chr(10)) if volume_force_n_m3 is not None else ''}End
Equation 1
  Active Solvers(2) = 1 2
End
Material 1
  Density = {material.density_kg_m3:.17g}
  Youngs Modulus = {material.young_pa:.17g}
  Poisson Ratio = {material.poisson:.17g}
End
Initial Condition 1
  Displacement 1 = 0
  Displacement 2 = 0
  Displacement 3 = 0
End
Solver 1
  Equation = StressSolver
  Variable = -dofs 3 Displacement
  Procedure = "StressSolve" "StressSolver"
  Displace Mesh = False
  Linear System Solver = Direct
  Linear System Direct Method = umfpack
  Nonlinear System Max Iterations = 1
End
Solver 2
  Exec Solver = After Timestep
  Equation = SaveScalars
  Procedure = "SaveData" "SaveScalars"
  Filename = "probe.dat"
  Variable 1 = Time
  Operator 1 = value
  Variable 2 = Displacement 3
  Operator 2 = mean
  Variable 3 = Displacement 3
  Operator 3 = min
  Variable 4 = Displacement 3
  Operator 4 = max
  Save Points(1) = {probe_node}
End
{forcing}'''


def read_scalar_probe(directory: Path, *, expected_times, required_names) -> dict:
    """Match named columns and exact requested times; never infer column order."""
    directory = Path(directory)
    data_path, names_path = directory / 'probe.dat', directory / 'probe.dat.names'
    if data_path.stat().st_size > 8 * 1024**2 or names_path.stat().st_size > 256 * 1024:
        raise MemoryError('scalar probe export exceeds its bounded size')
    entries = [re.fullmatch(r'\s*(\d+):\s*(.+?)\s*', line)
               for line in names_path.read_text(encoding='utf-8').splitlines()]
    entries = [(int(match[1]), match[2].casefold()) for match in entries if match]
    if (not entries or [i for i, _ in entries] != list(range(1, len(entries) + 1))
            or len({name for _, name in entries}) != len(entries)):
        raise ValueError('scalar column metadata is missing, duplicate or nonsequential')
    times = np.asarray(expected_times, dtype=float)
    if (times.ndim != 1 or not 1 <= len(times) <= 5000 or not np.isfinite(times).all()
            or (times <= 0).any() or (np.diff(times) <= 0).any()):
        raise ValueError('expected probe times must be finite, positive and increasing')
    values = np.loadtxt(data_path, ndmin=2)
    if values.shape != (len(times), len(entries)) or not np.isfinite(values).all():
        raise ValueError('scalar data dimensions/values disagree with metadata and requested times')
    columns = {name: values[:, index - 1] for index, name in entries}
    time_key = 'value: time scalar variable'
    if time_key not in columns or not np.allclose(columns[time_key], times, rtol=2e-12, atol=1e-18):
        raise ValueError('solver probe time grid differs from the requested grid')
    if any(name not in columns for name in required_names):
        raise ValueError('a required scalar probe column is missing')
    return {name: columns[name].copy() for name in required_names}


def _run_transient(solver, env, mesh_path, output, material, guard, *, dt_s, steps, omega,
                   probe_node, probe_component, force=None, volume=None):
    output.mkdir()
    shutil.copytree(mesh_path, output / 'mesh')
    (output / 'transient.sif').write_text(transient_sif(material, dt_s=dt_s, steps=steps, drive_omega=omega,
                                                     probe_node=probe_node, nodal_force_n=force,
                                                     volume_force_n_m3=volume), encoding='ascii')
    process = run_monitored([str(solver), 'transient.sif'], output, env, guard, output / 'solver.log')
    log = (output / 'solver.log').read_text(encoding='utf-8', errors='replace')
    if process['exit_code'] != 0 or 'ELMER SOLVER FINISHED AT:' not in log:
        raise RuntimeError('Elmer transient did not finish; inspect solver.log')
    keys = [f'value: displacement {probe_component} at node {probe_node}',
            'mean: displacement 3', 'min: displacement 3', 'max: displacement 3']
    times = np.arange(1, steps + 1) * dt_s
    values = read_scalar_probe(output, expected_times=times, required_names=keys)
    return values[keys[0]], values[keys[3]] - values[keys[2]], {
        'process': process, 'dt_s': dt_s, 'steps': steps, 'probe_node': probe_node,
        'probe_component': probe_component, 'probe_column': keys[0],
        'artifact_sha256': {str(p.relative_to(output)).replace('\\', '/'): sha256_file(p)
                            for p in sorted(output.rglob('*')) if p.is_file()}}


def verify_elmer_transient(home: Path, output: Path, guard) -> dict:
    solver, env = elmer_environment(home)
    grid = solver.with_name('ElmerGrid' + solver.suffix)
    if not grid.is_file():
        raise FileNotFoundError('ElmerGrid is required for the quadratic transient case')
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('transient reference needs a new or empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    material = Material()
    plan = {'kind': 'predeclared_elmer_transient_checks', 'plate': asdict(Plate(.04, .03, .003)),
            'cells': [2, 2, 1], 'material': asdict(material), 'orders': [1, 2],
            'steps_per_drive_cycle': [32, 64, 128], 'drive_cycles': 3,
            'uniform_frequency_hz': 100000., 'uniform_force_n_m3': 1e6,
            'modal_acceleration_m_s2': 1000., 'modal_drive_to_natural_frequency_ratio': .7,
            'finest_relative_l2_tolerance': {'rigid-volume': .001, 'rigid-nodal': .001, 'elastic-mode': .005},
            'minimum_observed_time_order': 1.7, 'maximum_observed_time_order': 2.3,
            'equivalent_load_relative_l2_tolerance': 1e-8, 'rigid_spread_relative_tolerance': 1e-9,
            'solver_executable_sha256': sha256_file(solver), 'grid_executable_sha256': sha256_file(grid),
            'source_sha256': {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            'scope': 'Force units, scalar probes and temporal convergence of fixed discrete free-body systems; no continuum spatial, finite-plate scattering, PZT or physical validation.',
            'ready_for_training_data': False}
    plan['fingerprint'] = canonical_hash(plan)
    write_json(output / 'transient-plan.json', plan)
    report = {'kind': 'independent_elmer_transient_checks', 'completed': False, 'passed': False,
              'plan_fingerprint': plan['fingerprint'], 'orders': [], 'ready_for_training_data': False,
              'training_updates': 0, 'scope': plan['scope']}
    started = time.perf_counter()
    try:
        for order in plan['orders']:
            root = output / f'p{order}'
            root.mkdir()
            mesh = structured_mesh(Plate(**plan['plate']), tuple(plan['cells']))
            if order == 1:
                write_elmer_mesh(mesh, root / 'mesh')
                nodes = len(mesh.points_m)
            else:
                write_elmer_mesh(mesh, root / 'linear-mesh')
                process = run_monitored([str(grid), '2', '2', 'linear-mesh', '-increase', '-decimals', '17', '-out', 'mesh'],
                                         root, env, guard, root / 'grid.log')
                if process['exit_code'] != 0:
                    raise RuntimeError('quadratic transient mesh conversion failed')
                nodes = len(read_quadratic_nodes(root / 'mesh'))
            (root / 'operator.sif').write_text(operator_sif(material), encoding='ascii')
            process = run_monitored([str(solver), 'operator.sif'], root, env, guard, root / 'operator.log')
            if process['exit_code'] != 0 or 'Just saved matrix and stopped!' not in (root / 'operator.log').read_text():
                raise RuntimeError('transient reference operator export failed')
            stiffness, mass = read_elmer_operators(root, nodes)
            eigenvalues, eigenvectors = eigh(stiffness, mass)
            if np.max(abs(eigenvalues[:6])) / eigenvalues[-1] >= 1e-10 or eigenvalues[6] <= 0:
                raise ArithmeticError('expected six rigid modes and a positive first elastic mode')
            natural = math.sqrt(eigenvalues[6])
            shape = eigenvectors[:, 6]
            probe_dof = int(abs(shape).argmax())
            shape = shape / shape[probe_dof]
            residual = np.linalg.norm(stiffness @ shape - eigenvalues[6] * mass @ shape) / np.linalg.norm(stiffness @ shape)
            if residual > 1e-9:
                raise ArithmeticError('manufactured elastic mode has excessive eigen residual')
            uniform = np.zeros(nodes * 3)
            uniform[2::3] = plan['uniform_force_n_m3'] / material.density_kg_m3
            uniform_force = mass @ uniform
            modal_force = mass @ shape * plan['modal_acceleration_m_s2']
            np.savez_compressed(root / 'loads.npz', uniform_force_n=uniform_force, modal_force_n=modal_force,
                                elastic_mode=shape, elastic_omega_rad_s=natural)
            rows, saved = [], {}
            for kind in ('rigid-volume', 'rigid-nodal', 'elastic-mode'):
                modal = kind == 'elastic-mode'
                omega = natural * .7 if modal else 2 * math.pi * plan['uniform_frequency_hz']
                amplitude = plan['modal_acceleration_m_s2'] if modal else plan['uniform_force_n_m3'] / material.density_kg_m3
                common_times = np.arange(1, 3 * 32 + 1) * (2 * math.pi / omega / 32)
                exact = sinusoidal_displacement(common_times, amplitude, omega, natural if modal else 0.)
                errors, cases = [], []
                for samples in plan['steps_per_drive_cycle']:
                    factor = samples // 32
                    values, spread, case = _run_transient(solver, env, root / 'mesh', root / f'{kind}-{samples}', material, guard,
                        dt_s=2 * math.pi / omega / samples, steps=3 * samples, omega=omega,
                        probe_node=probe_dof // 3 + 1 if modal else 1, probe_component=probe_dof % 3 + 1 if modal else 3,
                        force=modal_force if modal else uniform_force if kind == 'rigid-nodal' else None,
                        volume=plan['uniform_force_n_m3'] if kind == 'rigid-volume' else None)
                    common = values[factor - 1::factor]
                    error = float(np.linalg.norm(common - exact) / np.linalg.norm(exact))
                    case['relative_l2_error'] = error
                    case['maximum_rigid_spread_relative_to_reference_rms'] = float(spread.max() / np.sqrt(np.mean(exact**2))) if not modal else None
                    errors.append(error)
                    cases.append(case)
                    saved[kind, samples] = common
                observed = [math.log2(a / b) for a, b in zip(errors, errors[1:])] if min(errors) > 0 else []
                passed = (errors[-1] < plan['finest_relative_l2_tolerance'][kind] and len(observed) == 2
                          and all(plan['minimum_observed_time_order'] <= value <= plan['maximum_observed_time_order'] for value in observed)
                          and (modal or max(case['maximum_rigid_spread_relative_to_reference_rms'] for case in cases) < plan['rigid_spread_relative_tolerance']))
                rows.append({'kind': kind, 'passed': passed, 'relative_l2_errors': errors, 'observed_time_orders': observed,
                             'drive_omega_rad_s': omega, 'cases': cases})
            equivalence = [float(np.linalg.norm(saved['rigid-volume', n] - saved['rigid-nodal', n])
                                  / np.linalg.norm(saved['rigid-volume', n])) for n in plan['steps_per_drive_cycle']]
            report['orders'].append({'element_order': order, 'nodes': nodes, 'first_elastic_frequency_hz': natural / (2 * math.pi),
                                      'mode_eigen_residual': float(residual), 'negative_uniform_nodal_coefficients': int((uniform_force < 0).sum()),
                                      'equivalent_load_relative_l2_errors': equivalence,
                                      'passed': all(row['passed'] for row in rows) and max(equivalence) < plan['equivalent_load_relative_l2_tolerance'],
                                      'cases': rows, 'operator_artifact_sha256': {str(p.relative_to(root)).replace('\\', '/'): sha256_file(p)
                                          for p in sorted(root.glob('*')) if p.is_file()}})
        guard.check(force=True)
        report['completed'] = True
        report['passed'] = all(row['passed'] for row in report['orders'])
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter() - started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output / 'transient-report.json', report)
    return report
