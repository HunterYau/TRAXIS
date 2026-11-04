"""Independent P2 elastic guided-mode refinement with Elmer operators.

Periodic lateral faces are used only for the infinite-plate dispersion
benchmark. They must not replace finite physical edges in waveform studies.
"""

from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.linalg import eigh

from traxis.config import Material, Plate, positive
from traxis.dispersion import wavenumber_rad_m
from traxis.elmer_reference import elmer_environment, operator_sif, read_elmer_operators, run_monitored, write_elmer_mesh
from traxis.geometry import structured_mesh
from traxis.storage import canonical_hash, sha256_file, write_json


def sagittal_projection(points, plate: Plate, thickness_cells: int, wavenumber: float):
    """Tie corresponding periodic face nodes; retain interior P2 degrees.

    Only x/z displacement participates. The right x face has exp(i*k*Lx)
    times the left face value, and the y faces are periodic with phase one.
    Interior x/y nodes remain independent, rather than imposing exp(i*k*x)
    throughout the element (which would constrain the interpolation error).
    """
    positive('wavenumber', wavenumber)
    if type(thickness_cells) is not int or not 1 <= thickness_cells <= 4:
        raise ValueError('the bounded P2 cell uses one to four thickness cells')
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not 4 <= len(points) <= 128 or not np.isfinite(points).all():
        raise ValueError('invalid bounded cell nodes')
    scale = np.array([2 / plate.width_m, 2 / plate.height_m, 2 * thickness_cells / plate.thickness_m])
    scaled = points * scale
    integer = np.rint(scaled).astype(np.int64)
    if (not np.allclose(integer, scaled, atol=1e-9, rtol=0) or (integer < 0).any()
            or (integer > [2, 2, 2 * thickness_cells]).any()
            or len(np.unique(integer, axis=0)) != len(points)):
        raise ValueError('cell nodes must remain on the exact P1/P2 reference grid')
    available = {tuple(row) for row in integer}
    for row in integer:
        for axis in (0, 1):
            if row[axis] in (0, 2):
                partner = row.copy()
                partner[axis] = 2 - row[axis]
                if tuple(partner) not in available:
                    raise ValueError('periodic face node has no matching partner')
    canonical = integer.copy()
    canonical[:, :2] %= 2
    keys, inverse = np.unique(canonical, axis=0, return_inverse=True)
    rows = (np.arange(len(points))[:, None] * 3 + np.array([0, 2])).ravel()
    columns = (inverse[:, None] * 2 + np.arange(2)).ravel()
    phases = np.repeat(np.exp(1j * wavenumber * plate.width_m * (integer[:, 0] == 2)), 2)
    return sparse.coo_matrix((phases, (rows, columns)), shape=(len(points) * 3, len(keys) * 2)).tocsr()


def condensed_frequencies(stiffness, mass, projection) -> tuple[np.ndarray, dict]:
    n = projection.shape[0]
    if n > 384 or any(value.shape != (n, n) or not np.isfinite(value).all() for value in (stiffness, mass)):
        raise ValueError('invalid bounded cell operators')
    p = projection.toarray()
    reduced = [p.conj().T @ value @ p for value in (stiffness, mass)]
    symmetry = []
    for value in reduced:
        norm = np.linalg.norm(value)
        error = float(np.linalg.norm(value - value.conj().T) / norm) if norm else math.inf
        if not math.isfinite(error) or error > 1e-12:
            raise ArithmeticError('condensed operator is not Hermitian')
        symmetry.append(error)
    eigenvalues = eigh(*reduced, eigvals_only=True)
    if len(eigenvalues) < 2 or not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0:
        raise ArithmeticError('invalid positive Bloch eigenvalues')
    return np.sqrt(eigenvalues[:2]) / (2 * math.pi), {
        'full_dofs': n, 'condensed_dofs': projection.shape[1], 'hermitian_relative_errors': symmetry}


def read_quadratic_nodes(directory: Path) -> np.ndarray:
    directory = Path(directory)
    for name in ('mesh.nodes', 'mesh.elements', 'mesh.header'):
        if (directory / name).stat().st_size > 1024**2:
            raise MemoryError('quadratic unit-cell mesh exceeds 1 MiB')
    nodes = np.loadtxt(directory / 'mesh.nodes', ndmin=2)
    elements = np.loadtxt(directory / 'mesh.elements', ndmin=2)
    if (nodes.shape[1] != 5 or not 4 <= len(nodes) <= 128 or not np.isfinite(nodes).all()
            or not np.array_equal(nodes[:, 0], np.arange(1, len(nodes) + 1))
            or elements.shape[1] != 13 or not 1 <= len(elements) <= 24
            or not np.isfinite(elements).all() or not np.array_equal(elements, elements.astype(np.int64))
            or not np.array_equal(elements[:, 0], np.arange(1, len(elements) + 1))
            or not (elements[:, 1] == 1).all() or not (elements[:, 2] == 510).all()
            or elements[:, 3:].min() < 1 or elements[:, 3:].max() > len(nodes)
            or len(np.unique(elements[:, 3:])) != len(nodes)):
        raise ValueError('expected a complete bounded quadratic tetrahedron mesh')
    return nodes[:, 2:]


def quadratic_lamb_case(solver: Path, grid: Path, env: dict, output: Path, *,
                        wavenumber: float, thickness_m: float, cells_per_wavelength: int,
                        thickness_cells: int, material: Material, guard) -> dict:
    positive('wavenumber', wavenumber)
    positive('thickness', thickness_m)
    if type(cells_per_wavelength) is not int or not 4 <= cells_per_wavelength <= 64:
        raise ValueError('use 4–64 in-plane cells per wavelength')
    if type(thickness_cells) is not int or not 1 <= thickness_cells <= 4:
        raise ValueError('use 1–4 thickness cells')
    width = 2 * math.pi / wavenumber / cells_per_wavelength
    plate = Plate(width, width, thickness_m)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_elmer_mesh(structured_mesh(plate, (1, 1, thickness_cells)), output / 'linear-mesh')
    conversion = run_monitored([str(grid), '2', '2', 'linear-mesh', '-increase', '-decimals', '17', '-out', 'mesh'],
                                output, env, guard, output / 'grid.log')
    if conversion['exit_code'] != 0:
        raise RuntimeError('ElmerGrid quadratic conversion failed; inspect grid.log')
    points = read_quadratic_nodes(output / 'mesh')
    projection = sagittal_projection(points, plate, thickness_cells, wavenumber)
    (output / 'operator.sif').write_text(operator_sif(material), encoding='ascii')
    process = run_monitored([str(solver), 'operator.sif'], output, env, guard, output / 'solver.log')
    log = (output / 'solver.log').read_text(encoding='utf-8', errors='replace')
    if process['exit_code'] != 0 or 'Just saved matrix and stopped!' not in log:
        raise RuntimeError('Elmer did not finish quadratic assembly; inspect solver.log')
    stiffness, mass = read_elmer_operators(output, len(points))
    frequencies, diagnostics = condensed_frequencies(stiffness, mass, projection)
    guard.check(force=True)
    return {'cells_per_wavelength': cells_per_wavelength, 'thickness_cells': thickness_cells,
            'cell': asdict(plate), 'wavenumber_rad_m': wavenumber, 'nodes': len(points),
            'fundamental_frequencies_hz': frequencies.tolist(), **diagnostics,
            'mesh_conversion': conversion, 'operator_export': process,
            'solver_banner': [line for line in log.splitlines() if 'MAIN: Version:' in line],
            'artifact_sha256': {str(p.relative_to(output)).replace('\\', '/'): sha256_file(p)
                                for p in sorted(output.rglob('*')) if p.is_file()}}


def verify_elmer_waves(home: Path, output: Path, guard, *, frequencies_hz=(100000., 200000.),
                       thicknesses_m=(.001, .003)) -> dict:
    frequencies_hz, thicknesses_m = tuple(frequencies_hz), tuple(thicknesses_m)
    if (not frequencies_hz or not thicknesses_m or len(set(frequencies_hz)) != len(frequencies_hz)
            or len(set(thicknesses_m)) != len(thicknesses_m)
            or len(frequencies_hz) * len(thicknesses_m) > 8):
        raise ValueError('provide unique frequencies/thicknesses with at most eight combinations')
    for value in (*frequencies_hz, *thicknesses_m):
        positive('wave benchmark input', value)
    material = Material()
    # Resolve all independent traction roots before any external solve.
    targets = [(frequency, thickness, mode, index, wavenumber_rad_m(frequency, thickness, mode, material))
               for frequency in frequencies_hz for thickness in thicknesses_m
               for mode, index in (('A0', 0), ('S0', 1))]
    solver, env = elmer_environment(home)
    grid = solver.with_name('ElmerGrid' + solver.suffix)
    if not grid.is_file():
        raise FileNotFoundError('ElmerGrid is required for quadratic mesh conversion')
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('guided-mode reference needs a new or empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    plan = {'kind': 'predeclared_elmer_p2_guided_refinement', 'frequencies_hz': frequencies_hz,
            'thicknesses_m': thicknesses_m, 'material': asdict(material), 'in_plane_levels': [4, 8, 16],
            'thickness_levels': [1, 2, 4], 'maximum_final_relative_frequency_error': .005,
            'required_strict_error_decrease': True, 'operator_exports': len(targets) * 5,
            'solver_executable_sha256': sha256_file(solver), 'grid_executable_sha256': sha256_file(grid),
            'source_sha256': {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            'scope': 'Lossless infinite-plate sagittal A0/S0 dispersion only; no finite-edge, defect, transient, PZT or physical validation.',
            'ready_for_training_data': False}
    plan['fingerprint'] = canonical_hash(plan)
    write_json(output / 'guided-plan.json', plan)
    report = {'kind': 'independent_elmer_p2_guided_refinement', 'completed': False, 'passed': False,
              'ready_for_training_data': False, 'training_updates': 0, 'plan_fingerprint': plan['fingerprint'],
              'scope': plan['scope'], 'targets': []}
    started = time.perf_counter()
    try:
        for number, (frequency, thickness, mode, index, k) in enumerate(targets):
            target = output / f'target-{number:02d}'
            target.mkdir()
            cases = []
            # Spatial x/y refinement holds z fixed; z refinement holds x/y fixed.
            for case_index, (n, nz) in enumerate(((4, 4), (8, 4), (16, 4), (16, 1), (16, 2))):
                case = quadratic_lamb_case(solver, grid, env, target / f'case-{case_index:02d}',
                                           wavenumber=k, thickness_m=thickness, cells_per_wavelength=n,
                                           thickness_cells=nz, material=material, guard=guard)
                case['relative_frequency_error'] = abs(case['fundamental_frequencies_hz'][index] / frequency - 1)
                cases.append(case)
            xy = [cases[i]['relative_frequency_error'] for i in (0, 1, 2)]
            z = [cases[i]['relative_frequency_error'] for i in (3, 4, 2)]
            passed = (xy[-1] < plan['maximum_final_relative_frequency_error']
                      and all(a > b for values in (xy, z) for a, b in zip(values, values[1:])))
            report['targets'].append({'frequency_hz': frequency, 'thickness_m': thickness, 'mode': mode,
                                       'in_plane_relative_errors': xy, 'thickness_relative_errors': z,
                                       'passed': passed, 'cases': cases})
        guard.check(force=True)
        report['completed'] = True
        report['passed'] = all(target['passed'] for target in report['targets'])
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter() - started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output / 'guided-report.json', report)
    return report


def plot_elmer_waves(directory: Path, output: Path, guard=None) -> dict:
    """Plot independent refinement axes from fingerprint-checked reports."""
    from traxis.reporting import _figure_api, _new_figure, _save_figure
    directory, output = Path(directory).resolve(), Path(output)
    paths = [directory / name for name in ('guided-report.json', 'guided-plan.json')]
    if any(path.stat().st_size > 16 * 1024**2 for path in paths):
        raise MemoryError('guided-mode report/plan exceeds 16 MiB')
    report, plan = [json.loads(path.read_text(encoding='utf-8-sig')) for path in paths]
    for value in (report, plan):
        if canonical_hash({k: v for k, v in value.items() if k != 'fingerprint'}) != value['fingerprint']:
            raise ValueError('guided-mode report/plan changed')
    if not report['completed'] or report['plan_fingerprint'] != plan['fingerprint']:
        raise ValueError('guided-mode report is incomplete or refers to another plan')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('guided-mode figures need a new or empty directory')
    output.mkdir(parents=True, exist_ok=True)
    api = _figure_api(output / '.matplotlib-cache')
    figure = _new_figure(api, (13, 8))
    axes = figure.subplots(2, 2, squeeze=False)
    for row, mode in enumerate(('A0', 'S0')):
        for target in report['targets']:
            if target['mode'] != mode:
                continue
            label = f"{target['frequency_hz'] / 1000:g} kHz, {target['thickness_m'] * 1000:g} mm"
            for column, (levels, errors) in enumerate((
                (plan['in_plane_levels'], target['in_plane_relative_errors']),
                (plan['thickness_levels'], target['thickness_relative_errors']),
            )):
                if len(errors) != 3 or not np.isfinite(errors).all() or any(v < 0 for v in errors):
                    raise ValueError('invalid guided-mode refinement errors')
                axes[row, column].plot(levels, np.maximum(np.array(errors) * 100, 1e-12), 'o-', label=label)
        for column, (title, levels, xlabel) in enumerate((
            ('In-plane refinement, 4 thickness cells', plan['in_plane_levels'], 'Cells per wavelength in x/y'),
            ('Thickness refinement, 16 cells/wavelength', plan['thickness_levels'], 'Cells through thickness'),
        )):
            axis = axes[row, column]
            axis.axhline(plan['maximum_final_relative_frequency_error'] * 100, color='0.4', linestyle='--', linewidth=1)
            axis.set_xscale('log', base=2)
            axis.set_yscale('log')
            axis.set_xticks(levels, [str(value) for value in levels])
            axis.set(title=f'{mode}: {title}', xlabel=xlabel, ylabel='Frequency error relative to traction root (%)')
            axis.grid(alpha=.2, which='both')
            axis.legend(fontsize=8)
    figure.suptitle('P2 infinite-plate dispersion — strict refinement screen ' + ('PASSED' if report['passed'] else 'FAILED'), fontsize=13)
    figure.supxlabel(f"Independent Elmer operators vs Rayleigh–Lamb. Dashed: {plan['maximum_final_relative_frequency_error'] * 100:g}% finest-grid screen. Display floor: 1e-12%. No finite-plate/PZT/physical validation.", fontsize=8)
    files = _save_figure(figure, output, 'elmer-p2-guided-refinement', guard)
    result = {'kind': 'elmer_p2_guided_refinement_figures', 'files': files,
              'guided_report_sha256': sha256_file(paths[0]), 'guided_plan_sha256': sha256_file(paths[1]),
              'file_sha256': {name: sha256_file(output / name) for name in files},
              'ready_for_training_data': False, 'resources': guard.report() if guard else None}
    write_json(output / 'figures.json', result)
    return result
