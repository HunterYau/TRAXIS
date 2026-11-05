"""Independent Elmer assembly checks on tiny free 3D tetrahedral meshes.

Elmer owns the reference stiffness and consistent mass assembly. These checks
cannot establish continuum wave accuracy, transducer fidelity or physical fit.
No software is downloaded or installed by this module.
"""

from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import psutil
from scipy import sparse
from scipy.linalg import eigh

from traxis.config import Material, Plate
from traxis.fem import assemble
from traxis.geometry import TetMesh, structured_mesh
from traxis.resources import ResourceGuard, ResourceLimitError
from traxis.storage import canonical_hash, sha256_file, write_json


def write_elmer_mesh(mesh: TetMesh, directory: Path, *, max_nodes=128, max_tetrahedra=512) -> None:
    """Write bounded linear tets with free natural boundaries, in meters."""
    if (type(max_nodes) is not int or not 4 <= max_nodes <= 20000
            or type(max_tetrahedra) is not int or not 1 <= max_tetrahedra <= 40000):
        raise ValueError('invalid native mesh export bounds')
    points, tets = np.asarray(mesh.points_m), np.asarray(mesh.tetrahedra)
    if (points.ndim != 2 or points.shape[1] != 3 or not 4 <= len(points) <= max_nodes
            or not np.isfinite(points).all() or tets.ndim != 2 or tets.shape[1] != 4
            or tets.dtype.kind not in 'iu' or not 1 <= len(tets) <= max_tetrahedra
            or tets.min() < 0 or tets.max() >= len(points)
            or len(np.unique(tets)) != len(points)):
        raise ValueError("reference mesh exceeds declared bounds or has invalid/unused nodes")
    tets = tets.copy()
    edges = points[tets[:, 1:]] - points[tets[:, :1]]
    determinant = np.linalg.det(edges)
    if np.any(abs(determinant) <= 1e-12 * np.linalg.norm(edges, axis=2).max(axis=1)**3):
        raise ValueError("reference mesh contains degenerate tetrahedra")
    negative = determinant < 0
    tets[negative, :2] = tets[negative, 1::-1]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'mesh.header').write_text(f'{len(points)} {len(tets)} 0\n1\n504 {len(tets)}\n', encoding='ascii')
    (directory / 'mesh.nodes').write_text(''.join(
        f'{i + 1} -1 {x:.17g} {y:.17g} {z:.17g}\n' for i, (x, y, z) in enumerate(points)), encoding='ascii')
    (directory / 'mesh.elements').write_text(''.join(
        f'{i + 1} 1 504 ' + ' '.join(str(v + 1) for v in tet) + '\n'
        for i, tet in enumerate(tets)), encoding='ascii')
    (directory / 'mesh.boundary').write_text('', encoding='ascii')


def operator_sif(material: Material, *, relative_integration_order=1) -> str:
    """Export before constraints/scaling/eigen solution; retain consistent mass.

    The save flags and format are implemented in Elmer's SolverUtils.F90;
    some are absent from that release's keyword warning dictionary.
    """
    if type(relative_integration_order) is not int or relative_integration_order not in (0, 1):
        raise ValueError('relative integration order must be zero or one')
    # P2 mass products have degree four. The release's default 27-point
    # Duffy rule is not exact for them; the higher 64-point rule is.
    return f'''Header
  CHECK KEYWORDS Warn
  Mesh DB "." "mesh"
End
Simulation
  Max Output Level = 5
  Coordinate System = Cartesian 3D
  Simulation Type = Steady State
  Steady State Max Iterations = 1
  Output Intervals = 0
End
Body 1
  Target Bodies(1) = 1
  Equation = 1
  Material = 1
End
Equation 1
  Active Solvers(1) = 1
End
Solver 1
  Equation = StressSolver
  Variable = -dofs 3 Displacement
  Procedure = "StressSolve" "StressSolver"
  Eigen Analysis = Logical True
  Eigen System Values = 10
  Displace Mesh = Logical False
  Relative Integration Order = Integer {relative_integration_order}
  Optimize Bandwidth = Logical False
  Linear System Save = Logical True
  Linear System Save Slot = String "assembly"
  Linear System Save Mass = Logical True
  Linear System Save Perm = Logical True
  Linear System Save And Stop = Logical True
  Linear System Save Prefix = String "operator"
End
Material 1
  Density = Real {material.density_kg_m3:.17g}
  Youngs Modulus = Real {material.young_pa:.17g}
  Poisson Ratio = Real {material.poisson:.17g}
End
'''


def elmer_environment(home: Path) -> tuple[Path, dict]:
    home = Path(home).resolve()
    executable = home / 'bin' / ('ElmerSolver.exe' if os.name == 'nt' else 'ElmerSolver')
    if not executable.is_file():
        raise FileNotFoundError(f'ElmerSolver executable is missing under {home / "bin"}')
    env = os.environ.copy()
    env.update({'ELMER_HOME': str(home), 'ELMER_LIB': str(home / 'share/elmersolver/lib'),
                'PATH': str(home / 'bin') + os.pathsep + str(home / 'lib') + os.pathsep + env.get('PATH', ''),
                'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    return executable, env


def _stop_tree(process):
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.terminate()
    _, alive = psutil.wait_procs(children, timeout=2)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def run_monitored(command, directory: Path, env: dict, guard: ResourceGuard, log: Path) -> dict:
    """Monitor parent plus external process tree; terminate it on a guard stop."""
    guard.check(force=True)
    started = time.perf_counter()
    peak_processes = 0
    with Path(log).open('xb') as output:
        process = subprocess.Popen(command, cwd=directory, env=env, stdout=output, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            while True:
                guard.check(force=True)
                sample = guard.history[-1]
                try:
                    root = psutil.Process(process.pid)
                    tree = [root, *root.children(recursive=True)]
                    peak_processes = max(peak_processes, len(tree))
                    for child in tree:
                        try:
                            sample['process_rss_mib'] += child.memory_info().rss / 1024**2
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass
                # An access-denied measurement is an error, not a zero-RAM claim.
                guard.enforce(sample, guard.last_gpus)
                if Path(log).stat().st_size > 4 * 1024**2:
                    raise ResourceLimitError('external solver log exceeds 4 MiB')
                code = process.poll()
                if code is not None:
                    break
                time.sleep(.2)
        except BaseException:
            _stop_tree(process)
            raise
    return {'exit_code': code, 'wall_seconds': time.perf_counter() - started,
            'maximum_observed_external_processes': peak_processes,
            'memory_scope': 'TRAXIS monitor plus external executable and all observed descendants'}


def read_elmer_operators(directory: Path, nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Read bounded serial triplets and undo Elmer's node permutation."""
    if type(nodes) is not int or not 4 <= nodes <= 128:
        raise ValueError('operator reader is limited to 4–128 nodes')
    directory = Path(directory)
    for name in ('operator_a.dat', 'operator_perm.dat', 'operator_sizes.dat'):
        if (directory / name).stat().st_size > 16 * 1024**2:
            raise MemoryError('Elmer operator export exceeds 16 MiB')
    entries = np.loadtxt(directory / 'operator_a.dat', ndmin=2)
    permutation = np.loadtxt(directory / 'operator_perm.dat', ndmin=2)
    sizes = np.loadtxt(directory / 'operator_sizes.dat', ndmin=1)
    ndof = nodes * 3
    if (entries.ndim != 2 or entries.shape[1] != 4 or not np.isfinite(entries).all()
            or permutation.shape != (nodes, 2) or not np.isfinite(permutation).all()
            or sizes.shape != (5,) or not np.isfinite(sizes).all()
            or not np.array_equal(sizes, sizes.astype(np.int64))
            or not np.array_equal(sizes[:3], [ndof, len(entries), nodes])):
        raise ValueError('invalid Elmer stiffness/mass/permutation export')
    if (not np.array_equal(permutation, permutation.astype(np.int64))
            or not np.array_equal(permutation[:, 0], np.arange(1, nodes + 1))
            or not np.array_equal(np.sort(permutation[:, 1]), np.arange(1, nodes + 1))):
        raise ValueError('Elmer node permutation is not a complete bijection')
    indexes = entries[:, :2]
    if (not np.array_equal(indexes, indexes.astype(np.int64)) or indexes.min() < 1 or indexes.max() > ndof
            or not np.array_equal(sizes[3:], [indexes[:, 1].min(), indexes[:, 1].max()])):
        raise ValueError('invalid Elmer matrix indexes')
    indexes = indexes.astype(np.int64) - 1
    if len(np.unique(indexes[:, 0] * ndof + indexes[:, 1])) != len(indexes):
        raise ValueError('duplicate Elmer operator entry')
    order = (3 * (permutation[:, 1].astype(np.int64) - 1)[:, None] + np.arange(3)).ravel()
    result = []
    for column in (2, 3):
        matrix = sparse.coo_matrix((entries[:, column], indexes.T), shape=(ndof, ndof)).toarray()
        result.append(matrix[np.ix_(order, order)])
    return tuple(result)


def compare_operators(mesh: TetMesh, material: Material, stiffness, mass, guard=None) -> dict:
    local = assemble(mesh, material, guard)
    ndof = local.ndof
    if ndof > 384:
        raise MemoryError('dense reference diagnostics are limited to 384 degrees of freedom')
    if any(v.shape != (ndof, ndof) or not np.isfinite(v).all() for v in (stiffness, mass)):
        raise ValueError('reference operators have invalid shape/values')
    analytic_mass = np.zeros_like(mass)
    # Exact integral of linear barycentric products: rho*V*(ones+I)/20.
    for tet in mesh.tetrahedra:
        points = mesh.points_m[tet]
        volume = abs(np.linalg.det(points[1:] - points[0])) / 6
        block = material.density_kg_m3 * volume / 20 * (np.ones((4, 4)) + np.eye(4))
        for coordinate in range(3):
            dofs = 3 * tet + coordinate
            analytic_mass[np.ix_(dofs, dofs)] += block
    def relative(candidate, reference):
        return float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))
    stiffness_error = relative(stiffness, local.stiffness.toarray())
    mass_error = relative(mass, analytic_mass)
    lumped_error = relative(mass.sum(axis=1), local.mass_kg)
    symmetry = [relative(v, v.T) for v in (stiffness, mass)]
    if max(symmetry) > 1e-12:
        raise ArithmeticError('reference operators are not symmetric')
    # Eigh assumes symmetry; check it first. A non-SPD mass fails here.
    eigenvalues = eigh(stiffness, mass, eigvals_only=True)
    if eigenvalues[-1] <= 0:
        raise ArithmeticError('reference elastic operator has no positive eigenvalue')
    normalized = eigenvalues / eigenvalues[-1]
    six_rigid = bool(np.max(abs(normalized[:6])) < 1e-10 and normalized[6] > 1e-10)
    passed = max(stiffness_error, mass_error, lumped_error) < 1e-10 and six_rigid
    return {'passed': passed, 'nodes': len(mesh.points_m), 'tetrahedra': len(mesh.tetrahedra), 'dofs': ndof,
            'material': asdict(material), 'stiffness_relative_frobenius': stiffness_error,
            'consistent_mass_relative_frobenius': mass_error, 'row_summed_mass_relative_l2': lumped_error,
            'symmetry_relative_frobenius': symmetry, 'smallest_six_eigenvalues_over_largest': normalized[:6].tolist(),
            'seventh_eigenvalue_over_largest': float(normalized[6]), 'six_rigid_modes': six_rigid,
            'relative_matrix_tolerance': 1e-10, 'relative_rigid_eigenvalue_tolerance': 1e-10}


def verify_elmer(home: Path, output: Path, guard: ResourceGuard) -> dict:
    executable, env = elmer_environment(home)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('independent reference checks require a new or empty directory')
    output.mkdir(parents=True, exist_ok=True)
    report = {'kind': 'independent_elmer_operator_verification', 'completed': False, 'passed': False, 'ready_for_training_data': False,
              'training_updates': 0, 'cases': [], 'executable_sha256': sha256_file(executable),
              'source_sha256': {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
              'scope': 'Independent assembly of identical coarse 3D P1 tetrahedra. No continuum wave, PZT or physical validation.'}
    module = Path(env['ELMER_LIB']) / ('StressSolve.dll' if os.name == 'nt' else 'StressSolve.so')
    if module.exists():
        report['stress_module_sha256'] = sha256_file(module)
    started = time.perf_counter()
    try:
        for name, transform, material in (
            ('rectangular', np.eye(3), Material()),
            ('affine-skewed', np.array([[1., .12, .04], [.05, .9, .03], [.01, .02, 1.1]]),
             Material(young_pa=71e9, poisson=.29, density_kg_m3=2800.)),
        ):
            guard.check(force=True)
            original = structured_mesh(Plate(.04, .03, .003), (2, 2, 1))
            mesh = TetMesh(original.points_m @ transform.T, original.tetrahedra, original.plate, 0., 0.)
            case = output / name
            case.mkdir()
            write_elmer_mesh(mesh, case / 'mesh')
            (case / 'operator.sif').write_text(operator_sif(material), encoding='ascii')
            process = run_monitored([str(executable), 'operator.sif'], case, env, guard, case / 'solver.log')
            log = (case / 'solver.log').read_text(encoding='utf-8', errors='replace')
            if process['exit_code'] != 0 or 'Just saved matrix and stopped!' not in log:
                raise RuntimeError('Elmer did not finish the requested assembly export; inspect solver.log')
            stiffness, mass = read_elmer_operators(case, len(mesh.points_m))
            result = compare_operators(mesh, material, stiffness, mass, guard)
            result.update({'name': name, 'affine_transform': transform.tolist(), 'process': process,
                           'artifact_sha256': {str(p.relative_to(case)).replace('\\', '/'): sha256_file(p)
                                               for p in sorted(case.rglob('*')) if p.is_file()},
                           'solver_banner': [line for line in log.splitlines() if 'MAIN: Version:' in line]})
            report['cases'].append(result)
        guard.check(force=True)
        report['completed'] = True
        report['passed'] = all(case['passed'] for case in report['cases'])
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter() - started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output / 'elmer-report.json', report)
    return report
