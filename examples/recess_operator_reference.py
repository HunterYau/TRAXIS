"""Mapped recess elasticity against native element and global applications."""
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
from examples.tensor_wave import PROJECT, checked_receipt, identities, read_json, sealed, snapshot_sources
from traxis.config import CircularRecess, Material, Plate
from traxis.elmer_reference import run_monitored
from traxis.highorder_verification import distribution_identity
from traxis.mapped_tensor import _local_majorant, action, assemble_bounded, build_system, local_action
from traxis.recess_hex import make_recess_mesh
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import sha256_file, write_json
from traxis.tensor_hex import line_rule


def _dense_local(system, element):
    n = 3*(system.mesh.order+1)**3
    matrix = np.empty((n, n))
    for first in range(0, n, 32):
        last = min(first+32, n); values = np.zeros((last-first, n))
        values[np.arange(last-first), np.arange(first, last)] = 1.
        matrix[:, first:last] = local_action(values, system.mesh.order, system.inverse_jacobians[element],
            system.volume_weights_m3[element], system.material).T
    return matrix


def compare_native(system, model, geometry_details, output, guard):
    import ngsolve as ng
    mesh, material = system.mesh, system.material
    native = model['mesh']; ns = len(mesh.points_m); p = mesh.order; n = (p+1)**3
    scalar = ng.H1(native, order=p); space = ng.FESpace([scalar]*3)
    if scalar.ndof != ns or space.ndof != 3*ns:
        raise ArithmeticError('native displacement topology differs')
    u, v = space.TnT(); du, dv = ([ng.grad(f[c]) for c in range(3)] for f in (u, v))
    lam, mu = material.lame
    stiffness_expression = mu/2*sum((du[i][j]+du[j][i])*(dv[i][j]+dv[j][i])
        for i in range(3) for j in range(3)) + lam*sum(du[i][i] for i in range(3))*sum(dv[i][i] for i in range(3))
    mass_expression = material.density_kg_m3*sum(u[i]*v[i] for i in range(3))
    nodes, weights, *_ = line_rule(p)
    reference = np.stack(np.meshgrid(*[(nodes+1)/2]*3, indexing='ij'), axis=-1).reshape(-1, 3)
    rule = ng.IntegrationRule(points=reference.tolist(), weights=(np.kron(np.kron(weights, weights), weights)/8).tolist())
    integrators, forms = [], []
    for expression in (stiffness_expression, mass_expression):
        integrator = ng.SymbolicBFI(expression); integrator.SetIntegrationRule(ng.ET.HEX, rule)
        form = ng.BilinearForm(space, nonassemble=True); form += integrator
        integrators.append(integrator); forms.append(form)
    gradient = np.array([[.13, -.07, .02], [.04, .09, -.03], [-.05, .08, .11]])
    rng = np.random.default_rng(3149)
    fields = [rng.normal(size=3*ns)*1e-9 for _ in range(2)]
    fields.append((mesh.points_m@gradient.T).T.ravel())
    for axis in np.eye(3):
        fields.extend([np.broadcast_to(axis, mesh.points_m.shape).T.ravel(),
                       np.cross(axis, mesh.points_m-mesh.points_m.mean(axis=0)).T.ravel()])
    fields = np.asarray(fields); coefficients = np.zeros((len(fields), 3, ns))
    seen_coefficients = np.zeros(ns, dtype=bool); transform_rows = [None]*ns
    global_native = np.zeros((3*ns, 3*ns)) if 3*ns <= 2000 else None
    rows, files, shared_error, coefficient_error = [], [], 0., 0.
    (output/'elements').mkdir()
    for element in native.Elements(ng.VOL):
        guard.check(); e = int(element.nr); eid = ng.ElementId(ng.VOL, e)
        ids = mesh.element_nodes[e]; dofs = np.asarray(scalar.GetDofNrs(eid), dtype=int)
        vector_dofs = np.asarray(space.GetDofNrs(eid), dtype=int)
        if len(dofs) != n or not np.array_equal(vector_dofs, np.concatenate([dofs+c*ns for c in range(3)])):
            raise ArithmeticError('native component/local basis ordering differs')
        fe, trafo = scalar.GetFE(eid), native.GetTrafo(eid)
        basis = np.empty((n, n)); seen = np.zeros(n, dtype=bool)
        jacobians, physical = np.empty((n, 3, 3)), np.empty((n, 3))
        frame = np.asarray(model['reference_frames'][e]); rotation = np.rint(frame[1:].T)
        if np.max(abs(frame[1:].T-rotation)) > 1e-12 or abs(np.linalg.det(rotation)-8) > 1e-10:
            raise ArithmeticError('native reference frame is not a positive cube permutation')
        for point, ip in zip(reference, rule):
            mip = trafo(ip); measured = np.array(mip.point)
            distance = np.linalg.norm(mesh.points_m[ids]-measured, axis=1); local = int(distance.argmin())
            if distance[local] > 1e-12 or seen[local]:
                raise ArithmeticError('native GLL positions differ or repeat')
            basis[local] = fe.CalcShape(*point); seen[local] = True
            physical[local] = measured; jacobians[local] = np.array(mip.jacobi) @ np.linalg.inv(rotation)
        if not seen.all():
            raise ArithmeticError('native local nodal transform is incomplete')
        inverse = np.linalg.inv(basis)
        for node, shape in zip(ids, basis):
            nonzero = shape != 0.; indices, values = dofs[nonzero], shape[nonzero]
            sorting = np.argsort(indices); indices, values = indices[sorting], values[sorting]
            if transform_rows[node] is None:
                transform_rows[node] = indices, values
            else:
                old_indices, old_values = transform_rows[node]
                union = np.union1d(indices, old_indices); a, b = np.zeros(len(union)), np.zeros(len(union))
                a[np.searchsorted(union, indices)] = values; b[np.searchsorted(union, old_indices)] = old_values
                shared_error = max(shared_error, float(np.max(abs(a-b))))
        local_coefficients = np.einsum('ij,bcj->bci', inverse, fields.reshape((-1, 3, ns))[:, :, ids])
        for local, dof in enumerate(dofs):
            if seen_coefficients[dof]:
                difference = abs(coefficients[:, :, dof]-local_coefficients[:, :, local])
                scale = np.max(abs(fields), axis=1)
                coefficient_error = max(coefficient_error, float(np.max(difference/scale[:, None])))
            else:
                coefficients[:, :, dof] = local_coefficients[:, :, local]; seen_coefficients[dof] = True
        native_hierarchical = np.asarray(integrators[0].CalcElementMatrix(space.GetFE(eid), trafo, heapsize=64*1024**2))
        native_mass = np.asarray(integrators[1].CalcElementMatrix(space.GetFE(eid), trafo, heapsize=64*1024**2))
        native_nodal = np.empty((3*n, 3*n))
        for c in range(3):
            for d in range(3):
                native_nodal[c*n:(c+1)*n, d*n:(d+1)*n] = inverse.T @ native_hierarchical[c*n:(c+1)*n, d*n:(d+1)*n] @ inverse
        mapped = _dense_local(system, e)
        mass_scalar = native_mass[:n, :n]
        native_nodal_mass = inverse.T @ mass_scalar @ inverse
        local_mass = material.density_kg_m3*system.volume_weights_m3[e]
        block_mass_error = 0.
        for c in range(3):
            for d in range(3):
                block = native_mass[c*n:(c+1)*n, d*n:(d+1)*n]
                expected = mass_scalar if c == d else np.zeros_like(mass_scalar)
                block_mass_error = max(block_mass_error, float(np.linalg.norm(block-expected)/np.linalg.norm(mass_scalar)))
        k_error = float(np.linalg.norm(mapped-native_nodal)/np.linalg.norm(native_nodal))
        m_error = float(np.linalg.norm(native_nodal_mass-np.diag(local_mass))/np.linalg.norm(local_mass))
        majorant = _local_majorant(p, system.inverse_jacobians[e], system.volume_weights_m3[e], material)
        local_majorant = np.tile(majorant, 3)
        majorant_minimum = float(eigh(np.diag(local_majorant)-native_nodal, eigvals_only=True,
                                      subset_by_index=(0, 0))[0]/max(local_majorant))
        mass_vector = np.tile(local_mass, 3); upper = float(np.max(local_majorant/mass_vector))
        eigenvalues = eigh(native_nodal, np.diag(mass_vector), eigvals_only=True, subset_by_index=(0, 6))
        only_six = bool(np.max(abs(eigenvalues[:6]/upper)) < 1e-12 and eigenvalues[6]/upper > 1e-12)
        passed = bool(max(k_error, m_error, block_mass_error) < 1e-10 and majorant_minimum > -1e-12 and only_six)
        rows.append(dict(element=e, stiffness_relative_l2=k_error, mass_relative_l2=m_error,
            native_mass_component_error=block_mass_error, diagonal_majorant_minimum_normalized_eigenvalue=majorant_minimum,
            only_six_rigid_modes=only_six, smallest_generalized_eigenvalues=eigenvalues.tolist(),
            local_eigenvalue_upper_s2=upper, passed=passed))
        path = output/'elements'/f'e{e:04d}.npz'
        np.savez_compressed(path, basis=basis, native_stiffness_hierarchical=native_hierarchical,
            native_stiffness_nodal=native_nodal, mapped_stiffness=mapped, native_mass_scalar_hierarchical=mass_scalar,
            mass_scalar=local_mass, majorant_scalar=majorant, native_gll_jacobians=jacobians,
            native_gll_points=physical, connectivity=ids)
        files.append(dict(path=str(path.relative_to(output)), sha256=sha256_file(path)))
        if global_native is not None:
            indices = np.concatenate([ids+c*ns for c in range(3)])
            global_native[np.ix_(indices, indices)] += native_nodal
        write_json(output/'progress.json', dict(completed=False, elements_completed=e+1, elements_planned=len(mesh.element_nodes)), overwrite=True)
    if not seen_coefficients.all() or any(r is None for r in transform_rows) or max(shared_error, coefficient_error) > 1e-10:
        raise ArithmeticError('native connected coefficients or shared scalar traces differ')
    pointers = np.r_[0, np.cumsum([len(row[0]) for row in transform_rows])]
    transform = sparse.csr_matrix((np.concatenate([r[1] for r in transform_rows]),
        np.concatenate([r[0] for r in transform_rows]), pointers), shape=(ns, ns))
    coefficients = coefficients.reshape((-1, 3*ns)); mapped_actions = action(system, fields, guard)
    native_actions, native_mass_actions = [], []
    gridfunction = ng.GridFunction(space); vector = gridfunction.vec.CreateVector()
    for field in coefficients:
        guard.check(); gridfunction.vec.FV().NumPy()[:] = field
        forms[0].Apply(gridfunction.vec, vector); native_actions.append(np.array(vector.FV().NumPy(), copy=True))
        forms[1].Apply(gridfunction.vec, vector); native_mass_actions.append(np.array(vector.FV().NumPy(), copy=True))
    native_actions, native_mass_actions = np.asarray(native_actions), np.asarray(native_mass_actions)
    def dual_transform(values):
        return np.array([[transform.T @ c for c in row.reshape(3, ns)] for row in values]).reshape((-1, 3*ns))
    dual_actions = dual_transform(mapped_actions); dual_mass = dual_transform(fields*system.mass_kg)
    action_errors = [float(np.linalg.norm(a-b)/np.linalg.norm(b)) for a, b in zip(dual_actions[:3], native_actions[:3])]
    mass_errors = [float(np.linalg.norm(a-b)/np.linalg.norm(b)) for a, b in zip(dual_mass, native_mass_actions)]
    rigid_error = float(np.linalg.norm(mapped_actions[3:])/(np.linalg.norm(system.stiffness_majorant_kg_s2)*np.linalg.norm(fields[3:])))
    strain = (gradient+gradient.T)/2
    expected_energy = geometry_details['native_volume_m3']*(mu*np.sum(strain*strain)+lam*np.trace(strain)**2/2)
    energies = [float(fields[2]@mapped_actions[2]/2), float(coefficients[2]@native_actions[2]/2)]
    energy_error = max(abs(value/expected_energy-1) for value in energies)
    full = None
    if global_native is not None:
        assembled = assemble_bounded(system, guard)
        discrepancy = float(np.linalg.norm(assembled-global_native)/np.linalg.norm(global_native))
        upper = system.eigenvalue_upper_s2
        eigenvalues = eigh(global_native, np.diag(system.mass_kg), eigvals_only=True, subset_by_index=(0, 6))
        largest = float(eigh(global_native, np.diag(system.mass_kg), eigvals_only=True,
                            subset_by_index=(3*ns-1, 3*ns-1))[0])
        full = dict(all_column_relative_l2=discrepancy, largest_native_eigenvalue_s2=largest,
            smallest_native_eigenvalues=eigenvalues.tolist(),
            only_six_rigid_modes=bool(np.max(abs(eigenvalues[:6]/upper)) < 1e-12 and eigenvalues[6]/upper > 1e-12),
            stable_dt_from_native_s=2/math.sqrt(largest), bound_to_native_eigenvalue_ratio=upper/largest)
        np.savez_compressed(output/'global-matrices.npz', native_stiffness=global_native, mapped_stiffness=assembled,
            mass=system.mass_kg, majorant=system.stiffness_majorant_kg_s2, smallest_eigenvalues=eigenvalues)
        full['matrices_sha256'] = sha256_file(output/'global-matrices.npz')
        if discrepancy >= 1e-10 or largest > upper*(1+1e-12) or not full['only_six_rigid_modes']:
            raise ArithmeticError('connected all-column or native spectrum check failed')
    np.savez_compressed(output/'global-actions.npz', nodal_states=fields, native_coefficients=coefficients,
        mapped_actions=mapped_actions, mapped_actions_native_dual=dual_actions, native_actions=native_actions,
        mapped_mass_native_dual=dual_mass, native_mass_actions=native_mass_actions,
        mass=system.mass_kg, majorant=system.stiffness_majorant_kg_s2,
        transform_data=transform.data, transform_indices=transform.indices, transform_indptr=transform.indptr)
    return dict(passed=bool(all(r['passed'] for r in rows) and max(action_errors+mass_errors+[energy_error]) < 1e-10 and rigid_error < 1e-12),
        elements=rows, element_files=files, global_action_relative_errors=action_errors, global_mass_relative_errors=mass_errors,
        affine_energy_relative_error=energy_error, affine_expected_energy_j=expected_energy, affine_energies_j=energies,
        rigid_motion_relative_residual=rigid_error, shared_native_trace_error=shared_error,
        shared_native_coefficient_relative_error=coefficient_error, bounded_full_matrix=full,
        global_actions_sha256=sha256_file(output/'global-actions.npz'))


def worker(plan_path, output, guard):
    plan = checked_receipt(read_json(plan_path)); output.mkdir()
    def check_sources():
        if (any(plan[k] != v for k, v in identities().items())
                or any(sha256_file(PROJECT/name) != digest for name, digest in plan['analysis_sources'].items())
                or plan['runtime'] != distribution_identity(guard)):
            raise ValueError('mapped operator source/runtime changed')
    report = dict(completed=False, passed=False, ready_for_training_data=False, training_updates=0,
                  plan_fingerprint=plan['fingerprint'])
    try:
        check_sources()
        mesh = make_recess_mesh(Plate(**plan['plate']), CircularRecess(**plan['recess']), plan['order'],
            angular_cells=plan['angular_cells'], geometry_order=3, defective=plan['defective'])
        system = build_system(mesh, Material(), guard)
        geometry, proof, model = geometry_reference.native_geometry(mesh, guard, return_model=True)
        comparison = compare_native(system, model, geometry, output, guard)
        check_sources()
        report.update(completed=True, passed=comparison['passed'], geometry=geometry, system=system.screen,
            stable_dt_s=system.stable_dt_s, eigenvalue_upper_s2=system.eigenvalue_upper_s2, comparison=comparison,
            scope='Mapped elastic operator, positive GLL mass and conservative stability reference; no transient, wave convergence or PZT/data acceptance.')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'; raise
    finally:
        report['resources'] = guard.report(); write_json(output/'operator-report.json', sealed(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--order', type=int, choices=(3, 4, 5), default=3)
    parser.add_argument('--angular-cells', type=int, choices=(1, 4), default=4)
    parser.add_argument('--intact', action='store_true'); parser.add_argument('--worker', action='store_true')
    parser.add_argument('--plan', type=Path); args = parser.parse_args(); output = args.output.resolve()
    guard = ResourceGuard(ResourceLimits(max_seconds=300, max_rss_mib=1024, max_system_memory_percent=85, min_available_mib=2048))
    if args.worker:
        return 0 if worker(args.plan, output, guard)['passed'] else 2
    if not output.is_relative_to(PROJECT/'outputs'):
        raise ValueError('mapped operator evidence must remain in project outputs')
    output.mkdir(parents=True, exist_ok=False)
    analysis = {str(Path(__file__).relative_to(PROJECT)).replace('\\','/'):sha256_file(__file__),
                'examples/recess_geometry_reference.py':sha256_file(geometry_reference.__file__)}
    plan = dict(kind='mapped_recess_operator_reference', plate=asdict(Plate(.04, .03, .003)),
        recess=asdict(CircularRecess(.021, .014, .008, .0015)), order=args.order, angular_cells=args.angular_cells,
        defective=not args.intact, geometry_order=3, **identities(), analysis_sources=analysis,
        runtime=distribution_identity(guard), comparison_tolerance=1e-10, rigid_zero_tolerance=1e-12)
    snapshot_sources(output/'source-snapshot.zip', plan)
    with zipfile.ZipFile(output/'source-snapshot.zip', 'a', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in analysis:
            archive.writestr(name, (PROJECT/name).read_bytes())
    plan['source_snapshot_sha256'] = sha256_file(output/'source-snapshot.zip'); write_json(output/'operator-plan.json', sealed(plan))
    temporary = output/'temporary'; temporary.mkdir()
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
           **{name:str(temporary) for name in ('TEMP', 'TMP', 'TMPDIR')}}
    result = dict(completed=False, passed=False, ready_for_training_data=False, training_updates=0,
                  plan_fingerprint=plan['fingerprint'])
    try:
        command = [sys.executable, '-m', 'examples.recess_operator_reference', '--worker',
                   '--plan', str(output/'operator-plan.json'), '--output', str(output/'native')]
        result['worker_run'] = run_monitored(command, PROJECT, env, guard, output/'worker.log')
        if result['worker_run']['exit_code'] not in (0, 2):
            raise RuntimeError('mapped operator worker failed; inspect worker.log')
        report = checked_receipt(read_json(output/'native/operator-report.json'))
        if not report['completed'] or report['plan_fingerprint'] != plan['fingerprint']:
            raise ValueError('mapped operator evidence is incomplete or changed')
        for item in report['comparison']['element_files']:
            if sha256_file(output/'native'/item['path']) != item['sha256']:
                raise ValueError('native element archive changed')
        if report['comparison']['global_actions_sha256'] != sha256_file(output/'native/global-actions.npz'):
            raise ValueError('native global action archive changed')
        full = report['comparison']['bounded_full_matrix']
        if full is not None and full['matrices_sha256'] != sha256_file(output/'native/global-matrices.npz'):
            raise ValueError('bounded native matrix archive changed')
        result.update(completed=True, passed=report['passed'], report_sha256=sha256_file(output/'native/operator-report.json'))
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        result['resources'] = guard.report(); write_json(output/'monitor-result.json', sealed(result))
    print(json.dumps({k:v for k,v in result.items() if k != 'resources'}))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
