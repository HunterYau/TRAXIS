"""Check conforming recess topology, represented shape and native geometry."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss

from examples.tensor_wave import PROJECT, checked_receipt, identities, read_json, sealed, snapshot_sources
from traxis.config import CircularRecess, Plate
from traxis.elmer_reference import run_monitored
from traxis.highorder_verification import distribution_identity
from traxis.recess_hex import evaluate_geometry, make_recess_mesh
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import sha256_file, write_json
from traxis.tensor_hex import line_rule


def circle_reference(radius, order, angular_cells):
    """Closed polynomial area and radial extrema, independent of volume mapping."""
    nodes = line_rule(order)[0]
    area, radial_error = 0., 0.
    for segment in range(4*angular_cells):
        angle = -math.pi/4 + (segment+(nodes+1)/2)*math.pi/(2*angular_cells)
        x = Polynomial.fit(nodes, radius*np.cos(angle), order).convert()
        y = Polynomial.fit(nodes, radius*np.sin(angle), order).convert()
        anti = ((x*y.deriv()-y*x.deriv())/2).integ()
        area += float(anti(1)-anti(-1))
        squared = x*x+y*y
        roots = squared.deriv().roots()
        real = roots.real[(abs(roots.imag) < 1e-9) & (roots.real > -1) & (roots.real < 1)]
        candidates = np.r_[-1., real, 1.]
        radial_error = max(radial_error, float(np.max(abs(np.sqrt(squared(candidates))/radius-1))))
    return area, radial_error


def _positive_bernstein(coefficients, depth=0):
    if coefficients.min() > 1e-10*max(abs(coefficients)):
        return float(coefficients.min())
    if depth == 10 or min(coefficients[0], coefficients[-1]) <= 0:
        raise ArithmeticError('polynomial geometry has no positive determinant certificate')
    triangle = [np.asarray(coefficients)]
    while len(triangle[-1]) > 1:
        triangle.append((triangle[-1][:-1]+triangle[-1][1:])/2)
    left = np.array([row[0] for row in triangle])
    right = np.array([row[-1] for row in triangle[::-1]])
    return min(_positive_bernstein(left, depth+1), _positive_bernstein(right, depth+1))


def determinant_certificate(mesh, element):
    # This block family is linear in u,z. det(J) is linear in u, independent
    # of z, and has degree at most 2*g-1 in v. Bound its two u endpoints.
    degree = 2*mesh.geometry_order-1
    s = (1-np.cos(np.linspace(0, math.pi, degree+1)))/2
    basis = np.array([math.comb(degree, k)*s**k*(1-s)**(degree-k) for k in range(degree+1)]).T
    bounds = []
    for u in (-1., 1.):
        points = np.column_stack([np.full(len(s), u), 2*s-1, np.zeros(len(s))])
        det = np.linalg.det(evaluate_geometry(mesh, element, points)[1])
        bounds.append(_positive_bernstein(np.linalg.solve(basis, det)))
    return min(bounds)


def measure(mesh, guard):
    g = mesh.geometry_order
    u, uw = leggauss(2); v, vw = leggauss(g+2)
    reference = np.stack(np.meshgrid(u, v, u, indexing='ij'), axis=-1).reshape(-1, 3)
    weights = np.kron(np.kron(uw, vw), uw)
    volume, minimum, condition, certificate = 0., math.inf, 0., math.inf
    for e in range(len(mesh.element_nodes)):
        guard.check()
        _, jac = evaluate_geometry(mesh, e, reference)
        det = np.linalg.det(jac)
        volume += float(weights @ det)
        minimum = min(minimum, float(det.min()))
        condition = max(condition, float(np.linalg.cond(jac).max()))
        certificate = min(certificate, determinant_certificate(mesh, e))
    a, w = leggauss(max(12, g+2))
    uv = np.stack(np.meshgrid(a, a, indexing='ij'), axis=-1).reshape(-1, 2)
    weights = np.kron(w, w)
    normal, divergence, areas = np.zeros(3), 0., {}
    for name, e, axis, side in mesh.boundary_faces:
        guard.check()
        points = np.empty((len(uv), 3)); points[:, axis] = -1 if side == 0 else 1
        points[:, [i for i in range(3) if i != axis]] = uv
        physical, jac = evaluate_geometry(mesh, e, points)
        vector = np.cross(jac[:, :, (axis+1) % 3], jac[:, :, (axis+2) % 3])*(-1 if side == 0 else 1)
        normal += weights @ vector
        divergence += float(weights @ np.einsum('qi,qi->q', physical, vector))/3
        areas[name] = areas.get(name, 0.) + float(weights @ np.linalg.norm(vector, axis=1))
    plate, recess = mesh.plate, mesh.recess
    full = plate.width_m*plate.height_m*plate.thickness_m
    circle_area, radius_error = circle_reference(recess.radius_m, g, mesh.angular_cells)
    expected = full-(circle_area*recess.depth_m if mesh.defective else 0)
    exact = full-(recess.volume_m3 if mesh.defective else 0)
    errors = dict(represented_volume_relative_error=abs(volume/expected-1),
        divergence_volume_relative_error=abs(divergence/volume-1),
        closed_normal_relative_residual=float(np.linalg.norm(normal)/(plate.width_m*plate.height_m)),
        top_area_relative_error=abs(areas['top']/(plate.width_m*plate.height_m)-1))
    if mesh.defective:
        errors['floor_area_relative_error'] = abs(areas['recess_floor']/circle_area-1)
    return dict(volume_m3=volume, analytic_circular_volume_m3=exact, represented_volume_m3=expected,
        polynomial_circle_area_m2=circle_area, circular_area_relative_error=abs(circle_area/(math.pi*recess.radius_m**2)-1),
        circle_radial_relative_error=radius_error, areas_m2=areas, minimum_sampled_jacobian=minimum,
        polynomial_determinant_lower_bound=certificate, maximum_sampled_jacobian_condition=condition,
        vector_dofs=3*len(mesh.points_m), elements=len(mesh.element_nodes), boundary_faces=len(mesh.boundary_faces),
        shared_faces=mesh.shared_faces, shared_coordinate_error_m=mesh.shared_coordinate_error_m,
        **errors, passed=bool(max(errors.values()) < 1e-10 and minimum > 0 and certificate > 0))


def native_geometry(mesh, guard, *, return_model=False):
    if len(mesh.geometry_points_m) > 3500 or len(mesh.element_nodes) > 128:
        raise MemoryError('native geometry interpolation exceeds the bounded dense transform')
    import ngsolve as ng
    import netgen.meshing as nm
    ng.SetNumThreads(1)
    g = mesh.geometry_order
    base = make_recess_mesh(mesh.plate, mesh.recess, g, geometry_order=1, defective=mesh.defective,
        angular_cells=mesh.angular_cells, radial_cells=mesh.radial_cells, layers=mesh.layers)
    raw = nm.Mesh(dim=3)
    vertices = [raw.Add(nm.MeshPoint(nm.Pnt(*p))) for p in base.corner_points_m]
    region = raw.AddRegion('solid', dim=3)
    for cell in base.corner_elements:
        raw.Add(nm.Element3D(region, [vertices[i] for i in cell]))
    native = ng.Mesh(raw)
    scalar = ng.H1(native, order=g)
    ns = len(base.points_m)
    if scalar.ndof != ns:
        raise ArithmeticError('native scalar geometry topology disagrees')
    q = (line_rule(g)[0]+1)/2
    points = np.stack(np.meshgrid(q, q, q, indexing='ij'), axis=-1).reshape(-1, 3)
    rule = ng.IntegrationRule(points=points.tolist(), weights=np.ones(len(points)).tolist())
    transform = np.zeros((ns, ns)); seen = np.zeros(ns, dtype=bool); overlap = 0.
    corner_reference = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0],
                                  [0,0,1], [1,0,1], [1,1,1], [0,1,1]])
    corner_rule = ng.IntegrationRule(points=corner_reference.tolist(), weights=[1.]*8)
    frames = []
    for element in native.Elements(ng.VOL):
        guard.check()
        e = int(element.nr); eid = ng.ElementId(ng.VOL, e)
        fe, trafo = scalar.GetFE(eid), native.GetTrafo(eid)
        ids = base.element_nodes[e]; candidates = base.points_m[ids]
        corners = base.corner_points_m[base.corner_elements[e]]
        mapped = np.array([np.array(trafo(ip).point) for ip in corner_rule])
        distance = np.linalg.norm(mapped[:, None]-corners[None], axis=2)
        nearest = distance.argmin(axis=1)
        if np.max(distance[np.arange(8), nearest]) > 1e-12 or len(np.unique(nearest)) != 8:
            raise ArithmeticError('native cell order or corner geometry differs')
        coefficients = np.linalg.lstsq(np.c_[np.ones(8), corner_reference],
                                       2*corner_reference[nearest]-1, rcond=None)[0]
        frames.append(coefficients)
        for point, ip in zip(points, rule):
            measured = np.array(trafo(ip).point)
            distance = np.linalg.norm(candidates-measured, axis=1); local = int(distance.argmin())
            if distance[local] > 1e-12:
                raise ArithmeticError('native straight geometry does not match nodal reference')
            node = ids[local]; row = np.zeros(ns); row[np.asarray(scalar.GetDofNrs(eid), int)] = fe.CalcShape(*point)
            if seen[node]:
                overlap = max(overlap, float(np.max(abs(row-transform[node]))))
            else:
                transform[node] = row; seen[node] = True
    if not seen.all() or overlap > 1e-11:
        raise ArithmeticError('native geometry interpolation is incomplete or discontinuous')
    if mesh.geometry_points_m.shape != base.points_m.shape:
        raise ArithmeticError('geometry control inventory differs')
    values = np.linalg.solve(transform, mesh.geometry_points_m-base.points_m)
    deformation = ng.GridFunction(ng.VectorH1(native, order=g))
    if len(deformation.components) != 3:
        raise ArithmeticError('native geometry needs three scalar coefficient blocks')
    for axis in range(3):
        coefficients = deformation.components[axis].vec.FV().NumPy()
        if len(coefficients) != ns:
            raise ArithmeticError('native vector component basis differs from scalar H1')
        coefficients[:] = values[:, axis]
    native.SetDeformation(deformation)
    a, _ = leggauss(g+2)
    points = (np.stack(np.meshgrid(a, a, a, indexing='ij'), axis=-1).reshape(-1, 3)+1)/2
    rule = ng.IntegrationRule(points=points.tolist(), weights=np.ones(len(points)).tolist())
    coordinate_error, jacobian_error, minimum = 0., 0., math.inf
    native_points, independent_points, native_jac, independent_jac = [], [], [], []
    for e, frame in enumerate(frames):
        guard.check()
        reference = np.c_[np.ones(len(points)), points] @ frame
        physical, jac = evaluate_geometry(mesh, e, reference)
        expected = jac @ frame[1:].T
        trafo = native.GetTrafo(ng.ElementId(ng.VOL, e))
        measured = np.array([np.array(trafo(ip).point) for ip in rule])
        matrices = np.array([np.array(trafo(ip).jacobi) for ip in rule])
        coordinate_error = max(coordinate_error, float(np.max(abs(measured-physical))))
        jacobian_error = max(jacobian_error, float(np.linalg.norm(matrices-expected)/np.linalg.norm(expected)))
        minimum = min(minimum, float(np.linalg.det(matrices).min()))
        native_points.append(measured); independent_points.append(physical)
        native_jac.append(matrices); independent_jac.append(expected)
    volume = float(ng.Integrate(ng.CoefficientFunction(1), native, order=2*g+4))
    details = dict(native_volume_m3=volume, coordinate_maximum_absolute_error_m=coordinate_error,
        jacobian_relative_l2=jacobian_error, minimum_native_jacobian=minimum,
        shared_transform_maximum_error=overlap)
    proofs = dict(native_points=np.asarray(native_points), independent_points=np.asarray(independent_points),
        native_jacobians=np.asarray(native_jac), independent_jacobians=np.asarray(independent_jac))
    if return_model:
        return details, proofs, dict(mesh=native, deformation=deformation, reference_frames=frames)
    return details, proofs


def worker(plan_path, output, guard):
    plan = checked_receipt(read_json(plan_path)); output.mkdir()
    def check_sources():
        if (any(plan[k] != v for k, v in identities().items())
                or plan['driver_sha256_self'] != sha256_file(__file__)
                or plan['runtime'] != distribution_identity(guard)):
            raise ValueError('recess geometry source/runtime changed')
    report = dict(completed=False, passed=False, ready_for_training_data=False, training_updates=0,
                  plan_fingerprint=plan['fingerprint'])
    try:
        check_sources(); rows, native_rows, arrays = [], [], {}
        for case in plan['geometry_cases']:
            guard.check(); plate, recess = Plate(**case['plate']), CircularRecess(**case['recess'])
            mesh = make_recess_mesh(plate, recess, 1, geometry_order=case['geometry_order'], defective=case['defective'])
            rows.append(dict(**case, **measure(mesh, guard)))
        for i, case in enumerate(plan['native_cases']):
            guard.check(); plate, recess = Plate(**case['plate']), CircularRecess(**case['recess'])
            mesh = make_recess_mesh(plate, recess, 3, geometry_order=case['geometry_order'], defective=case['defective'])
            measured = measure(mesh, guard); native, proof = native_geometry(mesh, guard)
            volume_error = abs(native['native_volume_m3']/measured['volume_m3']-1)
            passed = bool(measured['passed'] and volume_error < 1e-10
                and native['coordinate_maximum_absolute_error_m'] < 1e-12
                and native['jacobian_relative_l2'] < 1e-10 and native['minimum_native_jacobian'] > 0)
            native_rows.append({**case, **measured, **native,
                                'native_volume_relative_error': volume_error, 'passed': passed})
            for name, value in dict(**proof, geometry_points=mesh.geometry_points_m,
                    geometry_elements=mesh.geometry_elements, nodes=mesh.points_m,
                    element_nodes=mesh.element_nodes, cell_boxes=mesh.cell_boxes).items():
                arrays[f'case{i}_'+name] = value
        for defective in (False, True):
            for diameter in (.02, .03, .04):
                for depth in (.00075, .0015, .002):
                    selected = [r for r in rows if r['defective'] == defective and
                        r['recess']['radius_m'] == diameter/2 and r['recess']['depth_m'] == depth]
                    selected.sort(key=lambda r: r['geometry_order'])
                    errors = [r['circular_area_relative_error'] for r in selected]
                    if len(errors) != 5 or any(b >= a for a, b in zip(errors, errors[1:])) or errors[-1] >= 1e-5:
                        raise ArithmeticError('declared circular geometry refinement screen failed')
        np.savez_compressed(output/'geometry-proofs.npz', **arrays)
        check_sources()
        report.update(completed=True, passed=all(r['passed'] for r in rows+native_rows),
            geometry_cases=rows, native_cases=native_rows, archive_sha256=sha256_file(output/'geometry-proofs.npz'),
            scope='Conforming material geometry and native mapped geometry only; no transient, wave mesh/time or PZT/data acceptance.')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'; raise
    finally:
        report['resources'] = guard.report(); write_json(output/'geometry-report.json', sealed(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--plan', type=Path); args = parser.parse_args()
    output = args.output.resolve()
    guard = ResourceGuard(ResourceLimits(max_seconds=180, max_rss_mib=1024, max_system_memory_percent=85, min_available_mib=2048))
    if args.worker:
        return 0 if worker(args.plan, output, guard)['passed'] else 2
    if not output.is_relative_to(PROJECT/'outputs'):
        raise ValueError('geometry evidence must remain in project outputs')
    output.mkdir(parents=True, exist_ok=False)
    cases = [dict(plate=asdict(Plate()), recess=asdict(CircularRecess(.25, .25, diameter/2, depth)),
                  geometry_order=g, defective=defective)
        for defective in (False, True) for diameter in (.02, .03, .04)
        for depth in (.00075, .0015, .002) for g in range(1, 6)]
    native = [dict(plate=asdict(Plate(.04, .03, .003)), recess=asdict(CircularRecess(.021, .014, .008, .0015)),
                   geometry_order=g, defective=defective) for defective in (False, True) for g in range(1, 6)]
    plan = dict(kind='conforming_recess_geometry_reference', geometry_cases=cases, native_cases=native,
        **identities(), driver_sha256_self=sha256_file(__file__), runtime=distribution_identity(guard),
        numerical_tolerance=1e-10, finest_circle_area_relative_limit=1e-5)
    snapshot_sources(output/'source-snapshot.zip', plan)
    with zipfile.ZipFile(output/'source-snapshot.zip', 'a', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('examples/recess_geometry_reference.py', Path(__file__).read_bytes())
    plan['source_snapshot_sha256'] = sha256_file(output/'source-snapshot.zip')
    write_json(output/'geometry-plan.json', sealed(plan))
    temporary = output/'temporary'; temporary.mkdir()
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
           **{name:str(temporary) for name in ('TEMP', 'TMP', 'TMPDIR')}}
    result = dict(completed=False, passed=False, ready_for_training_data=False, training_updates=0,
                  plan_fingerprint=plan['fingerprint'])
    try:
        command = [sys.executable, '-m', 'examples.recess_geometry_reference', '--worker',
                   '--plan', str(output/'geometry-plan.json'), '--output', str(output/'native')]
        result['worker_run'] = run_monitored(command, PROJECT, env, guard, output/'worker.log')
        if result['worker_run']['exit_code'] not in (0, 2):
            raise RuntimeError('recess geometry worker failed; inspect worker.log')
        report = checked_receipt(read_json(output/'native/geometry-report.json'))
        if (not report['completed'] or report['plan_fingerprint'] != plan['fingerprint']
                or report['archive_sha256'] != sha256_file(output/'native/geometry-proofs.npz')):
            raise ValueError('recess geometry evidence is incomplete or changed')
        result.update(completed=True, passed=report['passed'], report_sha256=sha256_file(output/'native/geometry-report.json'))
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        result['resources'] = guard.report(); write_json(output/'monitor-result.json', sealed(result))
    print(json.dumps({k:v for k,v in result.items() if k != 'resources'}))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
