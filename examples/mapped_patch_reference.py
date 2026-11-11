"""Native hierarchical shapes and independent Gaussian clipped-disk integration."""
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

from examples import recess_geometry_reference as geometry_reference
from examples.tensor_wave import PROJECT, checked_receipt, identities, read_json, sealed, snapshot_sources
from traxis.config import CircularRecess, Plate
from traxis.elmer_reference import run_monitored
from traxis.highorder_verification import distribution_identity
from traxis.mapped_patches import circular_patch
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import sha256_file, write_json
from traxis.tensor_hex import line_rule


def recipes():
    return [['corner', [.006, .006], .003], ['remounted-corner', [.0064, .0057], .0033],
        ['curved-interface', [.029, .014], .003],
        ['curved-vertex', [.021+.008/math.sqrt(2), .014+.008/math.sqrt(2)], .003]]


def native_covector(mesh, model, patch, guard):
    """Direct native shapes/Jacobians; Gauss integration in both coordinates.

    Edge and tangency polynomials come from independent samples of the native
    deformed geometry. A cosine change of variable smooths interval endpoints.
    No producer shape antiderivatives or stored per-face integrals are used.
    """
    import ngsolve as ng
    native = model['mesh']; scalar = ng.H1(native, order=mesh.order); ns = scalar.ndof
    if ns != len(mesh.points_m):
        raise ArithmeticError('native surface topology differs')
    center = np.asarray(patch.report['center_xy_m']); radius = patch.report['radius_m']
    endpoint_samples = leggauss(mesh.geometry_order+1)[0]
    nodes = line_rule(mesh.order)[0]; rows = {}; faces = []
    for name, e, axis, side in mesh.boundary_faces:
        if name != 'top':
            continue
        guard.check(); frame = np.asarray(model['reference_frames'][e])
        rotation_inverse = np.linalg.inv(frame[1:])
        eid = ng.ElementId(ng.VOL, e); fe, trafo = scalar.GetFE(eid), native.GetTrafo(eid)
        dofs = np.asarray(scalar.GetDofNrs(eid), dtype=int)
        def ref(point):
            return (point-frame[0])@rotation_inverse
        edges = []
        for u in (-1., 1.):
            reference = [ref(np.array([u, v, 1.])) for v in endpoint_samples]
            rule = ng.IntegrationRule(points=[p.tolist() for p in reference], weights=[1.]*len(reference))
            values = (np.array([np.array(trafo(ip).point)[:2] for ip in rule])-center)/radius
            edges.append([Polynomial.fit(endpoint_samples, values[:, a], mesh.geometry_order).convert() for a in range(2)])
        # Conservative polynomial sample/control enclosure, independently
        # converted to Bernstein coefficients on [0,1].
        bounds = []
        for edge in edges:
            bound = []
            for value in edge:
                power = value(Polynomial([-1., 2.])).coef; degree = len(power)-1
                controls = [sum(power[i]*math.comb(k, i)/math.comb(degree, i) for i in range(k+1)) for k in range(degree+1)]
                bound.append((min(controls), max(controls)))
            bounds.append(bound)
        bounds = np.asarray(bounds); lower = bounds[:, :, 0].min(axis=0); upper = bounds[:, :, 1].max(axis=0)
        nearest = np.maximum(np.maximum(lower, -upper), 0.)
        if nearest@nearest > 1+1e-11:
            continue
        a = [(edges[0][i]+edges[1][i])/2 for i in range(2)]
        b = [(edges[1][i]-edges[0][i])/2 for i in range(2)]
        boundary = [edge[0]**2+edge[1]**2-1 for edge in edges]
        boundary.append(b[0]**2+b[1]**2-(a[0]*b[1]-a[1]*b[0])**2)
        cuts = [-1., 1.]
        for polynomial in boundary:
            for value in (polynomial, polynomial.deriv()):
                scale = max(abs(value.coef), default=0.)
                if scale:
                    roots = value.trim(tol=32*np.finfo(float).eps*scale).roots()
                    cuts.extend(float(r.real) for r in roots if abs(r.imag) < 1e-7 and -1 < r.real < 1)
        cuts = np.array(sorted(cuts)); cuts = cuts[np.r_[True, np.diff(cuts) > 1e-12]]
        ids = mesh.element_nodes[e].reshape((mesh.order+1,)*3)[:, :, -1].ravel()
        for node, (u, v) in zip(ids, np.array(np.meshgrid(nodes, nodes, indexing='ij')).reshape(2, -1).T):
            point = ref(np.array([u, v, 1.])); values = np.asarray(fe.CalcShape(*point))
            if node in rows:
                old_ids, old_values = rows[node]
                diff = np.zeros(ns); np.add.at(diff, old_ids, old_values); np.add.at(diff, dofs, -values)
                if np.max(abs(diff)) > 1e-11:
                    raise ArithmeticError('native shared surface trace differs')
            else:
                rows[node] = dofs.copy(), values.copy()
        faces.append((e, a, b, cuts, dofs, fe, trafo, frame, rotation_inverse))
    dual = np.zeros(ns)
    for node, weight in zip(patch.indices, patch.weights):
        if node not in rows:
            raise ArithmeticError('native surface support is incomplete')
        dofs, values = rows[node]; np.add.at(dual, dofs, weight*values)
    gu, wu = leggauss(mesh.order+2)
    levels = []; previous = None; final = None
    for count in (16, 32, 64, 128):
        gv, wv = leggauss(count); result = np.zeros(ns); evaluations = 0
        for e, a, b, cuts, dofs, fe, trafo, frame, inverse in faces:
            local = np.zeros(len(dofs))
            for left, right in zip(cuts[:-1], cuts[1:]):
                theta = (gv+1)*math.pi/2
                vs = (left+right)/2-(right-left)*np.cos(theta)/2
                weights = wv*(math.pi/2)*(right-left)*np.sin(theta)/2
                for v, weight_v in zip(vs, weights):
                    guard.check()
                    origin = np.array([f(v) for f in a]); direction = np.array([f(v) for f in b])
                    quadratic = float(direction@direction); linear = float(origin@direction)
                    constant = float(origin@origin)-1.
                    delta = linear*linear-quadratic*constant
                    if delta <= 0:
                        continue
                    lo, hi = max(-1., (-linear-math.sqrt(delta))/quadratic), min(1., (-linear+math.sqrt(delta))/quadratic)
                    if hi <= lo:
                        continue
                    us = (hi+lo)/2+(hi-lo)*gu/2
                    reference = [(np.array([u, v, 1.])-frame[0])@inverse for u in us]
                    rule = ng.IntegrationRule(points=[p.tolist() for p in reference], weights=[1.]*len(reference))
                    for point, ip, weight_u in zip(reference, rule, wu):
                        jac = np.asarray(trafo(ip).jacobi)@inverse.T
                        area = np.linalg.norm(np.cross(jac[:, 0], jac[:, 1]))
                        local += np.asarray(fe.CalcShape(*point))*(area*weight_u*(hi-lo)/2*weight_v/(math.pi*radius**2))
                        evaluations += 1
            np.add.at(result, dofs, local)
        change = None if previous is None else float(np.linalg.norm(result-previous)/np.linalg.norm(result))
        levels.append(dict(outer_gauss_points=count, relative_change=change, native_shape_evaluations=evaluations))
        final = result
        if count >= 32 and change < 1e-11:
            break
        previous = result
    converged = levels[-1]['relative_change'] is not None and levels[-1]['relative_change'] < 1e-11
    error = float(np.linalg.norm(final-dual)/np.linalg.norm(final))
    return final, dual, dict(passed=bool(converged and error < 1e-9), relative_l2=error,
        quadrature_converged=bool(converged), levels=levels, candidate_faces=len(faces))


def worker(plan_path, output, guard):
    plan = checked_receipt(read_json(plan_path)); output.mkdir()
    def source_check():
        if (any(plan[k] != v for k, v in identities().items())
                or any(sha256_file(PROJECT/name) != digest for name, digest in plan['analysis_sources'].items())
                or plan['runtime'] != distribution_identity(guard)):
            raise ValueError('mapped surface reference source/runtime changed')
    report = dict(completed=False, passed=False, ready_for_training_data=False, training_updates=0,
                  plan_fingerprint=plan['fingerprint'])
    try:
        source_check()
        if (plan['kind'] != 'mapped_surface_native_reference' or plan['plate'] != asdict(Plate(.04, .03, .003175))
                or plan['recess'] != asdict(CircularRecess(.021, .014, .008, .0015))
                or type(plan['order']) is not int or plan['order'] not in (3, 4, 5)
                or plan['angular_cells'] != 4 or type(plan['defective']) is not bool or plan['cases'] != recipes()
                or plan['native_tolerance'] != 1e-9 or plan['gauss_change_tolerance'] != 1e-11):
            raise ValueError('mapped surface reference geometry or criteria changed')
        mesh = geometry_reference.make_recess_mesh(Plate(**plan['plate']), CircularRecess(**plan['recess']),
            plan['order'], angular_cells=plan['angular_cells'], geometry_order=3, defective=plan['defective'])
        geometry, _, model = geometry_reference.native_geometry(mesh, guard, return_model=True)
        cases, arrays = [], {}
        for index, (name, center, radius) in enumerate(plan['cases']):
            port = circular_patch(mesh, center, radius, guard=guard)
            native, dual, details = native_covector(mesh, model, port, guard)
            cases.append(dict(name=name, patch=port.report, native=details, passed=details['passed']))
            arrays[f'case{index}_native_hierarchical'] = native; arrays[f'case{index}_mapped_native_dual'] = dual
            arrays[f'case{index}_nodal_indices'] = port.indices; arrays[f'case{index}_nodal_weights'] = port.weights
            arrays[f'case{index}_physical_support_m'] = mesh.points_m[port.indices]
            write_json(output/'progress.json', dict(completed=False, cases_completed=index+1, cases_planned=len(plan['cases'])), overwrite=True)
        np.savez_compressed(output/'surface-proofs.npz', **arrays)
        source_check()
        report.update(completed=True, passed=bool(all(c['passed'] for c in cases)), cases=cases,
            geometry=geometry, proof_sha256=sha256_file(output/'surface-proofs.npz'),
            scope='Bounded physical disk covectors and native shapes at 3.175 mm; no full-size propagation, physical clamp or PZT-voltage acceptance.')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'; raise
    finally:
        report['resources'] = guard.report(); write_json(output/'surface-report.json', sealed(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--order', type=int, choices=(3, 4, 5), default=3)
    parser.add_argument('--intact', action='store_true'); parser.add_argument('--worker', action='store_true')
    parser.add_argument('--plan', type=Path); args = parser.parse_args(); output = args.output.resolve()
    guard = ResourceGuard(ResourceLimits(max_seconds=300, max_rss_mib=1024, max_system_memory_percent=85, min_available_mib=2048))
    if args.worker:
        return 0 if worker(args.plan, output, guard)['passed'] else 2
    if not output.is_relative_to(PROJECT/'outputs'):
        raise ValueError('mapped surface evidence must remain in project outputs')
    output.mkdir(parents=True, exist_ok=False)
    analysis = {str(Path(name).resolve().relative_to(PROJECT)).replace('\\', '/'):sha256_file(name)
                for name in (__file__, geometry_reference.__file__)}
    plan = dict(kind='mapped_surface_native_reference', plate=asdict(Plate(.04, .03, .003175)),
        recess=asdict(CircularRecess(.021, .014, .008, .0015)), order=args.order, angular_cells=4,
        defective=not args.intact, cases=recipes(), native_tolerance=1e-9, gauss_change_tolerance=1e-11,
        **identities(), analysis_sources=analysis, runtime=distribution_identity(guard))
    snapshot_sources(output/'source-snapshot.zip', plan)
    with zipfile.ZipFile(output/'source-snapshot.zip', 'a', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in analysis: archive.writestr(name, (PROJECT/name).read_bytes())
    plan['source_snapshot_sha256'] = sha256_file(output/'source-snapshot.zip'); write_json(output/'surface-plan.json', sealed(plan))
    temporary = output/'temporary'; temporary.mkdir()
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', **{name:str(temporary) for name in ('TEMP', 'TMP', 'TMPDIR')}}
    result = dict(completed=False, passed=False, ready_for_training_data=False, training_updates=0, plan_fingerprint=plan['fingerprint'])
    try:
        command = [sys.executable, '-m', 'examples.mapped_patch_reference', '--worker', '--plan', str(output/'surface-plan.json'), '--output', str(output/'native')]
        result['worker_run'] = run_monitored(command, PROJECT, env, guard, output/'worker.log')
        if result['worker_run']['exit_code'] not in (0, 2): raise RuntimeError('mapped surface worker failed; inspect worker.log')
        report = checked_receipt(read_json(output/'native/surface-report.json'))
        if (not report['completed'] or report['plan_fingerprint'] != plan['fingerprint']
                or report['proof_sha256'] != sha256_file(output/'native/surface-proofs.npz')):
            raise ValueError('mapped surface evidence incomplete or changed')
        result.update(completed=True, passed=bool(report['passed'] and result['worker_run']['exit_code'] == 0),
            report_sha256=sha256_file(output/'native/surface-report.json'))
    except Exception as error: result['error'] = f'{type(error).__name__}: {error}'
    finally:
        result['resources'] = guard.report(); write_json(output/'monitor-result.json', sealed(result))
    print(json.dumps({k:v for k,v in result.items() if k != 'resources'})); return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
