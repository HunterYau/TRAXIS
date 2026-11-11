"""Circular mechanical loads/probes on the flat, ruled recess-mesh top faces."""
import math

import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import quad_vec

from traxis.recess_hex import RecessHexMesh
from traxis.tensor_hex import line_rule
from traxis.tensor_patches import TensorPatch, _basis


def _bernstein(polynomial):
    power = polynomial(Polynomial([-1., 2.])).coef
    degree = len(power)-1
    return np.array([sum(power[i]*math.comb(k, i)/math.comb(degree, i) for i in range(k+1))
                     for k in range(degree+1)])


def _positive_bound(coefficients, depth=0):
    if coefficients.min() > 1e-12*max(abs(coefficients)):
        return float(coefficients.min())
    if depth == 12 or min(coefficients[0], coefficients[-1]) <= 0:
        raise ArithmeticError('mapped surface has no positive area-Jacobian bound')
    triangle = [coefficients]
    while len(triangle[-1]) > 1:
        triangle.append((triangle[-1][:-1]+triangle[-1][1:])/2)
    return min(_positive_bound(np.array([row[0] for row in triangle]), depth+1),
               _positive_bound(np.array([row[-1] for row in triangle[::-1]]), depth+1))


def _cross(a, b):
    return a[0]*b[1]-a[1]*b[0]


def _face_polynomials(mesh, element, center, radius):
    g = mesh.geometry_order+1
    surface = mesh.geometry_points_m[mesh.geometry_elements[element]].reshape((g, g, g, 3))[:, :, -1]
    nodes = line_rule(mesh.geometry_order)[0]
    scale = max(mesh.plate.width_m, mesh.plate.height_m)
    affine_u = ((1-nodes[:, None, None])*surface[0][None]
                +(1+nodes[:, None, None])*surface[-1][None])/2
    if (np.max(abs(surface-affine_u)) > 1e-12*scale
            or np.max(abs(surface[:, :, 2]-mesh.plate.thickness_m)) > 1e-12*scale):
        raise ValueError('mapped disk integration needs a flat top face linear in its first reference axis')
    basis = _basis(nodes)
    edges = []
    for endpoint in (surface[0], surface[-1]):
        values = (endpoint[:, :2]-center)/radius
        edges.append([sum((basis[j]*values[j, axis] for j in range(g)), Polynomial([0.])) for axis in range(2)])
    a = [(edges[0][i]+edges[1][i])/2 for i in range(2)]
    b = [(edges[1][i]-edges[0][i])/2 for i in range(2)]
    bounds = np.array([[_bernstein(edge[axis]).min(), _bernstein(edge[axis]).max()]
                       for edge in edges for axis in range(2)]).reshape(2, 2, 2)
    lower, upper = bounds[:, :, 0].min(axis=0), bounds[:, :, 1].max(axis=0)
    nearest = np.maximum(np.maximum(lower, -upper), 0.)
    if nearest@nearest > 1+1e-12:
        return None
    j0 = _cross(b, [value.deriv() for value in a])
    j1 = _cross(b, [value.deriv() for value in b])
    minimum = min(_positive_bound(_bernstein(j0-j1)), _positive_bound(_bernstein(j0+j1)))
    return a, b, j0, j1, minimum


def _cuts(a, b):
    endpoints = [[a[i]+sign*b[i] for i in range(2)] for sign in (-1., 1.)]
    polynomials = [sum((value*value for value in edge), Polynomial([-1.])) for edge in endpoints]
    length = b[0]*b[0]+b[1]*b[1]
    polynomials.append(length-_cross(a, b)**2)
    cuts = [-1., 1.]
    for polynomial in polynomials:
        # Also split at extrema, including near-tangent intersections whose
        # multiple roots can acquire small imaginary parts in floating point.
        for candidate in (polynomial, polynomial.deriv()):
            largest = max(abs(candidate.coef), default=0.)
            if largest == 0:
                continue
            candidate = candidate.trim(tol=32*np.finfo(float).eps*largest)
            roots = candidate.roots()
            cuts.extend(float(np.clip(r.real, -1., 1.)) for r in roots
                        if abs(r.imag) < 1e-7 and -1-1e-10 < r.real < 1+1e-10)
    result = []
    for value in sorted(cuts):
        if not result or value-result[-1] > 1e-12:
            result.append(value)
    return result


def _interval(a, b, v):
    origin = np.array([value(v) for value in a])
    tangent = np.array([value(v) for value in b])
    length = float(tangent@tangent)
    projection = float(origin@tangent)
    discriminant = length-float(np.linalg.det(np.stack([origin, tangent])))**2
    if discriminant <= 0:
        return None
    root = math.sqrt(discriminant)
    lo, hi = max(-1., (-projection-root)/length), min(1., (-projection+root)/length)
    return (lo, hi) if hi > lo else None


def circular_patch(mesh, center_xy_m, radius_m, direction=(0., 0., 1.), guard=None,
                   *, max_faces=256, max_workspace_mib=128):
    """Unit-total-force covector, also an area-average displacement probe.

    The geometry is linear in reference u. A disk intersects each u-line in
    an analytically known interval. Integrate the u-polynomial exactly, then
    adapt in v with disk/edge and tangency breakpoints. Normalize by the
    analytic physical disk area, retaining signed finite-element weights.
    """
    center, direction = np.asarray(center_xy_m), np.asarray(direction)
    if (not isinstance(mesh, RecessHexMesh) or center.shape != (2,) or np.iscomplexobj(center)
            or not np.isfinite(center).all() or direction.shape != (3,) or np.iscomplexobj(direction)
            or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0
            or not math.isfinite(radius_m) or radius_m <= 0 or mesh.geometry_order > mesh.order
            or type(max_faces) is not int or not 1 <= max_faces <= 1024
            or not math.isfinite(max_workspace_mib) or not 0 < max_workspace_mib <= 512):
        raise ValueError('mapped disk needs a checked mesh, finite center/direction and positive bounded controls')
    center = center.astype(float); direction = direction.astype(float)/np.linalg.norm(direction)
    if np.any(center-radius_m < 0) or np.any(center+radius_m > [mesh.plate.width_m, mesh.plate.height_m]):
        raise ValueError('the complete mapped mechanical disk must be inside the plate')
    q, g = mesh.order+1, mesh.geometry_order+1
    top = [(int(e), axis, side) for name, e, axis, side in mesh.boundary_faces if name == 'top']
    forecast = len(top)*(g**3*3*8+1024) + max_faces*q*q*64 + 4*1024**2
    if forecast > max_workspace_mib*1024**2 or len(top) > 100000:
        raise MemoryError('mapped disk surface scan exceeds its workspace screen')
    candidates = []
    for e, axis, side in top:
        if guard:
            guard.check()
        if (axis, side) != (2, 1):
            raise ValueError('mapped top-face orientation differs')
        polynomials = _face_polynomials(mesh, e, center, radius_m)
        if polynomials is not None:
            candidates.append((e, polynomials))
            if len(candidates) > max_faces:
                raise MemoryError('mapped disk exceeds its surface-cell bound')
    basis = _basis(line_rule(mesh.order)[0])
    antiderivatives = [value.integ() for value in basis]
    first_antiderivatives = [(value*Polynomial([0., 1.])).integ() for value in basis]
    indices, integrals, details = [], [], []
    error = 0.; evaluations = 0
    for e, (a, b, j0, j1, minimum) in candidates:
        cuts = _cuts(a, b)
        def integrand(v):
            if guard:
                guard.check()
            interval = _interval(a, b, v)
            if interval is None:
                return np.zeros((q, q))
            lo, hi = interval
            inner = np.array([j0(v)*(f(hi)-f(lo))+j1(v)*(h(hi)-h(lo))
                              for f, h in zip(antiderivatives, first_antiderivatives)])
            return inner[:, None]*np.array([value(v) for value in basis])[None, :]
        local, estimate, info = quad_vec(integrand, -1., 1., points=cuts[1:-1],
            epsabs=2e-13/max(1, len(candidates)), epsrel=2e-12, workers=1,
            cache_size=1024**2, limit=2048, full_output=True)
        if not info.success or not np.isfinite(local).all():
            raise ArithmeticError('mapped circular surface quadrature did not converge')
        evaluations += int(info.neval); error += float(estimate)
        if not np.any(local):
            continue
        indices.append(mesh.element_nodes[e].reshape((q, q, q))[:, :, -1].ravel())
        integrals.append(local.ravel()/math.pi)
        details.append(dict(element=e, cuts=cuts, minimum_area_jacobian_m2=minimum*radius_m**2))
    if not indices:
        raise ArithmeticError('mapped disk has no surface support')
    flat = np.concatenate(indices); unique, inverse = np.unique(flat, return_inverse=True)
    weights = np.zeros(len(unique)); np.add.at(weights, inverse, np.concatenate(integrals))
    area_error = abs(float(weights.sum())-1.)
    first_moment = (weights@mesh.points_m[unique, :2]-center)/radius_m
    moment_error = float(np.max(abs(first_moment)))
    if max(area_error, moment_error, error/math.pi) > 1e-10:
        raise ArithmeticError('mapped disk fails analytic area or first-moment agreement')
    report = dict(center_xy_m=center.tolist(), radius_m=radius_m, direction=direction.tolist(),
        analytic_area_m2=math.pi*radius_m**2, integrated_area_m2=float(weights.sum())*math.pi*radius_m**2,
        relative_area_error=area_error, radius_scaled_first_moment_error=moment_error,
        estimated_integral_error_m2=error*radius_m**2, quadrature_evaluations=evaluations,
        candidate_faces=len(candidates), surface_cells=len(details), scalar_support=len(unique),
        negative_weights=int(np.count_nonzero(weights < 0)), faces=details, workspace_screen_bytes=forecast,
        normalization='analytic disk area', integration='exact ruled-u polynomial with adaptive clipped-v integration',
        scope='Work-conjugate mechanical force/area-average displacement; no piezoelectric voltage or clamp-contact fidelity.')
    for value in (unique, weights, direction):
        value.setflags(write=False)
    return TensorPatch(unique, weights, direction, report)


def scenario_patches(mesh, scenario, state='current', guard=None, **limits):
    """Resolve each state's actual mounting; scalar transfer gains stay separate."""
    positions, radii = scenario.sensor_geometry(state)
    if (mesh.plate != scenario.plate or len(scenario.recesses) > 1
            or mesh.defective != (state == 'current' and bool(scenario.recesses))
            or (scenario.recesses and mesh.recess != scenario.recesses[0])):
        raise ValueError('mapped port mesh does not match the scenario and measurement state')
    return tuple(circular_patch(mesh, xy, float(radius), guard=guard, **limits) for xy, radius in zip(positions, radii))
