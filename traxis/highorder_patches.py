"""Work-conjugate circular traction/probe vectors in the native H1 basis."""

import math

import numpy as np

from traxis.config import positive
from traxis.polynomial_patches import disk_triangle_barycentric_moments, triangle_powers


def patch_polynomial_moments(system, vector, center, radius, direction, region, order):
    """Compare represented centered monomials to exact disk-sector averages."""
    if region not in ('full', 'half', 'quarter'):
        raise ValueError('choose a full disk, positive-x half disk or positive-xy quarter disk')
    import ngsolve as ng
    direction = np.array(direction, float)/np.linalg.norm(direction)
    field = ng.GridFunction(system.scalar_space)
    rows = []
    for a, b in triangle_powers(order):
        quarter = math.gamma((a+1)/2)*math.gamma((b+1)/2)/(2*(a+b+2)*math.gamma((a+b+2)/2))
        if region == 'quarter':
            average = quarter/(math.pi/4)
        elif region == 'half':
            average = 0. if b % 2 else 2*quarter/(math.pi/2)
        else:
            average = 0. if a % 2 or b % 2 else 4*quarter/math.pi
        field.Set(((ng.x-center[0])/radius)**a*((ng.y-center[1])/radius)**b)
        measured = vector.reshape(3, -1) @ field.vec.FV().NumPy()
        expected = direction*average
        rows.append({'powers': [a, b], 'expected_component_averages': expected.tolist(),
            'measured_component_averages': measured.tolist(), 'absolute_error': float(np.max(abs(measured-expected)))})
    return {'monomials': rows, 'maximum_absolute_error': max(row['absolute_error'] for row in rows)}


def highorder_surface_patch(system, plate, center_xy_m, radius_m, direction=(0., 0., 1.), guard=None,
                            *, max_scalar_dofs=20000, max_tetrahedra=40000, max_workspace_mib=128,
                            trace_degree=None):
    """Unit-total-force load and identical area-average displacement probe.

    Integrate the native finite-element trace polynomial on each free, flat
    top face. The returned covector is component-major, with all hierarchical
    signs retained. This is a mechanical footprint, not a piezoelectric model.
    """
    positive('patch radius', radius_m)
    center, direction = np.asarray(center_xy_m), np.asarray(direction)
    if (np.iscomplexobj(center) or center.shape != (2,) or not np.isfinite(center).all()
            or np.iscomplexobj(direction) or direction.shape != (3,) or not np.isfinite(direction).all()
            or np.linalg.norm(direction) == 0):
        raise ValueError('patch center and nonzero direction must be finite real coordinates')
    center, direction = center.astype(float), direction.astype(float)/np.linalg.norm(direction)
    if (type(max_scalar_dofs) is not int or not 4 <= max_scalar_dofs <= 2000000
            or type(max_tetrahedra) is not int or not 1 <= max_tetrahedra <= 1000000
            or not math.isfinite(max_workspace_mib) or not 0 < max_workspace_mib <= 2048):
        raise ValueError('invalid circular H1 surface extraction limits')
    space_order = system.scalar_space.globalorder
    order = space_order if trace_degree is None else trace_degree
    if type(order) is not int or not space_order<=order<=4:
        raise ValueError('surface trace degree must cover the declared space order through degree four')
    powers = triangle_powers(order)
    if not 1 <= order <= 4 or not 4 <= system.scalar_space.ndof <= max_scalar_dofs:
        raise ValueError('circular H1 footprints require bounded degree-one through degree-four spaces')
    if not 1 <= system.mesh.ne <= max_tetrahedra or not 4 <= system.mesh.nv <= 1000000:
        raise ValueError('surface extraction exceeds the reference element/vertex bound')
    workspace_mib = (system.mesh.ne*1024+system.mesh.nv*24+system.scalar_space.ndof*32)/1024**2
    if workspace_mib > max_workspace_mib:
        raise MemoryError('circular H1 surface extraction exceeds its workspace screen')
    import ngsolve as ng
    points = np.array([tuple(vertex.point) for vertex in system.mesh.vertices])
    elements = list(system.mesh.Elements(ng.VOL))
    if len(elements) != system.mesh.ne:
        raise ValueError('native volume element count changed during surface extraction')
    tets = np.array([[v.nr for v in element.vertices] for element in elements], dtype=int)
    if tets.shape != (len(elements), 4):
        raise ValueError('circular H1 footprints require straight tetrahedra')
    dimensions = np.array([plate.width_m, plate.height_m, plate.thickness_m])
    tolerance = dimensions.max()*1e-10
    if (points < -tolerance).any() or (points > dimensions+tolerance).any():
        raise ValueError('native footprint mesh escapes the axis-aligned plate')
    faces = tets[:, ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))].reshape(-1, 3)
    owners = np.repeat(np.arange(len(tets)), 4)
    top = np.isclose(points[:, 2], plate.thickness_m, rtol=0, atol=plate.thickness_m*1e-10)
    selected = top[faces].all(axis=1)
    faces, owners = faces[selected], owners[selected]
    unique, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    if (counts > 2).any():
        raise ValueError('nonmanifold top-surface topology')
    faces, owners = unique[counts == 1], owners[first[counts == 1]]
    if not len(faces):
        raise ValueError('native mesh has no free flat top faces')
    lower, upper = points[faces, :2].min(axis=1)-center, points[faces, :2].max(axis=1)-center
    nearest = np.maximum(np.maximum(lower, -upper), 0)
    selected = (nearest**2).sum(axis=1) <= radius_m**2
    nodes = np.array([(i/order, j/order) for i, j in powers])
    vandermonde = np.array([[x**i*y**j for i, j in powers] for x, y in nodes])
    condition = float(np.linalg.cond(vandermonde))
    if condition > 1e5:
        raise ArithmeticError('surface trace interpolation is ill-conditioned')
    reference_vertices = np.vstack([np.eye(3), np.zeros(3)])
    weights = np.zeros(system.scalar_space.ndof)
    area, covered, maximum_fit_error = 0., 0, 0.
    for face, owner in zip(faces[selected], owners[selected]):
        if guard:
            guard.check()
        moments = disk_triangle_barycentric_moments(points[face, :2], center, radius_m, order)
        if moments[0] == 0:
            continue
        element = elements[owner]
        eid = ng.ElementId(ng.VOL, element.nr)
        finite_element = system.scalar_space.GetFE(eid)
        dofs = np.array(system.scalar_space.GetDofNrs(eid))
        if (dofs < 0).any() or dofs.max() >= len(weights):
            raise ValueError('native footprint needs unconstrained scalar H1 coefficients')
        # Check the reference-vertex convention rather than assuming that
        # the origin belongs to the first physical vertex.
        at_vertices = np.array([finite_element.CalcShape(*p) for p in reference_vertices])
        if not np.allclose(at_vertices[:, :4], np.eye(4), rtol=0, atol=1e-14):
            raise ArithmeticError('native tetrahedron reference vertex ordering changed')
        local = [int(np.flatnonzero(tets[owner] == vertex)[0]) for vertex in face]
        r0, r1, r2 = reference_vertices[local]
        def shapes(x, y):
            return np.array(finite_element.CalcShape(*(r0+x*(r1-r0)+y*(r2-r0))))
        coefficients = np.linalg.solve(vandermonde, np.array([shapes(x, y) for x, y in nodes]))
        for x, y in ((.137, .271), (.419, .083), (.227, .593)):
            measured = np.array([x**i*y**j for i, j in powers]) @ coefficients
            maximum_fit_error = max(maximum_fit_error, float(np.max(abs(measured-shapes(x, y)))))
        weights[dofs] += moments @ coefficients
        area += moments[0]
        covered += 1
    if area <= 0:
        raise ValueError('patch has no positive-area intersection with the top surface')
    if maximum_fit_error > 2e-12:
        raise ArithmeticError('native trace does not match the integrated surface polynomial')
    weights /= area
    if not np.isfinite(weights).all():
        raise ArithmeticError('nonfinite native footprint coefficients')
    clipped = bool(np.any(center < radius_m) or np.any(center > dimensions[:2]-radius_m))
    report = {'integration': 'analytic_disk_native_H1_trace_polynomial', 'order': order,
        'space_order':space_order,'trace_degree':order,
        'requested_center_xy_m': center.tolist(), 'radius_m': radius_m, 'direction': direction.tolist(),
        'area_m2': float(area), 'edge_clipped': clipped,
        'full_disk_area_relative_error': float(abs(area/(math.pi*radius_m**2)-1)),
        'covered_surface_triangles': covered, 'support_coefficients': int(np.count_nonzero(abs(weights) > 1e-13)),
        'negative_coefficients': int(np.count_nonzero(weights < -1e-13)),
        'maximum_trace_fit_absolute_error': maximum_fit_error, 'trace_vandermonde_condition': condition,
        'extraction_workspace_screen_mib': workspace_mib,
        'coefficient_layout': getattr(system,'screen',{}).get('coefficient_layout','component_major_hierarchical_H1'),
        'work_conjugate': True}
    return (direction[:, None]*weights).ravel(), report
