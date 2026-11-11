"""Conforming block hexahedra for one flat-bottom backside recess.

Geometry order and displacement order are independent. Circle arcs are
piecewise polynomial approximations; their shape error must be measured.
This module does not select a wave mesh or an explicit time step.
"""

from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial import Polynomial

from traxis.config import CircularRecess, Plate
from traxis.tensor_hex import line_rule


@dataclass(frozen=True)
class RecessHexMesh:
    plate: Plate
    recess: CircularRecess
    defective: bool
    order: int
    geometry_order: int
    angular_cells: int
    radial_cells: tuple
    layers: tuple
    points_m: np.ndarray
    element_nodes: np.ndarray
    geometry_points_m: np.ndarray
    geometry_elements: np.ndarray
    corner_points_m: np.ndarray
    corner_elements: np.ndarray
    cell_faces: np.ndarray
    cell_boxes: np.ndarray
    boundary_faces: tuple
    shared_faces: int
    shared_coordinate_error_m: float
    allocation_upper_bytes: int


def lagrange_values(order, points):
    nodes = line_rule(order)[0]
    points = np.asarray(points, dtype=float)
    if points.ndim != 1 or not np.isfinite(points).all():
        raise ValueError('reference coordinates must be a finite vector')
    values, derivatives = [], []
    for i in range(order + 1):
        basis = Polynomial.fromroots(np.delete(nodes, i)) / np.prod(nodes[i] - np.delete(nodes, i))
        values.append(basis(points))
        derivatives.append(basis.deriv()(points))
    return np.array(values).T, np.array(derivatives).T


def _face_vertices(face):
    # Rings start at the southeast corner and run counterclockwise.
    if face == 0:
        return 3, 0, 1, 2
    side = (face - 1) % 4
    inner = 0 if face <= 4 else 4
    outer = inner + 4
    next_side = (side + 1) % 4
    return inner + side, outer + side, outer + next_side, inner + next_side


def _planar_map(plate, recess, face, u, v, angular_cells, geometry_order):
    center = np.array([recess.x_m, recess.y_m])
    half = recess.radius_m / 2
    if face == 0:
        return center + half * np.column_stack([2*u - 1, 2*v - 1])
    side = (face - 1) % 4
    segment = np.minimum((v * angular_cells).astype(int), angular_cells - 1)
    local = 2 * (v * angular_cells - segment) - 1
    shape, _ = lagrange_values(geometry_order, local)
    nodes = (line_rule(geometry_order)[0] + 1) / 2
    theta = -math.pi/4 + side*math.pi/2 + (segment[:, None] + nodes)*math.pi/(2*angular_cells)
    controls = center + recess.radius_m*np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    circle = np.einsum('qi,qic->qc', shape, controls)
    square = center + half*np.column_stack([np.ones(len(v)), 2*v - 1]) @ np.array(
        [[math.cos(side*math.pi/2), math.sin(side*math.pi/2)],
         [-math.sin(side*math.pi/2), math.cos(side*math.pi/2)]])
    corners = np.array([[plate.width_m, 0], [plate.width_m, plate.height_m],
                        [0, plate.height_m], [0, 0]])
    boundary = (1-v[:, None])*corners[side] + v[:, None]*corners[(side+1) % 4]
    a, b = (square, circle) if face <= 4 else (circle, boundary)
    return (1-u[:, None])*a + u[:, None]*b


def _planar_key(face, i, j, nu, nv):
    a, b, c, d = _face_vertices(face)
    if i in (0, nu) and j in (0, nv):
        return 'v', {(0, 0): a, (nu, 0): b, (nu, nv): c, (0, nv): d}[i, j]
    if j == 0:
        first, last, index, count = a, b, i, nu
    elif i == nu:
        first, last, index, count = b, c, j, nv
    elif j == nv:
        first, last, index, count = d, c, i, nu
    elif i == 0:
        first, last, index, count = a, d, j, nv
    else:
        return 'f', face, i, j
    if first > last:
        first, last, index = last, first, count-index
    return 'e', first, last, index, count


def _connectivity(plate, recess, defective, angular_cells, radial_cells, layers,
                  order, geometry_order, descriptors):
    nodes = (line_rule(order)[0] + 1) / 2
    # Interior GLL nodes need their cell index as well as local node index.
    # All adjacent edges have the same partition; no coordinate rounding is used.
    coordinates, lookup, elements = [], {}, []
    error = 0.
    for face, iu, iv, iz in descriptors:
        nu = angular_cells if face == 0 else radial_cells[0 if face <= 4 else 1]
        nv = angular_cells
        lower = iz < layers[0]
        z0, z1, nz = (0., recess.depth_m, layers[0]) if lower else (
            recess.depth_m, plate.thickness_m, layers[1])
        kz = iz if lower else iz-layers[0]
        u, v, z = np.meshgrid((iu+nodes)/nu, (iv+nodes)/nv,
                             z0+(kz+nodes)*(z1-z0)/nz, indexing='ij')
        xy = _planar_map(plate, recess, face, u.ravel(), v.ravel(), angular_cells, geometry_order)
        physical = np.column_stack([xy, z.ravel()])
        ids = []
        for (i, j, k), point in zip(np.ndindex((order+1,)*3), physical):
            planar = _planar_key(face, iu*order+i, iv*order+j, nu*order, nv*order)
            key = planar, iz*order+k
            if key not in lookup:
                lookup[key] = len(coordinates)
                coordinates.append(point)
            else:
                error = max(error, float(np.max(abs(coordinates[lookup[key]]-point))))
            ids.append(lookup[key])
        elements.append(ids)
    if error > 1e-12*max(plate.width_m, plate.height_m, plate.thickness_m):
        raise ArithmeticError('shared topological nodes disagree in physical coordinates')
    return np.asarray(coordinates), np.asarray(elements, dtype=np.int64), error


def _boundary_name(mesh, element, axis, side):
    face = mesh.cell_faces[element]
    u0, u1, _, _, z0, z1 = mesh.cell_boxes[element]
    if axis == 2:
        z = z0 if side == 0 else z1
        if z == 0:
            return 'bottom'
        if z == mesh.plate.thickness_m:
            return 'top'
        if mesh.defective and z == mesh.recess.depth_m and face <= 4 and side == 0:
            return 'recess_floor'
    if axis == 0 and face >= 5:
        if side == 1 and u1 == 1:
            return 'edge'
        if mesh.defective and side == 0 and u0 == 0 and z1 <= mesh.recess.depth_m:
            return 'recess_wall'
    raise ArithmeticError('unmatched material face is not a declared physical boundary')


def _faces(mesh):
    owners, boundary, shared, neighbors = {}, [], 0, [set() for _ in mesh.element_nodes]
    q, g = mesh.order+1, mesh.geometry_order+1
    for e, (nodes, geometry) in enumerate(zip(mesh.element_nodes, mesh.geometry_elements)):
        nodes, geometry = nodes.reshape((q,)*3), geometry.reshape((g,)*3)
        for axis in range(3):
            for side in (0, 1):
                values = np.take(nodes, 0 if side == 0 else -1, axis=axis)
                geom = np.take(geometry, 0 if side == 0 else -1, axis=axis)
                key = tuple(sorted(values[np.ix_([0, q-1], [0, q-1])].ravel()))
                owners.setdefault(key, []).append((e, axis, side, np.sort(values.ravel()), np.sort(geom.ravel())))
    for entries in owners.values():
        if len(entries) == 2:
            a, b = entries
            if not np.array_equal(a[3], b[3]) or not np.array_equal(a[4], b[4]):
                raise ArithmeticError('shared face has inconsistent displacement or geometry nodes')
            neighbors[a[0]].add(b[0]); neighbors[b[0]].add(a[0]); shared += 1
        elif len(entries) == 1:
            e, axis, side, *_ = entries[0]
            boundary.append((_boundary_name(mesh, e, axis, side), e, axis, side))
        else:
            raise ArithmeticError('more than two material cells share a face')
    seen, pending = {0}, [0]
    while pending:
        for neighbor in neighbors[pending.pop()]:
            if neighbor not in seen:
                seen.add(neighbor); pending.append(neighbor)
    if len(seen) != len(neighbors):
        raise ArithmeticError('recess mesh has disconnected material')
    return tuple(boundary), shared


def make_recess_mesh(plate, recess, order, *, geometry_order=3, defective=True,
                     angular_cells=1, radial_cells=(1, 1), layers=(1, 1),
                     max_cells=10000, max_vector_dofs=100000, max_bytes=256*1024**2):
    """Build a bounded mesh with node identity defined by topology.

    The intact control retains the same recess-floor partition. Increasing
    angular_cells also refines the represented circular geometry; report that
    change separately from pure displacement p-refinement.
    """
    if not isinstance(plate, Plate) or not isinstance(recess, CircularRecess):
        raise ValueError('recess geometry needs a Plate and CircularRecess')
    recess.validate(plate)
    if min(recess.x_m, plate.width_m-recess.x_m, recess.y_m, plate.height_m-recess.y_m) <= recess.radius_m:
        raise ValueError('block recess must lie strictly inside the plate')
    line_rule(order); line_rule(geometry_order)
    if (type(defective) is not bool or type(angular_cells) is not int or not 1 <= angular_cells <= 64
            or len(radial_cells) != 2 or any(type(n) is not int or not 1 <= n <= 64 for n in radial_cells)
            or len(layers) != 2 or any(type(n) is not int or not 1 <= n <= 16 for n in layers)
            or any(type(n) is not int or n <= 0 for n in (max_cells, max_vector_dofs, max_bytes))):
        raise ValueError('invalid block mesh controls or allocation bounds')
    disk = angular_cells**2 + 4*radial_cells[0]*angular_cells
    ring = 4*radial_cells[1]*angular_cells
    count = (disk+ring)*layers[1] + (ring if defective else disk+ring)*layers[0]
    local = count*((order+1)**3+(geometry_order+1)**3)
    # Includes deliberately conservative dictionary/node/face working storage.
    allocation = local*1024 + count*1024
    if count > max_cells or 3*count*(order+1)**3 > max_vector_dofs or allocation > max_bytes:
        raise MemoryError('recess mesh exceeds its preallocation screen')
    descriptors, boxes = [], []
    for iz in range(sum(layers)):
        for face in range(9):
            if defective and iz < layers[0] and face <= 4:
                continue
            nu = angular_cells if face == 0 else radial_cells[0 if face <= 4 else 1]
            z0, z1, nz = (0., recess.depth_m, layers[0]) if iz < layers[0] else (
                recess.depth_m, plate.thickness_m, layers[1])
            kz = iz if iz < layers[0] else iz-layers[0]
            for iu in range(nu):
                for iv in range(angular_cells):
                    descriptors.append((face, iu, iv, iz))
                    boxes.append((iu/nu, (iu+1)/nu, iv/angular_cells, (iv+1)/angular_cells,
                                  z0+kz*(z1-z0)/nz, z0+(kz+1)*(z1-z0)/nz))
    points, elements, error = _connectivity(plate, recess, defective, angular_cells, radial_cells, layers,
                                           order, geometry_order, descriptors)
    geom, geom_elements, geom_error = _connectivity(plate, recess, defective, angular_cells, radial_cells, layers,
                                                  geometry_order, geometry_order, descriptors)
    q = order+1
    corners = elements.reshape((-1, q, q, q))[:, (0, -1, -1, 0, 0, -1, -1, 0),
                       (0, 0, -1, -1, 0, 0, -1, -1), (0, 0, 0, 0, -1, -1, -1, -1)]
    corner_ids, corner_inverse = np.unique(corners, return_inverse=True)
    mesh = RecessHexMesh(plate, recess, defective, order, geometry_order, angular_cells,
        tuple(radial_cells), tuple(layers), points, elements, geom, geom_elements,
        points[corner_ids], corner_inverse.reshape((-1, 8)), np.array([d[0] for d in descriptors]),
        np.asarray(boxes), (), 0, max(error, geom_error), allocation)
    boundary, shared = _faces(mesh)
    object.__setattr__(mesh, 'boundary_faces', boundary)
    object.__setattr__(mesh, 'shared_faces', shared)
    for value in vars(mesh).values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return mesh


def evaluate_geometry(mesh, element, reference_points):
    """Physical position and dX/d(reference coordinate), with columns as axes."""
    points = np.asarray(reference_points)
    if (type(element) is not int or not 0 <= element < len(mesh.element_nodes)
            or points.ndim != 2 or points.shape[1] != 3 or np.iscomplexobj(points)
            or not np.isfinite(points).all() or np.max(abs(points), initial=0) > 1+1e-12):
        raise ValueError('invalid element or reference point array')
    g = mesh.geometry_order+1
    controls = mesh.geometry_points_m[mesh.geometry_elements[element]].reshape((g, g, g, 3))
    values, derivatives = zip(*(lagrange_values(mesh.geometry_order, points[:, axis]) for axis in range(3)))
    physical = np.einsum('qi,qj,qk,ijkc->qc', *values, controls, optimize=True)
    jacobian = np.empty((len(points), 3, 3))
    for axis in range(3):
        factors = list(values); factors[axis] = derivatives[axis]
        jacobian[:, :, axis] = np.einsum('qi,qj,qk,ijkc->qc', *factors, controls, optimize=True)
    return physical, jacobian
