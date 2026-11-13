"""A local conforming recess grid embedded in a Cartesian plate exterior."""
from dataclasses import dataclass
import math

import numpy as np

from traxis.compact_recess import _extrude, _faces
from traxis.config import CircularRecess, Plate
from traxis.recess_hex import RecessHexMesh, _face_vertices, _planar_map
from traxis.tensor_hex import line_rule


@dataclass(frozen=True)
class EmbeddedRecessMesh(RecessHexMesh):
    patch_bounds_m: tuple
    exterior_cells: tuple
    outer_cell_size_m: float


def _definitions(plate, recess, angular, radial, bounds, spacing):
    if bounds is None:
        hx = min(2*recess.radius_m, (recess.radius_m+min(recess.x_m, plate.width_m-recess.x_m))/2)
        hy = min(2*recess.radius_m, (recess.radius_m+min(recess.y_m, plate.height_m-recess.y_m))/2)
        bounds = (recess.x_m-hx, recess.y_m-hy, recess.x_m+hx, recess.y_m+hy)
    bounds = np.asarray(bounds)
    if (bounds.shape != (4,) or np.iscomplexobj(bounds) or not np.isfinite(bounds).all()
            or not math.isfinite(spacing) or spacing <= 0):
        raise ValueError('embedded grid needs finite patch bounds and positive exterior spacing')
    x0, y0, x1, y1 = map(float, bounds)
    r, cx, cy = recess.radius_m, recess.x_m, recess.y_m
    if not (0 < x0 < cx-r < cx+r < x1 < plate.width_m and 0 < y0 < cy-r < cy+r < y1 < plate.height_m):
        raise ValueError('embedded patch must enclose the recess strictly inside the plate')
    lengths = (x0, plate.width_m-x1, y0, plate.height_m-y1)
    ratios = [length/spacing for length in lengths]
    if any(not math.isfinite(value) or value > 512 for value in ratios):
        raise ValueError('embedded exterior subdivisions exceed 512 per strip')
    # Avoid an extra strip cell when a nominal integer quotient differs by
    # a few floating-point ulps. The resulting sizes still have roundoff only.
    counts = tuple(max(1, math.ceil(value-32*np.finfo(float).eps*max(1., value))) for value in ratios)
    nx, ny = (counts[0], angular, counts[1]), (counts[2], angular, counts[3])
    grid_id = lambda i, j: 4*i+j
    remap = {i:16+i for i in range(4)} | {i:16+i for i in range(4, 8)}
    remap.update({8:grid_id(2, 1), 9:grid_id(2, 2), 10:grid_id(1, 2), 11:grid_id(1, 1)})
    blocks = []
    for face in range(9):
        vertices = tuple(remap[i] for i in _face_vertices(face))
        blocks.append((vertices, angular if face == 0 else radial[0 if face <= 4 else 1], angular, None))
    xs, ys = (0., x0, x1, plate.width_m), (0., y0, y1, plate.height_m)
    for i in range(3):
        for j in range(3):
            if (i, j) == (1, 1):
                continue
            vertices = (grid_id(i, j), grid_id(i+1, j), grid_id(i+1, j+1), grid_id(i, j+1))
            blocks.append((vertices, nx[i], ny[j], (xs[i], ys[j], xs[i+1], ys[j+1])))
    return (x0, y0, x1, y1), counts, blocks


def storage_screen(plate, recess, order, *, geometry_order=3, defective=True, angular_cells=12,
        radial_cells=(2, 6), layers=(1, 1), patch_bounds_m=None, outer_cell_size_m=.005,
        max_cells=1000000, max_vector_dofs=12000000, max_bytes=512*1024**2):
    if not isinstance(plate, Plate) or not isinstance(recess, CircularRecess):
        raise ValueError('embedded grid needs a Plate and CircularRecess')
    recess.validate(plate); line_rule(order); line_rule(geometry_order)
    if (type(angular_cells) is not int or not 1 <= angular_cells <= 256
            or len(radial_cells) != 2 or any(type(n) is not int or not 1 <= n <= 256 for n in radial_cells)
            or len(layers) != 2 or any(type(n) is not int or not 1 <= n <= 16 for n in layers)
            or type(defective) is not bool
            or any(type(n) is not int or n <= 0 for n in (max_cells, max_vector_dofs, max_bytes))
            or max_cells > 1000000 or max_vector_dofs > 48000000 or max_bytes > 4096*1024**2):
        raise ValueError('invalid embedded counts or allocation limits')
    bounds, exterior, blocks = _definitions(plate, recess, angular_cells, radial_cells, patch_bounds_m, outer_cell_size_m)
    a = angular_cells; total = sum(nu*nv for _, nu, nv, _ in blocks)
    disk = a*a+4*radial_cells[0]*a; ring = total-disk
    outside = 2*(sum(exterior)+2*a); inside = 4*a
    elements = total*layers[1]+(ring if defective else total)*layers[0]
    def counts(p):
        full = total*p*p+outside*p//2+1
        annular = ring*p*p+(outside+inside)*p//2
        nodes = (annular if defective else full)*layers[0]*p+full*(layers[1]*p+1)
        upper = sum((nu*p+1)*(nv*p+1) for _, nu, nv, _ in blocks)
        return nodes, upper, full, annular
    p, g = counts(order), counts(geometry_order)
    forecast = (p[0]+g[0])*24+elements*((order+1)**3+(geometry_order+1)**3)*4
    forecast += max(p[1], g[1])*320+elements*1024+64*1024**2
    if elements > max_cells or 3*p[0] > max_vector_dofs or forecast > max_bytes:
        raise MemoryError('embedded grid exceeds its preallocation screen')
    return dict(elements=elements, planar_cells=total, disk_cells=disk, scalar_dofs=p[0], vector_dofs=3*p[0],
        geometry_nodes=g[0], planar_nodes=p[2], lower_annular_nodes=p[3], allocation_upper_bytes=forecast,
        patch_bounds_m=list(bounds), exterior_cells=list(exterior), outer_cell_size_m=outer_cell_size_m,
        angular_cells=a, radial_cells=list(radial_cells), layers=list(layers), order=order,
        geometry_order=geometry_order, defective=defective, index_dtype='int32', blocks=len(blocks))


def _node_key(vertices, face, i, j, nu, nv):
    a, b, c, d = vertices
    if i in (0, nu) and j in (0, nv):
        return 'v', {(0, 0):a, (nu, 0):b, (nu, nv):c, (0, nv):d}[i, j]
    if j == 0: first, last, index, count = a, b, i, nu
    elif i == nu: first, last, index, count = b, c, j, nv
    elif j == nv: first, last, index, count = d, c, i, nu
    elif i == 0: first, last, index, count = a, d, j, nv
    else: return 'f', face, i, j
    if first > last: first, last, index = last, first, count-index
    return 'e', first, last, index, count


def _planar(plate, recess, order, geometry_order, angular, bounds, blocks, guard):
    nodes = (line_rule(order)[0]+1)/2; q = order+1
    upper = sum((nu*order+1)*(nv*order+1) for _, nu, nv, _ in blocks)
    descriptors = np.array([(face, i, j) for face, (_, nu, nv, _) in enumerate(blocks)
                            for i in range(nu) for j in range(nv)], dtype=np.int32)
    points = np.empty((upper, 2)); elements = np.empty((len(descriptors), q*q), dtype=np.int32)
    x0, y0, x1, y1 = bounds
    local_plate = Plate(x1-x0, y1-y0, plate.thickness_m)
    local_recess = CircularRecess(recess.x_m-x0, recess.y_m-y0, recess.radius_m, recess.depth_m)
    lookup = {}; count = 0; error = 0.
    for e, (face, iu, iv) in enumerate(descriptors):
        if guard: guard.check()
        vertices, nu, nv, rectangle = blocks[face]
        u, v = np.meshgrid((iu+nodes)/nu, (iv+nodes)/nv, indexing='ij')
        if rectangle is None:
            xy = _planar_map(local_plate, local_recess, int(face), u.ravel(), v.ravel(), angular, geometry_order)+[x0, y0]
        else:
            left, bottom, right, top = rectangle
            xy = np.column_stack([left+(right-left)*u.ravel(), bottom+(top-bottom)*v.ravel()])
        for k, ((i, j), point) in enumerate(zip(np.ndindex((q, q)), xy)):
            key = _node_key(vertices, int(face), int(iu)*order+i, int(iv)*order+j, nu*order, nv*order)
            node = lookup.get(key)
            if node is None:
                node = count; lookup[key] = count; points[count] = point; count += 1
            else:
                error = max(error, float(np.max(abs(points[node]-point))))
            elements[e, k] = node
    if error > 1e-12*max(plate.width_m, plate.height_m):
        raise ArithmeticError('embedded patch/exterior interfaces disagree in physical coordinates')
    return points[:count].copy(), elements, descriptors, error


def _boundary(mesh, element, axis, side):
    face = mesh.cell_faces[element]; u0, _, _, _, z0, z1 = mesh.cell_boxes[element]
    if axis == 2:
        z = z0 if side == 0 else z1
        if z == 0: return 'bottom'
        if z == mesh.plate.thickness_m: return 'top'
        if mesh.defective and z == mesh.recess.depth_m and face <= 4 and side == 0: return 'recess_floor'
    elif (mesh.defective and axis == 0 and side == 0 and 5 <= face <= 8
          and u0 == 0 and z1 <= mesh.recess.depth_m):
        return 'recess_wall'
    q = mesh.order+1
    points = mesh.points_m[np.take(mesh.element_nodes[element].reshape((q,)*3), 0 if side == 0 else -1, axis=axis)]
    tolerance = 1e-12*max(mesh.plate.width_m, mesh.plate.height_m)
    for coordinate, limit in ((0, mesh.plate.width_m), (1, mesh.plate.height_m)):
        if np.max(abs(points[..., coordinate])) < tolerance or np.max(abs(points[..., coordinate]-limit)) < tolerance:
            return 'edge'
    raise ArithmeticError('unmatched embedded face is not a physical boundary')


def make_embedded_recess_mesh(plate, recess, order, *, geometry_order=3, defective=True,
        angular_cells=12, radial_cells=(2, 6), layers=(1, 1), patch_bounds_m=None,
        outer_cell_size_m=.005, guard=None, max_cells=1000000, max_vector_dofs=12000000, max_bytes=512*1024**2):
    screen = storage_screen(plate, recess, order, geometry_order=geometry_order, defective=defective,
        angular_cells=angular_cells, radial_cells=radial_cells, layers=layers, patch_bounds_m=patch_bounds_m,
        outer_cell_size_m=outer_cell_size_m, max_cells=max_cells, max_vector_dofs=max_vector_dofs, max_bytes=max_bytes)
    if guard: guard.check(force=True)
    bounds, exterior, blocks = _definitions(plate, recess, angular_cells, radial_cells, patch_bounds_m, outer_cell_size_m)
    data = []
    for p in (order, geometry_order):
        xy, planar, desc, error = _planar(plate, recess, p, geometry_order, angular_cells, bounds, blocks, guard)
        coordinates, elements, faces, boxes = _extrude(xy, planar, desc, p, plate, recess, layers, defective, guard)
        data.append((coordinates, elements, faces, boxes, error))
    points, elements, faces, boxes, error = data[0]; geometry, geometry_elements, _, _, geometry_error = data[1]
    if len(points) != screen['scalar_dofs'] or len(geometry) != screen['geometry_nodes']:
        raise ArithmeticError('embedded topology differs from its node-count screen')
    q = order+1
    corners = elements.reshape((-1, q, q, q))[:, (0,-1,-1,0,0,-1,-1,0),
        (0,0,-1,-1,0,0,-1,-1), (0,0,0,0,-1,-1,-1,-1)]
    corner_ids, inverse = np.unique(corners, return_inverse=True)
    mesh = EmbeddedRecessMesh(plate, recess, defective, order, geometry_order, angular_cells,
        tuple(radial_cells), tuple(layers), points, elements, geometry, geometry_elements, points[corner_ids],
        inverse.reshape((-1, 8)).astype(np.int32), faces, boxes, (), 0, max(error, geometry_error),
        screen['allocation_upper_bytes'], tuple(bounds), tuple(exterior), outer_cell_size_m)
    boundary, shared = _faces(mesh, guard, classify=_boundary)
    object.__setattr__(mesh, 'boundary_faces', boundary); object.__setattr__(mesh, 'shared_faces', shared)
    for value in vars(mesh).values():
        if isinstance(value, np.ndarray): value.setflags(write=False)
    if guard: guard.check(force=True)
    return mesh
