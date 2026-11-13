"""Array-based extrusion of the conforming planar recess blocks."""
import numpy as np

from traxis.config import CircularRecess, Plate
from traxis.recess_hex import RecessHexMesh, _boundary_name, _planar_key, _planar_map
from traxis.tensor_hex import line_rule


def storage_screen(order, geometry_order, angular_cells, radial_cells, layers, defective,
                   *, max_cells=1000000, max_vector_dofs=12000000, max_bytes=512*1024**2):
    line_rule(order); line_rule(geometry_order)
    if (type(angular_cells) is not int or not 1 <= angular_cells <= 256
            or len(radial_cells) != 2 or any(type(n) is not int or not 1 <= n <= 256 for n in radial_cells)
            or len(layers) != 2 or any(type(n) is not int or not 1 <= n <= 16 for n in layers)
            or type(defective) is not bool
            or any(type(n) is not int or n <= 0 for n in (max_cells, max_vector_dofs, max_bytes))
            or max_cells > 1000000 or max_vector_dofs > 48000000 or max_bytes > 4096*1024**2):
        raise ValueError('invalid compact recess counts or allocation limits')
    a = angular_cells; inner, outer = radial_cells
    disk, ring = a*a+4*inner*a, 4*outer*a
    ne = (disk+ring)*layers[1]+(ring if defective else disk+ring)*layers[0]
    # Exact planar node counts follow the nine-block disk/annulus topology.
    def counts(p):
        total = (a*p+1)**2+4*a*p*(inner*p+outer*p)
        # Each closed annular layer has 4*a*p nodes, including its corners.
        annulus = 4*(outer*p+1)*a*p
        nodes = (annulus if defective else total)*layers[0]*p+total*(layers[1]*p+1)
        upper = (a*p+1)**2+4*(inner*p+1)*(a*p+1)+4*(outer*p+1)*(a*p+1)
        return int(total), int(annulus), int(nodes), int(upper)
    p, g = counts(order), counts(geometry_order)
    # Includes both meshes, integer connectivity, bounded extrusion buffers,
    # a conservative planar Python lookup and face-sort/adjacency workspace.
    predicted = (p[2]+g[2])*3*8 + ne*((order+1)**3+(geometry_order+1)**3)*4
    predicted += max(p[3], g[3])*320 + ne*1024 + 64*1024**2
    if ne > max_cells or 3*p[2] > max_vector_dofs or predicted > max_bytes:
        raise MemoryError('compact recess mesh exceeds its preallocation screen')
    return dict(elements=ne, planar_cells=disk+ring, planar_nodes=p[0], lower_ring_nodes=p[1],
        scalar_dofs=p[2], vector_dofs=3*p[2], geometry_nodes=g[2],
        allocation_upper_bytes=predicted, planar_allocation_nodes=max(p[3], g[3]),
        order=order, geometry_order=geometry_order, angular_cells=a,
        radial_cells=list(radial_cells), layers=list(layers), defective=defective,
        global_node_dictionary=False, index_dtype='int32')


def _planar(plate, recess, order, geometry_order, angular, radial, guard):
    nodes = (line_rule(order)[0]+1)/2
    descriptors = [(face, i, j) for face in range(9)
        for i in range(angular if face == 0 else radial[0 if face <= 4 else 1]) for j in range(angular)]
    upper = sum((angular*order+1 if face == 0 else radial[0 if face <= 4 else 1]*order+1)
                *(angular*order+1) for face in range(9))
    points = np.empty((upper, 2)); elements = np.empty((len(descriptors), (order+1)**2), dtype=np.int32)
    lookup = {}; count = 0; error = 0.
    for e, (face, iu, iv) in enumerate(descriptors):
        if guard: guard.check()
        nu = angular if face == 0 else radial[0 if face <= 4 else 1]
        u, v = np.meshgrid((iu+nodes)/nu, (iv+nodes)/angular, indexing='ij')
        xy = _planar_map(plate, recess, face, u.ravel(), v.ravel(), angular, geometry_order)
        for k, ((i, j), point) in enumerate(zip(np.ndindex((order+1,)*2), xy)):
            key = _planar_key(face, iu*order+i, iv*order+j, nu*order, angular*order)
            node = lookup.get(key)
            if node is None:
                node = count; lookup[key] = count; points[count] = point; count += 1
            else:
                error = max(error, float(np.max(abs(points[node]-point))))
            elements[e, k] = node
    if error > 1e-12*max(plate.width_m, plate.height_m):
        raise ArithmeticError('compact planar interfaces disagree in physical coordinates')
    return points[:count].copy(), elements, np.asarray(descriptors, dtype=np.int32), error


def _extrude(points, planar, descriptors, order, plate, recess, layers, defective, guard):
    ns = len(points); q = order+1; floor = layers[0]*order
    ring_cells = descriptors[:, 0] >= 5
    ring = np.unique(planar[ring_cells]) if defective else np.arange(ns, dtype=np.int32)
    rank = np.full(ns, -1, dtype=np.int32); rank[ring] = np.arange(len(ring), dtype=np.int32)
    nodes = (line_rule(order)[0]+1)/2
    zlower = np.concatenate([(i+nodes[:-1])*recess.depth_m/layers[0] for i in range(layers[0])])
    zupper = np.r_[np.concatenate([recess.depth_m+(i+nodes[:-1])*(plate.thickness_m-recess.depth_m)/layers[1]
                                  for i in range(layers[1])]), plate.thickness_m]
    z = np.r_[zlower, zupper]
    offsets = np.where(np.arange(len(z)) < floor, np.arange(len(z))*len(ring),
                       floor*len(ring)+(np.arange(len(z))-floor)*ns)
    total = floor*len(ring)+(len(z)-floor)*ns
    coordinates = np.empty((total, 3))
    for k, height in enumerate(z):
        if guard: guard.check()
        selected = ring if k < floor else slice(None)
        n = len(ring) if k < floor else ns
        coordinates[offsets[k]:offsets[k]+n, :2] = points[selected]
        coordinates[offsets[k]:offsets[k]+n, 2] = height
    count = len(planar)*layers[1]+(int(ring_cells.sum()) if defective else len(planar))*layers[0]
    elements = np.empty((count, q**3), dtype=np.int32)
    cell_faces = np.empty(count, dtype=np.int32); boxes = np.empty((count, 6)); first = 0
    blocks = int(descriptors[:, 0].max())+1
    block_counts = np.array([descriptors[descriptors[:, 0] == face, 1].max()+1 for face in range(blocks)])
    vertical_counts = np.array([descriptors[descriptors[:, 0] == face, 2].max()+1 for face in range(blocks)])
    for iz in range(sum(layers)):
        active = np.flatnonzero(ring_cells) if defective and iz < layers[0] else np.arange(len(planar))
        planes = iz*order+np.arange(q)
        for begin in range(0, len(active), 512):
            if guard: guard.check()
            chosen = active[begin:begin+512]; ids = planar[chosen]
            local = np.where((planes < floor)[None, None, :], rank[ids][:, :, None], ids[:, :, None])
            stop = first+len(chosen)
            elements[first:stop] = (local+offsets[planes][None, None, :]).reshape(len(chosen), -1)
            desc = descriptors[chosen]; cell_faces[first:stop] = desc[:, 0]
            nu = block_counts[desc[:, 0]]
            nv = vertical_counts[desc[:, 0]]
            boxes[first:stop, :4] = np.column_stack([desc[:, 1]/nu, (desc[:, 1]+1)/nu,
                                                    desc[:, 2]/nv, (desc[:, 2]+1)/nv])
            boxes[first:stop, 4:] = [z[iz*order], z[(iz+1)*order]]
            first = stop
    return coordinates, elements, cell_faces, boxes


def _faces(mesh, guard, *, classify=None):
    ne = len(mesh.element_nodes); q = mesh.order+1; g = mesh.geometry_order+1
    def maps(n):
        array = np.arange(n**3).reshape((n,)*3)
        return np.array([np.take(array, 0 if side == 0 else -1, axis=axis).ravel()
                         for axis in range(3) for side in (0, 1)])
    faces, geometry_faces = maps(q), maps(g)
    corners = np.array([face.reshape(q, q)[np.ix_([0, q-1], [0, q-1])].ravel() for face in faces])
    keys = np.sort(mesh.element_nodes[:, corners].reshape(-1, 4), axis=1)
    order = np.lexsort(tuple(keys[:, i] for i in (3, 2, 1, 0)))
    sorted_keys = keys[order]; same = np.all(sorted_keys[1:] == sorted_keys[:-1], axis=1)
    if np.any(same[1:] & same[:-1]):
        raise ArithmeticError('compact mesh has a nonmanifold material face')
    left, right = order[:-1][same], order[1:][same]
    neighbors = np.full((ne, 6), -1, dtype=np.int32)
    neighbors[left//6, left%6] = right//6; neighbors[right//6, right%6] = left//6
    for first in range(0, len(left), 512):
        if guard: guard.check()
        a, b = left[first:first+512], right[first:first+512]
        for elements, mapping in ((mesh.element_nodes, faces), (mesh.geometry_elements, geometry_faces)):
            one = np.sort(elements[(a//6)[:, None], mapping[a%6]], axis=1)
            two = np.sort(elements[(b//6)[:, None], mapping[b%6]], axis=1)
            if not np.array_equal(one, two):
                raise ArithmeticError('compact shared face has inconsistent polynomial nodes')
    paired = np.r_[same, False] | np.r_[False, same]
    boundary = []
    classify = _boundary_name if classify is None else classify
    for item in order[~paired]:
        e, local = divmod(int(item), 6); axis, side = divmod(local, 2)
        boundary.append((classify(mesh, e, axis, side), e, axis, side))
    seen = np.zeros(ne, dtype=bool); seen[0] = True; pending = [0]
    while pending:
        if guard: guard.check()
        e = pending.pop()
        for neighbor in neighbors[e]:
            if neighbor >= 0 and not seen[neighbor]:
                seen[neighbor] = True; pending.append(int(neighbor))
    if not seen.all():
        raise ArithmeticError('compact recess mesh has disconnected material')
    return tuple(boundary), len(left)


def make_compact_recess_mesh(plate, recess, order, *, geometry_order=3, defective=True,
        angular_cells=4, radial_cells=(1, 1), layers=(1, 1), guard=None,
        max_cells=1000000, max_vector_dofs=12000000, max_bytes=512*1024**2):
    if not isinstance(plate, Plate) or not isinstance(recess, CircularRecess):
        raise ValueError('compact recess geometry needs a Plate and CircularRecess')
    recess.validate(plate)
    if min(recess.x_m, plate.width_m-recess.x_m, recess.y_m, plate.height_m-recess.y_m) <= recess.radius_m:
        raise ValueError('compact block recess must lie strictly inside the plate')
    screen = storage_screen(order, geometry_order, angular_cells, radial_cells, layers, defective,
                            max_cells=max_cells, max_vector_dofs=max_vector_dofs, max_bytes=max_bytes)
    if guard: guard.check(force=True)
    data = []
    for p in (order, geometry_order):
        xy, planar, desc, error = _planar(plate, recess, p, geometry_order, angular_cells, radial_cells, guard)
        xyz, elements, faces, boxes = _extrude(xy, planar, desc, p, plate, recess, layers, defective, guard)
        data.append((xyz, elements, faces, boxes, error))
    points, elements, faces, boxes, error = data[0]
    geom, geom_elements, _, _, geom_error = data[1]
    if len(points) != screen['scalar_dofs'] or len(geom) != screen['geometry_nodes']:
        raise ArithmeticError('compact mesh node count differs from its preallocation screen')
    q = order+1
    corners = elements.reshape((-1, q, q, q))[:, (0, -1, -1, 0, 0, -1, -1, 0),
        (0, 0, -1, -1, 0, 0, -1, -1), (0, 0, 0, 0, -1, -1, -1, -1)]
    corner_ids, inverse = np.unique(corners, return_inverse=True)
    mesh = RecessHexMesh(plate, recess, defective, order, geometry_order, angular_cells,
        tuple(radial_cells), tuple(layers), points, elements, geom, geom_elements, points[corner_ids],
        inverse.reshape((-1, 8)).astype(np.int32), faces, boxes, (), 0, max(error, geom_error), screen['allocation_upper_bytes'])
    boundary, shared = _faces(mesh, guard)
    object.__setattr__(mesh, 'boundary_faces', boundary); object.__setattr__(mesh, 'shared_faces', shared)
    for value in vars(mesh).values():
        if isinstance(value, np.ndarray): value.setflags(write=False)
    if guard: guard.check(force=True)
    return mesh
