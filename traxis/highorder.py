"""Optional NGSolve assembly in a scalar-component H1 coefficient basis.

Native libraries are imported only inside the isolated reference worker.
Coefficients are component-major and hierarchical above degree one; they
are not the nodal vectors used by the local/Elmer aperture interfaces.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse


def highorder_dofs(mesh, order):
    if type(order) is not int or not 1 <= order <= 4:
        raise ValueError('reference polynomial degree must lie in [1,4]')
    points, tets = np.asarray(mesh.points_m), np.asarray(mesh.tetrahedra)
    if (np.iscomplexobj(points) or points.ndim != 2 or points.shape[1] != 3 or not 4 <= len(points) <= 20000 or not np.isfinite(points).all()
            or tets.ndim != 2 or tets.shape[1] != 4 or tets.dtype.kind not in 'iu' or not 1 <= len(tets) <= 40000
            or tets.min() < 0 or tets.max() >= len(points) or len(np.unique(tets)) != len(points)):
        raise ValueError('need a bounded complete linear-tetrahedron geometry')
    if len(np.unique(np.sort(tets, axis=1), axis=0)) != len(tets):
        raise ValueError('duplicate reference tetrahedron')
    corners = points[tets]
    edges = corners[:, 1:] - corners[:, :1]
    if np.any(abs(np.linalg.det(edges)) <= 1e-12 * np.linalg.norm(edges, axis=2).max(axis=1)**3):
        raise ValueError('degenerate reference tetrahedron')
    edge_nodes = np.sort(tets[:, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))], axis=2).reshape(-1, 2)
    face_nodes = np.sort(tets[:, ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))], axis=2).reshape(-1, 3)
    edge_count, face_count = len(np.unique(edge_nodes, axis=0)), len(np.unique(face_nodes, axis=0))
    scalar = (len(points) + edge_count * (order-1) + face_count * (order-1) * (order-2) // 2
              + len(tets) * (order-1) * (order-2) * (order-3) // 6)
    local = math.comb(order + 3, 3) * 3
    return {'scalar_dofs': scalar, 'vector_dofs': scalar * 3, 'vertices': len(points), 'edges': edge_count,
            'faces': face_count, 'tetrahedra': len(tets), 'assembly_screen_mib': len(tets) * local**2 * 32 / 1024**2}


@dataclass
class HighOrderSystem:
    stiffness: object
    mass: object
    mesh: object
    scalar_space: object
    space: object
    screen: dict


def assemble_highorder(mesh, material, order, guard=None, *, max_dofs=60000, max_assembly_mib=256):
    screen = highorder_dofs(mesh, order)
    if (type(max_dofs) is not int or not 12 <= max_dofs <= 60000 or not 0 < max_assembly_mib <= 512
            or screen['vector_dofs'] > max_dofs or screen['assembly_screen_mib'] > max_assembly_mib):
        raise MemoryError('high-order reference exceeds its declared degree-of-freedom/assembly screen')
    if guard:
        guard.check(force=True)
    import ngsolve as ng
    import netgen.meshing as nm
    ng.SetNumThreads(1)
    native = nm.Mesh(dim=3)
    vertices = [native.Add(nm.MeshPoint(nm.Pnt(*point))) for point in mesh.points_m]
    region = native.AddRegion('aluminum', dim=3)
    for tet in mesh.tetrahedra:
        native.Add(nm.Element3D(region, [vertices[i] for i in tet]))
    native_mesh = ng.Mesh(native)
    scalar = ng.H1(native_mesh, order=order)
    space = ng.FESpace([scalar, scalar, scalar])
    if scalar.ndof != screen['scalar_dofs'] or space.ndof != screen['vector_dofs']:
        raise ArithmeticError('native H1 degree counts differ from the topological preflight')
    for c in range(3):
        if list(space.Range(c)) != list(range(c*scalar.ndof, (c+1)*scalar.ndof)):
            raise ArithmeticError('native component coefficient ordering changed')
    u, v = space.TnT()
    du, dv = [[ng.grad(field[i]) for i in range(3)] for field in (u, v)]
    lam, mu = material.lame
    elastic = mu/2 * sum((du[i][j]+du[j][i])*(dv[i][j]+dv[j][i]) for i in range(3) for j in range(3))
    elastic += lam * sum(du[i][i] for i in range(3)) * sum(dv[i][i] for i in range(3))
    stiffness = ng.BilinearForm(space, symmetric=False)
    stiffness += elastic * ng.dx(bonus_intorder=2)
    stiffness.Assemble()
    mass = ng.BilinearForm(space, symmetric=False)
    mass += material.density_kg_m3 * sum(u[i]*v[i] for i in range(3)) * ng.dx(bonus_intorder=2)
    mass.Assemble()
    def copy_csr(matrix):
        values, indices, pointers = matrix.CSR()
        if len(values) > 4000000:
            raise MemoryError('assembled high-order operator exceeds 4 million entries')
        result = sparse.csr_matrix((np.array(values), np.array(indices), np.array(pointers)), shape=(space.ndof, space.ndof))
        result.eliminate_zeros()
        result.sort_indices()
        if not np.isfinite(result.data).all() or abs(result - result.T).max() > 1e-11 * max(float(abs(result).max()), 1e-30):
            raise ArithmeticError('native high-order operator is nonfinite or nonsymmetric')
        return result
    k, m = copy_csr(stiffness.mat), copy_csr(mass.mat)
    if (m.diagonal() <= 0).any():
        raise ArithmeticError('native mass has a nonpositive diagonal')
    if guard:
        guard.check(force=True)
    return HighOrderSystem(k, m, native_mesh, scalar, space, screen)


def polynomial_energy_checks(system, material, lengths, transform, order):
    """Exact affine-box integrals for represented monomials and mixed dilation."""
    import ngsolve as ng
    lengths, transform = np.asarray(lengths, float), np.asarray(transform, float)
    inverse = np.linalg.inv(transform)
    volume = abs(float(np.linalg.det(transform))) * float(np.prod(lengths))
    lam, mu = material.lame
    coordinates = (ng.x, ng.y, ng.z)
    normalized = [sum(inverse[i,j] * coordinates[j] for j in range(3)) / lengths[i] for i in range(3)]
    field = ng.GridFunction(system.space)
    rows = []
    def evaluate(coefficients, expected_m, expected_k, name):
        field.vec[:] = 0.
        for c, value in enumerate(coefficients):
            field.components[c].Set(ng.CF(value))
        values = field.vec.FV().NumPy().copy()
        measured_m, measured_k = float(values @ (system.mass @ values)), float(values @ (system.stiffness @ values))
        rows.append({'field': name, 'expected_mass_quadratic_form': expected_m, 'mass_quadratic_form': measured_m,
            'expected_stiffness_quadratic_form': expected_k, 'stiffness_quadratic_form': measured_k,
            'mass_relative_error': abs(measured_m / expected_m - 1), 'stiffness_relative_error': abs(measured_k / expected_k - 1)})
    for axis in range(3):
        derivative = order * inverse[axis] / lengths[axis]
        for component in range(3):
            coefficients = [0., 0., 0.]
            coefficients[component] = normalized[axis]**order
            expected_k = (mu * float(derivative @ derivative) + (lam+mu)*derivative[component]**2) * volume / (2*order-1)
            evaluate(coefficients, material.density_kg_m3 * volume / (2*order+1), expected_k, f'axis-{axis}-component-{component}-degree-{order}')
    gradients = order * inverse / lengths[:,None]
    expected_k = mu * float(np.sum(gradients**2)) * volume / (2*order-1)
    for i in range(3):
        for j in range(3):
            integral = volume / (2*order-1) if i == j else volume / order**2
            expected_k += (mu*gradients[i,j]*gradients[j,i] + lam*gradients[i,i]*gradients[j,j])*integral
    evaluate([value**order for value in normalized], material.density_kg_m3 * volume * 3 / (2*order+1), expected_k, 'mixed-dilation')
    rigid = []
    for coefficients in ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.),(0.,-ng.z,ng.y),(ng.z,0.,-ng.x),(-ng.y,ng.x,0.)):
        field.vec[:] = 0.
        for c, value in enumerate(coefficients):
            field.components[c].Set(ng.CF(value))
        values = field.vec.FV().NumPy().copy()
        rigid.append(float(np.linalg.norm(system.stiffness @ values) / max(np.linalg.norm(system.stiffness.data)*np.linalg.norm(values), 1e-30)))
    return {'polynomials': rows, 'maximum_mass_relative_error': max(r['mass_relative_error'] for r in rows),
            'maximum_stiffness_relative_error': max(r['stiffness_relative_error'] for r in rows),
            'maximum_rigid_relative_force': max(rigid)}


def nodal_evaluation_transform(system, points):
    """Map component-major H1 coefficients to node-major point values."""
    import ngsolve as ng
    points = np.asarray(points, dtype=float)
    scalar = system.scalar_space
    if points.shape != (scalar.ndof, 3) or scalar.ndof > 128 or not np.isfinite(points).all():
        raise ValueError('basis equivalence uses a square point evaluation of at most 128 scalar dofs')
    positions = [system.mesh(*point) for point in points]
    field = ng.GridFunction(scalar)
    field.vec[:] = 0.
    evaluation = np.empty((len(points), scalar.ndof))
    for j in range(scalar.ndof):
        field.vec[j] = 1.
        evaluation[:,j] = [field(point) for point in positions]
        field.vec[j] = 0.
    result = np.zeros((len(points)*3, system.space.ndof))
    for c in range(3):
        result[c::3, c*scalar.ndof:(c+1)*scalar.ndof] = evaluation
    return result
