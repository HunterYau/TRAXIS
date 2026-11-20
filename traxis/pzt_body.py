"""Resolved 3D PZT bodies tied to the existing plate's surface interpolation.

Full consistent body mass, condensed electrostatics and reciprocal force.
Perfect bonding is explicit; no unknown adhesive or instrument is invented.
The disk boundary is a polygon, whose area error is reported. This is a
numerical model requiring refinement, not a calibrated real transducer.
"""
from dataclasses import dataclass
import math
from types import SimpleNamespace

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix

from traxis.piezoelectric import assemble_piezoelectric, electrode_port
from traxis.pzt_receiver import _point_weights
from traxis.pzt_bond import interface_operators


def disk_mesh(geometry, *, sectors=24, rings=2, layers=2):
    if (type(sectors) is not int or not 8 <= sectors <= 96
            or type(rings) is not int or not 1 <= rings <= 12
            or type(layers) is not int or not 1 <= layers <= 6
            or (1+sectors*rings)*(layers+1) > 1024):
        raise ValueError('disk resolution exceeds bounded 1024-node component')
    angles = np.arange(sectors)*2*np.pi/sectors
    xy = np.vstack([np.zeros((1, 2)), *[
        geometry.radius_m*j/rings*np.column_stack((np.cos(angles), np.sin(angles)))
        for j in range(1, rings+1)]])
    triangles = np.sort(Delaunay(xy).simplices, axis=1)
    n = len(xy)
    points = np.vstack([np.column_stack((xy, np.full(n, z)))
        for z in np.linspace(0, geometry.thickness_m, layers+1)])
    tets = []
    # Sorted prism vertices give the same diagonal on every shared side.
    for layer in range(layers):
        for triangle in triangles:
            a, b, c = triangle+layer*n
            A, B, C = a+n, b+n, c+n
            tets.extend(((a, b, c, C), (a, b, B, C), (a, A, B, C)))
    if len(tets) > 8192:
        raise ValueError('disk resolution exceeds bounded 8192-element component')
    return SimpleNamespace(points_m=points, tetrahedra=np.array(tets, dtype=int)), n


@dataclass
class BodyAttachment:
    indices: np.ndarray
    stiffness: np.ndarray
    mass: np.ndarray
    coupling: np.ndarray
    capacitance_f: float
    total_capacitance_f: float
    next_dof: int
    transform: object
    points_m: np.ndarray
    free_body_dofs: np.ndarray
    report: dict


def attach_disk(layout, material, geometry, *, first_dof=None,
                electrical='open', input_capacitance_f=0., bond='perfect', adhesive=None,
                sectors=24, rings=2, layers=2, max_workspace_mib=512, guard=None):
    """Eliminate tied surface DOFs by u_body=T*u_joint, retaining T' M T.

    electrical='voltage': prescribed voltage, force=-g*V (short if V=0).
    electrical='open': zero free charge, K=Ksc+g*g'/(Cp+Cin).
    Resistive circuit dynamics and unknown glue are deliberately unsupported.
    """
    if bond not in ('perfect','elastic_interface') or (bond=='elastic_interface' and adhesive is None):
        raise ValueError('bond requires perfect tie or an explicit elastic-interface profile')
    if electrical not in ('open', 'voltage'):
        raise ValueError('electrical state must be open or prescribed voltage')
    if (not math.isfinite(input_capacitance_f) or input_capacitance_f < 0
            or not math.isfinite(max_workspace_mib) or not 0 < max_workspace_mib <= 4096):
        raise ValueError('invalid capacitance or workspace budget')
    if not np.array_equal(layout.transform, np.eye(3)) or geometry.in_plane_angle_rad != 0:
        raise ValueError('body attachment currently requires aligned plate and material axes')
    center = np.array(geometry.center_xy_m); radius = geometry.radius_m
    if np.any(center-radius < 0) or np.any(center+radius > [layout.plate.width_m, layout.plate.height_m]):
        raise ValueError('entire disk must lie on plate')
    host_dofs = layout.screen['vector_dofs']
    first_dof = host_dofs if first_dof is None else first_dof
    if type(first_dof) is not int or first_dof < host_dofs:
        raise ValueError('body degrees of freedom must follow the plate')
    mesh, face_nodes = disk_mesh(geometry, sectors=sectors, rings=rings, layers=layers)
    n = len(mesh.points_m); lower = np.arange(face_nodes); upper = lower+layers*face_nodes
    contact = lower if geometry.surface == 'top' else upper
    z = layout.plate.thickness_m if geometry.surface == 'top' else 0.
    xyz = mesh.points_m.copy(); xyz[:, :2] += center
    xyz[:, 2] += z if geometry.surface == 'top' else -geometry.thickness_m
    tied = np.concatenate([contact+c*n for c in range(3)])
    free = np.arange(3*n) if bond=='elastic_interface' else np.setdiff1d(np.arange(3*n), tied)
    rows, cols, values = [], [], []
    for node in contact:
        ids, weights = _point_weights(layout, (*xyz[node, :2], z))
        # At a GLL surface the normal basis is exactly Kronecker. Polynomial
        # roundoff must not create fictitious ties to interior plate nodes.
        nz = layout.cells[2]*layout.order+1
        on_face = ids % nz == (nz-1 if geometry.surface == 'top' else 0)
        ids, weights = ids[on_face], weights[on_face]
        if abs(weights.sum()-1) > 1e-11:
            raise ArithmeticError('surface interpolation lost partition of unity')
        for c in range(3):
            rows.extend([node+c*n]*len(ids)); cols.extend(ids+c*host_dofs//3); values.extend(weights)
    host_rows,host_cols,host_values=rows.copy(),cols.copy(),values.copy()
    if bond=='elastic_interface':rows,cols,values=[],[],[]
    rows.extend(free); cols.extend(first_dof+np.arange(len(free))); values.extend(np.ones(len(free)))
    indices=np.unique(cols+host_cols)
    inverse=np.searchsorted(indices,cols)
    count = len(indices)
    # Dense body condensation and local joint matrices, including temporaries.
    forecast = (12*(3*n)**2+10*count**2)*8
    if forecast > max_workspace_mib*1024**2:
        raise MemoryError('PZT attachment exceeds declared workspace')
    if guard: guard.check(force=True)
    transform = coo_matrix((values, (rows, inverse)), shape=(3*n, count)).tocsr()
    port = electrode_port(assemble_piezoelectric(mesh, material, guard=guard), lower, upper)
    def project(matrix):
        answer = transform.T @ (transform.T @ matrix.T).T
        return (answer+answer.T)*.5
    stiffness = project(port.stiffness_sc); mass = project(port.mass)
    bond_report=None
    if bond=='elastic_interface':
        host_trace=coo_matrix((host_values,(host_rows,np.searchsorted(indices,host_cols))),
            shape=(3*n,count)).tocsr()[tied]
        bond_k,bond_m,bond_report=interface_operators(transform[tied],host_trace,
            xyz[contact,:2],adhesive)
        stiffness+=bond_k;mass+=bond_m
    coupling = np.asarray(transform.T @ port.coupling_n_v)*geometry.poling_sign
    capacitance = port.capacitance_f+input_capacitance_f
    if electrical == 'open':
        stiffness += np.outer(coupling, coupling)/capacitance
    polygon_area = sectors*.5*radius**2*math.sin(2*math.pi/sectors)
    report = dict(material_id=material.id, material_provenance=material.provenance,
        center_xy_m=center.tolist(), surface=geometry.surface, radius_m=radius,
        thickness_m=geometry.thickness_m, poling_sign=geometry.poling_sign,
        sectors=sectors, rings=rings, layers=layers, nodes=n, tetrahedra=len(mesh.tetrahedra),
        body_mass_kg=material.density_kg_m3*polygon_area*geometry.thickness_m,
        circular_area_relative_error=1-polygon_area/(math.pi*radius**2),
        capacitance_f=port.capacitance_f, input_capacitance_f=input_capacitance_f,
        electrical=electrical, bond='assumed perfect tie; not measured glue properties' if bond=='perfect' else 'assumed finite elastic adhesive interface',
        adhesive=bond_report,
        workspace_screen_mib=forecast/1024**2,
        limitations=['linear tetrahedra and polygon boundary require refinement',
            'nominal material tensors, not specimen calibration',
            'full face electrodes; no wrap-around electrode or solder/wire geometry',
            'no resolved adhesive volume or adhesive loss; resistive dynamics belong to transient solver'])
    return BodyAttachment(indices, stiffness, mass, coupling, port.capacitance_f,
        capacitance, first_dof+len(free), transform, xyz, free, report)


def merge_attachments(host_mass, attachments, *, max_workspace_mib=512, guard=None):
    """Merge overlapping supports before inverting mass; no diagonal lumping."""
    host_mass = np.asarray(host_mass)
    if (host_mass.ndim != 1 or np.iscomplexobj(host_mass) or not np.isfinite(host_mass).all()
            or np.any(host_mass <= 0) or not attachments
            or not math.isfinite(max_workspace_mib) or not 0 < max_workspace_mib <= 4096):
        raise ValueError('positive host mass, attachments and bounded workspace required')
    next_dof = len(host_mass)
    for i, body in enumerate(attachments):
        for previous in attachments[:i]:
            a,b=body.report,previous.report
            if (a['surface']==b['surface'] and
                    np.linalg.norm(np.array(a['center_xy_m'])-b['center_xy_m']) < a['radius_m']+b['radius_m']):
                raise ValueError('physical PZT bodies overlap on the same plate surface')
    for body in attachments:
        extra = body.indices[body.indices >= len(host_mass)]
        if not np.array_equal(extra, np.arange(next_dof, body.next_dof)):
            raise ValueError('body DOFs must be appended consecutively without overlap')
        next_dof = body.next_dof
    groups = []
    for body in attachments:
        indices = set(body.indices.tolist()); members = [body]; remaining = []
        for group_indices, group_members in groups:
            if indices.intersection(group_indices):
                indices.update(group_indices); members.extend(group_members)
            else:
                remaining.append((group_indices, group_members))
        groups = remaining+[(indices, members)]
    forecast = sum(10*len(ids)**2*8 for ids, _ in groups)
    if forecast > max_workspace_mib*1024**2:
        raise MemoryError('merged body supports exceed declared workspace')
    diagonal = np.r_[host_mass, np.zeros(next_dof-len(host_mass))]
    patches = []
    for ids, bodies in groups:
        if guard: guard.check(force=True)
        indices = np.array(sorted(ids), dtype=int); count = len(indices)
        mass = np.diag(diagonal[indices]); stiffness = np.zeros((count, count))
        for body in bodies:
            local = np.searchsorted(indices, body.indices); ix = np.ix_(local, local)
            mass[ix] += body.mass; stiffness[ix] += body.stiffness
        factor = cho_factor(mass, lower=True)
        inverse = cho_solve(factor, np.eye(count))
        error = np.linalg.norm(mass@inverse-np.eye(count), ord=np.inf)
        if not np.isfinite(inverse).all() or error > 1e-8:
            raise ArithmeticError('coupled mass inverse failed identity check')
        upper = float(eigh(stiffness, mass, subset_by_index=[count-1, count-1], eigvals_only=True)[0])
        if not math.isfinite(upper) or upper <= 0:
            raise ArithmeticError('invalid coupled component stability bound')
        diagonal[indices] = np.diag(mass)
        patches.append(SimpleNamespace(indices=indices, mass=mass, inverse=inverse,
            stiffness=stiffness, eigenvalue_upper_s2=upper*(1+1e-10), inverse_error=float(error)))
    return diagonal, patches
