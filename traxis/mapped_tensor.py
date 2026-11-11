"""Mapped GLL elasticity with explicit connectivity and positive nodal mass."""
from dataclasses import dataclass
import math

import numpy as np

from traxis.config import Material
from traxis.recess_hex import RecessHexMesh, evaluate_geometry
from traxis.tensor_hex import line_rule


@dataclass(frozen=True)
class MappedTensorSystem:
    mesh: RecessHexMesh
    material: Material
    inverse_jacobians: np.ndarray
    volume_weights_m3: np.ndarray
    mass_kg: np.ndarray
    stiffness_majorant_kg_s2: np.ndarray
    eigenvalue_upper_s2: float
    stable_dt_s: float
    screen: dict


def _states(values, dofs):
    values = np.asarray(values)
    vector = values.ndim == 1
    if (np.iscomplexobj(values) or values.ndim not in (1, 2) or values.shape[-1] != dofs
            or not np.isfinite(values).all() or (not vector and not 1 <= len(values) <= 32)):
        raise ValueError('mapped state needs one finite vector or at most 32 row vectors')
    return np.asarray(values, dtype=float).reshape(-1, dofs), vector


def local_action(values, order, inverse_jacobians, volume_weights_m3, material):
    """Gradient/stress/weak-divergence contractions, without a dense Ke."""
    _, _, derivative, *_ = line_rule(order)
    q, n = order+1, (order+1)**3
    states, vector = _states(values, 3*n)
    inverse, volume = np.asarray(inverse_jacobians), np.asarray(volume_weights_m3)
    if (inverse.shape != (n, 3, 3) or volume.shape != (n,) or np.iscomplexobj(inverse)
            or np.iscomplexobj(volume) or not np.isfinite(inverse).all() or not np.isfinite(volume).all()
            or np.any(volume <= 0) or not isinstance(material, Material)):
        raise ValueError('mapped element needs finite inverse Jacobians and positive volume weights')
    u = states.reshape((-1, 3, q, q, q))
    reference = np.stack([np.einsum('il,bcljk->bcijk', derivative, u),
        np.einsum('jl,bcilk->bcijk', derivative, u),
        np.einsum('kl,bcijl->bcijk', derivative, u)], axis=2).reshape((-1, 3, 3, n))
    gradient = np.einsum('bcin,nij->bcjn', reference, inverse)
    lam, mu = material.lame
    stress = mu*(gradient+gradient.swapaxes(1, 2))
    trace = sum(gradient[:, i, i] for i in range(3))
    for i in range(3):
        stress[:, i, i] += lam*trace
    flux = (np.einsum('bcjn,nij->bcin', stress, inverse)*volume).reshape((-1, 3, 3, q, q, q))
    result = (np.einsum('il,bcijk->bcljk', derivative, flux[:, :, 0])
        + np.einsum('jl,bcijk->bcilk', derivative, flux[:, :, 1])
        + np.einsum('kl,bcijk->bcijl', derivative, flux[:, :, 2])).reshape((-1, 3*n))
    if not np.isfinite(result).all():
        raise ArithmeticError('mapped action produced nonfinite values')
    return result[0] if vector else result


def _local_majorant(order, inverse, volume, material):
    """A diagonal energy majorant from absolute reference derivatives.

    C=max(2*mu,3*lambda+2*mu) bounds isotropic strain energy by C*|grad u|².
    Weighted Cauchy-Schwarz bounds each squared gradient by a diagonal nodal
    quadratic form. Absolute inverse-Jacobian contractions keep the bound
    valid without forming the physical gradient matrix or dense stiffness.
    """
    q = order+1
    derivative = abs(line_rule(order)[2])
    i, j, k = np.indices((q, q, q)).reshape(3, -1)
    row_sum = derivative.sum(axis=1)
    reference_sum = np.stack([row_sum[i], row_sum[j], row_sum[k]], axis=1)
    inverse_abs = abs(inverse)
    physical_sum = np.einsum('nra,nr->na', inverse_abs, reference_sum)
    lam, mu = material.lame
    constitutive_upper = max(2*mu, 3*lam+2*mu)
    flux = (constitutive_upper*volume[:, None]
        * np.einsum('na,nra->nr', physical_sum, inverse_abs)).T.reshape((3, q, q, q))
    return (np.einsum('il,ijk->ljk', derivative, flux[0])
        + np.einsum('jl,ijk->ilk', derivative, flux[1])
        + np.einsum('kl,ijk->ijl', derivative, flux[2])).ravel()


def build_system(mesh, material, guard=None, *, max_bytes=256*1024**2):
    if not isinstance(mesh, RecessHexMesh) or not isinstance(material, Material):
        raise ValueError('mapped elasticity needs a checked recess mesh and material')
    if mesh.geometry_order > mesh.order:
        raise ValueError('geometry degree must not exceed displacement degree for rigid motions')
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('mapped storage bound must be a positive integer')
    ns, ne, q = len(mesh.points_m), len(mesh.element_nodes), mesh.order+1
    # Inverse geometry, volume, positive mass/majorant, state/action vectors
    # and a bounded local contraction workspace; mesh storage is separate.
    predicted = ne*q**3*10*8 + ns*3*8*68 + 32*3*q**3*24*8
    if predicted > max_bytes:
        raise MemoryError('mapped operator exceeds its preallocation screen')
    nodes, weights, *_ = line_rule(mesh.order)
    reference = np.stack(np.meshgrid(nodes, nodes, nodes, indexing='ij'), axis=-1).reshape(-1, 3)
    product = np.kron(np.kron(weights, weights), weights)
    inverse = np.empty((ne, q**3, 3, 3)); volume = np.empty((ne, q**3))
    mass, majorant = np.zeros(ns), np.zeros(ns)
    minimum, condition = math.inf, 0.
    for e, ids in enumerate(mesh.element_nodes):
        if guard:
            guard.check()
        physical, jac = evaluate_geometry(mesh, e, reference)
        determinant, local_condition = np.linalg.det(jac), np.linalg.cond(jac)
        if (not np.isfinite(jac).all() or np.any(determinant <= 0) or np.any(local_condition > 1e6)
                or np.max(abs(physical-mesh.points_m[ids])) > 1e-12*max(mesh.plate.width_m, mesh.plate.height_m)):
            raise ValueError('mapped GLL geometry is invalid or disagrees with displacement nodes')
        inverse[e], volume[e] = np.linalg.inv(jac), product*determinant
        np.add.at(mass, ids, material.density_kg_m3*volume[e])
        np.add.at(majorant, ids, _local_majorant(mesh.order, inverse[e], volume[e], material))
        minimum, condition = min(minimum, float(determinant.min())), max(condition, float(local_condition.max()))
    if np.any(mass <= 0) or np.any(majorant <= 0) or not np.isfinite(mass).all() or not np.isfinite(majorant).all():
        raise ArithmeticError('mapped positive mass or diagonal stiffness bound failed')
    upper = float(np.max(majorant/mass))
    if not math.isfinite(upper) or upper <= 0:
        raise ArithmeticError('mapped stability bound is invalid')
    screen = dict(vector_dofs=3*ns, elements=ne, order=mesh.order, geometry_order=mesh.geometry_order,
        operator_storage_screen_bytes=predicted, minimum_gll_jacobian=minimum,
        maximum_gll_jacobian_condition=condition, stiffness_rule='gll', mass_rule='positive_gll',
        stability_bound='diagonal_energy_majorant', dense_element_matrices_stored=False)
    mass, majorant = np.tile(mass, 3), np.tile(majorant, 3)
    for value in (inverse, volume, mass, majorant):
        value.setflags(write=False)
    return MappedTensorSystem(mesh, material, inverse, volume, mass, majorant, upper, 2/math.sqrt(upper), screen)


def action(system, values, guard=None):
    ns = len(system.mesh.points_m)
    states, vector = _states(values, 3*ns)
    states = states.reshape((-1, 3, ns)); answer = np.zeros_like(states)
    for e, ids in enumerate(system.mesh.element_nodes):
        if guard:
            guard.check()
        local = local_action(states[:, :, ids].reshape((len(states), -1)), system.mesh.order,
            system.inverse_jacobians[e], system.volume_weights_m3[e], system.material).reshape((len(states), 3, -1))
        for b in range(len(states)):
            for c in range(3):
                np.add.at(answer[b, c], ids, local[b, c])
    answer = answer.reshape((len(states), 3*ns))
    return answer[0] if vector else answer


def assemble_bounded(system, guard=None, *, max_dofs=2000):
    """Dense audit reference only; production action stays matrix-free."""
    ns = len(system.mesh.points_m); ndof = 3*ns; q = system.mesh.order+1; local_dofs = 3*q**3
    if type(max_dofs) is not int or not 24 <= max_dofs <= 2400 or ndof > max_dofs:
        raise MemoryError('mapped dense assembly exceeds the bounded-reference screen')
    result = np.zeros((ndof, ndof))
    for e, ids in enumerate(system.mesh.element_nodes):
        if guard:
            guard.check()
        matrix = np.empty((local_dofs, local_dofs))
        for first in range(0, local_dofs, 32):
            last = min(first+32, local_dofs); states = np.zeros((last-first, local_dofs))
            states[np.arange(last-first), np.arange(first, last)] = 1.
            matrix[:, first:last] = local_action(states, system.mesh.order, system.inverse_jacobians[e],
                system.volume_weights_m3[e], system.material).T
        indices = np.concatenate([ids+c*ns for c in range(3)])
        result[np.ix_(indices, indices)] += matrix
    return result
