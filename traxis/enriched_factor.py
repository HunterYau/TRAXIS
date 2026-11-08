"""Exact reference-gradient factorization and conservative local CFL bounds."""

from functools import lru_cache
import numpy as np

from traxis.enriched_tet import WEIGHTS,reference_integrals


@lru_cache(maxsize=1)
def gradient_factor():
    """Factor the gradient Gram matrix; no displacement modes are removed."""
    _,gram = reference_integrals()
    matrix = gram.transpose(0,2,1,3).reshape(45,45)
    values,vectors = np.linalg.eigh((matrix+matrix.T)*.5)
    active = values>values[-1]*1e-11
    factor = (np.sqrt(values[active])[:,None]*vectors[:,active].T)
    error = float(np.linalg.norm(factor.T@factor-matrix)/np.linalg.norm(matrix))
    if active.sum()!=13 or values[0]<-1e-11*values[-1] or error>1e-11:
        raise ArithmeticError('enriched gradient factorization failed its rank/identity check')
    factor = factor.reshape(13,3,15)
    factor.setflags(write=False)
    return factor,error


def local_stiffness_batch(inverses,material):
    """Exact local matrices divided by absolute Jacobian determinant."""
    inverses = np.asarray(inverses)
    if (np.iscomplexobj(inverses) or inverses.ndim!=3 or inverses.shape[1:]!=(3,3)
            or not 1<=len(inverses)<=1024 or not np.isfinite(inverses).all()):
        raise ValueError('local stability needs bounded finite inverse Jacobians')
    _,reference = reference_integrals()
    gram = np.einsum('eda,efb,dfij->eabij',inverses,inverses,reference,optimize=True)
    lam,mu = material.lame
    blocks = lam*gram+mu*gram.swapaxes(1,2)
    trace = np.einsum('eddij->eij',gram)
    for component in range(3):
        blocks[:,component,component] += mu*trace
    stiffness = blocks.transpose(0,1,3,2,4).reshape(-1,45,45)
    if not np.isfinite(stiffness).all():
        raise ArithmeticError('nonfinite batch stiffness')
    return stiffness


def local_stability_data(inverses,material):
    """Local Gershgorin upper bounds also bound the assembled M^-1 K.

    Positive element nodal masses sum to the global mass. Applying each
    local Rayleigh bound and summing proves the global bound by their maximum.
    Determinants cancel from the normalized local matrix.
    """
    stiffness = local_stiffness_batch(inverses,material)
    diagonal = np.diagonal(stiffness,axis1=1,axis2=2).copy()
    invsqrt = 1/np.sqrt(np.tile(material.density_kg_m3*WEIGHTS,3))
    normalized = stiffness*invsqrt[None,:,None]*invsqrt[None,None,:]
    upper = np.max(np.sum(abs(normalized),axis=2),axis=1)*(1+1e-12)
    if not np.isfinite(upper).all() or np.any(upper<=0) or np.any(diagonal<0):
        raise ArithmeticError('local stiffness or stability bound is invalid')
    return upper,diagonal


def streamed_action(topology,material,values,guard=None,*,chunk_elements=128):
    """Independent CPU action from exact local matrices, without global CSR."""
    values = np.asarray(values)
    n = len(topology.points_m)
    if (type(chunk_elements) is not int or not 1<=chunk_elements<=1024
            or np.iscomplexobj(values) or values.shape!=(3*n,) or not np.isfinite(values).all()):
        raise ValueError('streamed enriched action needs bounded real coefficient data')
    geometry = topology.geometry
    result = np.zeros(3*n)
    for start in range(0,len(geometry.tetrahedra),chunk_elements):
        if guard:
            guard.check()
        corners = geometry.points_m[geometry.tetrahedra[start:start+chunk_elements]]
        jacobian = (corners[:,:3]-corners[:,3:4]).transpose(0,2,1)
        ids = topology.element_nodes[start:start+chunk_elements]
        vector_ids = np.concatenate([ids+c*n for c in range(3)],axis=1)
        stiffness = local_stiffness_batch(np.linalg.inv(jacobian),material)
        forces = np.einsum('eij,ej->ei',stiffness,values[vector_ids])*abs(np.linalg.det(jacobian))[:,None]
        np.add.at(result,vector_ids,forces)
    if not np.isfinite(result).all():
        raise ArithmeticError('nonfinite streamed enriched action')
    return result
