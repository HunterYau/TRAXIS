"""Bounded tensor-product elasticity references with positive GLL mass.

The stiffness rule is explicit: standard GLL quadrature or exact integration.
This module is a small reference, not a full-plate propagation backend.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from numpy.polynomial import Legendre, Polynomial
from scipy import sparse


@lru_cache(maxsize=4,typed=True)
def line_rule(order):
    if type(order) is not int or not 1<=order<=4:
        raise ValueError('tensor reference order must be an integer from one to four')
    polynomial = Legendre.basis(order)
    roots = polynomial.deriv().roots()
    if np.iscomplexobj(roots) and np.max(abs(roots.imag),initial=0)>1e-13:
        raise ArithmeticError('Legendre derivative roots are unexpectedly nonreal')
    nodes = np.r_[-1.,np.sort(roots.real),1.]
    weights = 2/(order*(order+1)*polynomial(nodes)**2)
    basis = [Polynomial.fromroots(np.delete(nodes,i))/np.prod(nodes[i]-np.delete(nodes,i)) for i in range(order+1)]
    derivative = np.array([[p.deriv()(x) for p in basis] for x in nodes])
    def integral(p):
        anti = p.integ()
        return float(anti(1)-anti(-1))
    mass = np.array([[integral(a*b) for b in basis] for a in basis])
    stiffness = np.array([[integral(a.deriv()*b.deriv()) for b in basis] for a in basis])
    cross = np.array([[integral(a.deriv()*b) for b in basis] for a in basis])
    for array in (nodes,weights,derivative,mass,stiffness,cross):
        array.setflags(write=False)
    return nodes,weights,derivative,mass,stiffness,cross


def product(matrices):
    return np.kron(np.kron(matrices[0],matrices[1]),matrices[2])


@lru_cache(maxsize=8,typed=True)
def reference_integrals(order,rule):
    if rule not in ('gll','exact'):
        raise ValueError('tensor stiffness quadrature must be gll or exact')
    _,weights,_,mass,stiffness,cross = line_rule(order)
    selected_mass = mass if rule=='exact' else np.diag(weights)
    gradients = np.empty((3,3,(order+1)**3,(order+1)**3))
    for i in range(3):
        for j in range(3):
            factors = [selected_mass]*3
            if i==j:
                factors[i] = stiffness
            else:
                factors[i],factors[j] = cross,cross.T
            gradients[i,j] = product(factors)
    consistent = product([mass]*3)
    lumped = product([weights]*3)
    for array in (gradients,consistent,lumped):
        array.setflags(write=False)
    return gradients,consistent,lumped


def element_matrices(order,jacobian,material,rule='gll'):
    jacobian = np.asarray(jacobian)
    if (jacobian.shape!=(3,3) or np.iscomplexobj(jacobian) or not np.isfinite(jacobian).all()
            or np.linalg.det(jacobian)<=0 or np.linalg.cond(jacobian)>1e6):
        raise ValueError('tensor reference requires a regular positively oriented affine map')
    gradients,consistent,lumped = reference_integrals(order,rule)
    inverse,det = np.linalg.inv(jacobian),float(np.linalg.det(jacobian))
    physical = np.einsum('ia,jb,ijuv->abuv',inverse,inverse,gradients,optimize=True)*det
    lam,mu = material.lame
    n = (order+1)**3
    k = np.empty((3*n,3*n))
    laplacian = sum(physical[i,i] for i in range(3))
    for a in range(3):
        for b in range(3):
            k[a*n:(a+1)*n,b*n:(b+1)*n] = lam*physical[a,b]+mu*physical[b,a]+(mu*laplacian if a==b else 0.)
    return k,material.density_kg_m3*det*consistent,material.density_kg_m3*det*lumped


@dataclass
class TensorGrid:
    plate: object
    cells: tuple
    order: int
    transform: np.ndarray
    points_m: np.ndarray
    element_nodes: np.ndarray
    corner_points_m: np.ndarray
    corner_elements: np.ndarray


def make_grid(plate,cells,order,transform=None,*,max_dofs=2000):
    nodes = line_rule(order)[0]
    if (len(cells)!=3 or any(type(n) is not int or not 1<=n<=16 for n in cells)
            or type(max_dofs) is not int or not 24<=max_dofs<=2000):
        raise ValueError('invalid bounded tensor grid')
    scalar = math.prod(n*order+1 for n in cells)
    if 3*scalar>max_dofs or (3*scalar)**2*48>128*1024**2:
        raise MemoryError('tensor grid exceeds the small-reference allocation screen')
    transform = np.eye(3) if transform is None else np.asarray(transform)
    if (transform.shape!=(3,3) or np.iscomplexobj(transform) or not np.isfinite(transform).all()
            or np.linalg.det(transform)<=0 or np.linalg.cond(transform)>1e4):
        raise ValueError('tensor grid needs a regular positive affine transform')
    lengths = (plate.width_m,plate.height_m,plate.thickness_m)
    axes = [np.r_[np.concatenate([(c+(nodes[:-1]+1)/2)*length/count for c in range(count)]),length]
            for count,length in zip(cells,lengths)]
    points = np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)@transform.T
    indices = np.arange(scalar).reshape(tuple(n*order+1 for n in cells))
    elements = np.array([indices[i*order:(i+1)*order+1,j*order:(j+1)*order+1,k*order:(k+1)*order+1].ravel()
        for i in range(cells[0]) for j in range(cells[1]) for k in range(cells[2])])
    corner_shape = tuple(n+1 for n in cells)
    corner_ids = np.arange(math.prod(corner_shape)).reshape(corner_shape)
    corners = np.stack(np.meshgrid(*[np.linspace(0,length,n+1) for n,length in zip(cells,lengths)],indexing='ij'),axis=-1).reshape(-1,3)@transform.T
    corner_elements = np.array([[corner_ids[i,j,k],corner_ids[i+1,j,k],corner_ids[i+1,j+1,k],corner_ids[i,j+1,k],
        corner_ids[i,j,k+1],corner_ids[i+1,j,k+1],corner_ids[i+1,j+1,k+1],corner_ids[i,j+1,k+1]]
        for i in range(cells[0]) for j in range(cells[1]) for k in range(cells[2])])
    return TensorGrid(plate,tuple(cells),order,transform,points,elements,corners,corner_elements)


@dataclass
class TensorSystem:
    stiffness: object
    consistent_mass: object
    mass_kg: np.ndarray
    grid: TensorGrid
    stable_dt_s: float
    rule: str


def assemble_tensor(grid,material,rule='gll',guard=None):
    n = len(grid.points_m)
    if n*3>2000:
        raise MemoryError('assembled tensor reference exceeds 2000 coefficients')
    lengths = np.array([grid.plate.width_m,grid.plate.height_m,grid.plate.thickness_m])/grid.cells
    jacobian = grid.transform@np.diag(lengths/2)
    ek,em,ed = element_matrices(grid.order,jacobian,material,rule)
    k,m,d = np.zeros((3*n,3*n)),np.zeros((n,n)),np.zeros(n)
    for nodes in grid.element_nodes:
        if guard:
            guard.check()
        ids = np.concatenate([nodes+c*n for c in range(3)])
        k[np.ix_(ids,ids)] += ek
        m[np.ix_(nodes,nodes)] += em
        d[nodes] += ed
    mass = np.tile(d,3)
    if np.any(mass<=0) or not np.isfinite(k).all() or np.linalg.norm(k-k.T)>1e-12*np.linalg.norm(k):
        raise ArithmeticError('tensor positive mass or stiffness symmetry failed')
    invsqrt = 1/np.sqrt(mass)
    bound = float(np.max(invsqrt*(abs(k)@invsqrt)))
    if not math.isfinite(bound) or bound<=0:
        raise ArithmeticError('tensor stability bound is invalid')
    return TensorSystem(sparse.csr_matrix(k),sparse.kron(sparse.eye(3),sparse.csr_matrix(m),format='csr'),
        mass,grid,2/math.sqrt(bound),rule)


def local_gll_action(values,order,jacobian,material):
    """Independent gradient/stress/divergence tensor contractions, no Ke."""
    _,weights,derivative,_,_,_ = line_rule(order)
    values = np.asarray(values)
    q = order+1
    if values.shape!=(3,q,q,q) or np.iscomplexobj(values) or not np.isfinite(values).all():
        raise ValueError('tensor element state must contain three complete finite nodal fields')
    jacobian = np.asarray(jacobian)
    if (jacobian.shape!=(3,3) or np.iscomplexobj(jacobian) or not np.isfinite(jacobian).all()
            or np.linalg.det(jacobian)<=0 or np.linalg.cond(jacobian)>1e6):
        raise ValueError('invalid tensor action geometry')
    inverse,det = np.linalg.inv(jacobian),float(np.linalg.det(jacobian))
    def axis_apply(a,matrix,axis):
        return np.moveaxis(np.tensordot(matrix,a,axes=(1,axis)),0,axis)
    reference = np.array([[axis_apply(values[c],derivative,i) for i in range(3)] for c in range(3)])
    gradient = np.einsum('uixyz,ij->ujxyz',reference,inverse)
    lam,mu = material.lame
    stress = mu*(gradient+gradient.swapaxes(0,1))
    trace = sum(gradient[i,i] for i in range(3))
    for i in range(3):
        stress[i,i] += lam*trace
    # Each reference derivative is paired with its weak transpose contraction.
    flux = np.einsum('ujxyz,ij->uixyz',stress,inverse)
    volume_weights = weights[:,None,None]*weights[None,:,None]*weights[None,None,:]*det
    result = np.zeros_like(values,dtype=float)
    for c in range(3):
        for i in range(3):
            result[c] += axis_apply(flux[c,i]*volume_weights,derivative.T,i)
    return result
