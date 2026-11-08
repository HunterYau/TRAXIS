"""Positive-mass 15-node tetrahedral candidate, separate from standard P2.

The space is P2 plus four cubic face bubbles and one quartic cell bubble.
Nodes and weights follow Table 1 of Geevers et al. (2018), arXiv:1803.10065.
Stiffness and the comparison consistent mass use exact barycentric integrals.
All mechanical coefficient vectors here are component-major.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import math

import numpy as np
from scipy import sparse

from traxis.highorder import highorder_dofs


EDGES = tuple(combinations(range(4),2))
FACES = tuple(combinations(range(4),3))
NODES = np.vstack((np.eye(4),*[np.eye(4)[list(e)].mean(axis=0)[None] for e in EDGES],
    *[np.eye(4)[list(f)].mean(axis=0)[None] for f in FACES],np.full((1,4),.25)))
WEIGHTS = np.array([17/5040]*4+[2/315]*6+[9/560]*4+[16/315])
NODES.setflags(write=False)
WEIGHTS.setflags(write=False)


def _combine(*terms):
    result = {}
    for scale,poly in terms:
        for powers,value in poly.items():
            result[powers] = result.get(powers,0.)+scale*value
    return {p:c for p,c in result.items() if c != 0.}


def _shapes():
    cell = {(1,1,1,1):256.}
    faces = []
    for face in FACES:
        powers = tuple(int(i in face) for i in range(4))
        faces.append(_combine((27.,{powers:1.}),(-27/64,cell)))
    edges = []
    for edge in EDGES:
        powers = tuple(int(i in edge) for i in range(4))
        edges.append(_combine((4.,{powers:1.}),(-.25,cell),
            *((-4/9,faces[j]) for j,f in enumerate(FACES) if set(edge).issubset(f))))
    vertices = []
    for i in range(4):
        one = tuple(int(j==i) for j in range(4))
        two = tuple(2*p for p in one)
        vertices.append(_combine((1.,{two:2.,one:-1.}),(.125,cell),
            *((1/9,faces[j]) for j,f in enumerate(FACES) if i in f)))
    return tuple(vertices+edges+faces+[cell])


_SHAPES = _shapes()


def _derivative(poly,axis):
    result = {}
    for powers,value in poly.items():
        if powers[axis]:
            reduced = list(powers)
            reduced[axis] -= 1
            result[tuple(reduced)] = value*powers[axis]
    return result


_GRADIENTS = tuple(tuple(_combine((1.,_derivative(p,d)),(-1.,_derivative(p,3)))
    for p in _SHAPES) for d in range(3))


def barycentric_integral(powers):
    """Integral on the unit-volume-1/6 reference tetrahedron."""
    if len(powers)!=4 or any(type(p) is not int or not 0<=p<=12 for p in powers):
        raise ValueError('expected four nonnegative bounded integer powers')
    return math.prod(math.factorial(p) for p in powers)/math.factorial(sum(powers)+3)


def _evaluate(polynomials,barycentric):
    return np.stack([sum(c*np.prod(barycentric**np.array(p),axis=-1) for p,c in poly.items())
        for poly in polynomials],axis=-1)


def basis(barycentric, *, gradients=False):
    """Evaluate the nodal basis, or its derivatives in (lambda0,lambda1,lambda2)."""
    barycentric = np.asarray(barycentric)
    if (np.iscomplexobj(barycentric) or barycentric.ndim not in (1,2) or barycentric.shape[-1]!=4
            or not 1<=barycentric.size//4<=65536 or not np.isfinite(barycentric).all()
            or np.any(barycentric < -1e-12) or not np.allclose(barycentric.sum(axis=-1),1.,rtol=0,atol=1e-12)):
        raise ValueError('basis needs bounded real tetrahedral barycentric coordinates')
    if gradients:
        return np.stack([_evaluate(p,barycentric) for p in _GRADIENTS],axis=-1)
    return _evaluate(_SHAPES,barycentric)


@lru_cache(maxsize=1)
def reference_integrals():
    def inner(a,b):
        return math.fsum(c*d*barycentric_integral(tuple(i+j for i,j in zip(p,q)))
            for p,c in a.items() for q,d in b.items())
    mass = np.array([[inner(a,b) for b in _SHAPES] for a in _SHAPES])
    gram = np.array([[[[inner(a,b) for b in right] for a in left]
        for right in _GRADIENTS] for left in _GRADIENTS])
    mass.setflags(write=False)
    gram.setflags(write=False)
    return mass,gram


def element_matrices(vertices,material):
    """Exact vector stiffness/consistent mass and intentional positive nodal mass."""
    vertices = np.asarray(vertices)
    if (np.iscomplexobj(vertices) or vertices.shape!=(4,3) or not np.isfinite(vertices).all()):
        raise ValueError('element needs four finite real physical vertices')
    jacobian = (vertices[:3]-vertices[3]).T
    determinant = abs(float(np.linalg.det(jacobian)))
    scale = float(np.linalg.norm(jacobian,axis=0).max())
    if determinant <= 1e-12*scale**3:
        raise ValueError('degenerate enriched tetrahedron')
    inverse = np.linalg.inv(jacobian)
    reference_mass,reference_gram = reference_integrals()
    gram = determinant*np.einsum('da,eb,deij->abij',inverse,inverse,reference_gram,optimize=True)
    lam,mu = material.lame
    blocks = lam*gram+mu*gram.swapaxes(0,1)
    for component in range(3):
        blocks[component,component] += mu*np.einsum('ddij->ij',gram)
    stiffness = blocks.transpose(0,2,1,3).reshape(45,45)
    mass = material.density_kg_m3*determinant*reference_mass
    lumped = material.density_kg_m3*determinant*WEIGHTS
    if not all(np.isfinite(a).all() for a in (stiffness,mass,lumped)) or np.any(lumped<=0):
        raise ArithmeticError('enriched element produced invalid stiffness or mass')
    return stiffness,mass,lumped


@dataclass
class EnrichedTopology:
    points_m: np.ndarray
    element_nodes: np.ndarray
    geometry: object
    screen: dict


def build_topology(mesh, *, max_vertices=20000,max_tetrahedra=40000,max_dofs=120000,max_workspace_mib=256):
    if (type(max_dofs) is not int or not 45<=max_dofs<=5000000
            or not math.isfinite(max_workspace_mib) or not 0<max_workspace_mib<=2048):
        raise ValueError('invalid enriched coefficient/workspace limit')
    screen = highorder_dofs(mesh,2,max_vertices=max_vertices,max_tetrahedra=max_tetrahedra,
        max_topology_mib=max_workspace_mib)
    n = sum(screen[k] for k in ('vertices','edges','faces','tetrahedra'))
    workspace = screen['topology_screen_mib']+(n*24+screen['tetrahedra']*15*8)/1024**2
    if 3*n>max_dofs or workspace>max_workspace_mib:
        raise MemoryError('enriched topology exceeds its declared screen')
    tets = np.asarray(mesh.tetrahedra)
    edges,edge_ids = np.unique(np.sort(tets[:,EDGES],axis=-1).reshape(-1,2),axis=0,return_inverse=True)
    faces,face_ids = np.unique(np.sort(tets[:,FACES],axis=-1).reshape(-1,3),axis=0,return_inverse=True)
    nv,ne,nf = len(mesh.points_m),len(edges),len(faces)
    element_nodes = np.column_stack((tets,nv+edge_ids.reshape(-1,6),nv+ne+face_ids.reshape(-1,4),
        nv+ne+nf+np.arange(len(tets))))
    points = np.vstack((mesh.points_m,mesh.points_m[edges].mean(axis=1),mesh.points_m[faces].mean(axis=1),
        mesh.points_m[tets].mean(axis=1)))
    if points.shape!=(n,3) or len(np.unique(element_nodes))!=n:
        raise ArithmeticError('enriched node count or coverage changed')
    screen.update(scalar_dofs=n,vector_dofs=3*n,local_scalar_dofs=15,
        assembly_screen_mib=len(tets)*45**2*32/1024**2,polynomial_degree=2,enrichment_maximum_degree=4,
        construction_workspace_screen_mib=workspace,coefficient_layout='component_major_enriched_nodal')
    return EnrichedTopology(points,element_nodes,mesh,screen)


@dataclass
class EnrichedSystem:
    stiffness: object
    consistent_mass: object
    mass_kg: np.ndarray
    topology: EnrichedTopology
    stable_dt_s: float

    @property
    def ndof(self):
        return len(self.mass_kg)


def assemble_enriched(mesh,material,guard=None, *, max_dofs=2000):
    """Bounded assembled reference; large GPU construction is a separate task."""
    if type(max_dofs) is not int or not 45<=max_dofs<=2000:
        raise ValueError('small enriched assembly is restricted to 2000 coefficients')
    topology = build_topology(mesh,max_dofs=max_dofs)
    n = len(topology.points_m)
    if (3*n)**2*32>128*1024**2:
        raise MemoryError('small enriched assembly exceeds 128 MiB workspace screen')
    k,m,d = np.zeros((3*n,3*n)),np.zeros((n,n)),np.zeros(n)
    for tet,ids in zip(mesh.tetrahedra,topology.element_nodes):
        if guard:
            guard.check()
        ek,em,ed = element_matrices(mesh.points_m[tet],material)
        vector_ids = np.concatenate([ids+c*n for c in range(3)])
        k[np.ix_(vector_ids,vector_ids)] += ek
        m[np.ix_(ids,ids)] += em
        d[ids] += ed
    mass = np.tile(d,3)
    if np.any(mass<=0) or not np.isfinite(k).all() or np.linalg.norm(k-k.T)>1e-11*np.linalg.norm(k):
        raise ArithmeticError('enriched positive mass or stiffness symmetry failed')
    invsqrt = 1/np.sqrt(mass)
    bound = float(np.max(invsqrt*(np.abs(k)@invsqrt)))
    if not math.isfinite(bound) or bound<=0:
        raise ArithmeticError('invalid enriched explicit stability bound')
    return EnrichedSystem(sparse.csr_matrix(k),sparse.kron(sparse.eye(3),sparse.csr_matrix(m),format='csr'),
        mass,topology,2/math.sqrt(bound))
