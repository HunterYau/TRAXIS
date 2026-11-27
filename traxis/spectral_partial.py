"""CPU tensor reference and partial assembly for unchanged Q2 piezoelectricity."""
from types import SimpleNamespace
import math
import numpy as np
import psutil
from numpy.polynomial import Polynomial
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import splu
from traxis.tensor_hex import line_rule
from traxis.spectral_piezoelectric import disk_mesh, basis, element_matrices


def evaluate(u,a,b,c):
    """Three 1D contractions, never a flattened quadrature-by-element matrix."""
    x=np.einsum('ai,eijk->eajk',a,u)
    x=np.einsum('bj,eajk->eabk',b,x)
    return np.einsum('ck,eabk->eabc',c,x)


def integrate(v,a,b,c):
    x=np.einsum('ck,eabc->eabk',c,v)
    x=np.einsum('bj,eabk->eajk',b,x)
    return np.einsum('ai,eajk->eijk',a,x)


class TensorReference:
    def __init__(self,mesh,material):
        if mesh.order!=2:raise ValueError('This experimental backend is Q2 only')
        self.mesh=mesh;self.n=len(mesh.points_m);self.material=material
        nodes=line_rule(2)[0];x,w=np.polynomial.legendre.leggauss(6)
        polys=[Polynomial.fromroots(np.delete(nodes,i))/np.prod(nodes[i]-np.delete(nodes,i)) for i in range(3)]
        self.N=np.array([p(x) for p in polys]).T
        self.G=np.array([p.deriv()(x) for p in polys]).T
        self.axes=[(self.G,self.N,self.N),(self.N,self.G,self.N),(self.N,self.N,self.G)]
        _,ref,weights=basis(2,6)
        J=ref[None]@mesh.points_m[mesh.hexahedra][:,None]
        determinant=np.linalg.det(J)
        if np.any(determinant<=0) or np.max(np.linalg.cond(J))>1e6:
            raise ValueError('Inverted or ill-conditioned Q2 geometry')
        self.inverse=np.linalg.inv(J).reshape(-1,6,6,6,3,3)
        self.weight=(determinant*weights).reshape(-1,6,6,6)

    def gradient(self,x):
        u=np.asarray(x)[self.mesh.hexahedra].reshape(-1,3,3,3)
        ref=np.stack([evaluate(u,*a) for a in self.axes],axis=-1)
        return np.einsum('e...ir,e...r->e...i',self.inverse,ref)

    def transpose_gradient(self,flux):
        ref=np.einsum('e...ir,e...i->e...r',self.inverse,flux)*self.weight[...,None]
        v=sum(integrate(ref[...,i],*a) for i,a in enumerate(self.axes))
        out=np.zeros(self.n);np.add.at(out,self.mesh.hexahedra.ravel(),v.ravel())
        return out

    def mechanical(self,u):
        u=np.asarray(u).reshape(3,self.n)
        g=np.stack([self.gradient(v) for v in u],axis=-2)
        strain=np.stack([g[...,0,0],g[...,1,1],g[...,2,2],
            g[...,1,2]+g[...,2,1],g[...,0,2]+g[...,2,0],g[...,0,1]+g[...,1,0]],axis=-1)
        c,e,_=self.material.stress_charge
        stress=strain@c.T
        mapping=((0,5,4),(5,1,3),(4,3,2))
        k=np.concatenate([self.transpose_gradient(stress[...,list(row)]) for row in mapping])
        charge=self.transpose_gradient(strain@e.T)
        return k,charge

    def electrical(self,phi):
        grad=self.gradient(phi);_,e,eps=self.material.stress_charge
        stress=grad@e;mapping=((0,5,4),(5,1,3),(4,3,2))
        b=np.concatenate([self.transpose_gradient(stress[...,list(row)]) for row in mapping])
        return b,self.transpose_gradient(grad@eps.T)

    def mass(self,x):
        u=np.asarray(x)[self.mesh.hexahedra].reshape(-1,3,3,3)
        v=integrate(evaluate(u,self.N,self.N,self.N)*self.weight*self.material.density_kg_m3,
            self.N,self.N,self.N)
        out=np.zeros(self.n);np.add.at(out,self.mesh.hexahedra.ravel(),v.ravel());return out


def component(geometry,material,*,rings,sectors,layers,max_workspace_mib=768,max_factor_mib=512):
    """Assemble only scalar D/M, retain the original element bound calculation."""
    if sectors%4:raise ValueError('Polygon sides must be divisible by four')
    mesh=disk_mesh(geometry,side_cells=sectors//4,radial_cells=rings,layers=layers,order=2)
    ne=len(mesh.hexahedra);n=len(mesh.points_m)
    forecast=ne*(216*9+2*27**2*3)*8
    if forecast>min(max_workspace_mib*2**20,.4*psutil.virtual_memory().available):
        raise MemoryError('Partial assembly exceeds component host budget')
    tensor=TensorReference(mesh,material);N,ref,_=basis(2,6)
    _,e,eps=material.stress_charge;rr=[];cc=[];dd=[];mm=[]
    for i,ids in enumerate(mesh.hexahedra):
        inverse=tensor.inverse[i].reshape(216,3,3);w=tensor.weight[i].ravel()
        grad=inverse@ref
        d=np.einsum('qji,jk,qkl,q->il',grad,eps,grad,w,optimize=True)
        m=material.density_kg_m3*N.T@(w[:,None]*N)
        rr.append(np.repeat(ids,27));cc.append(np.tile(ids,27));dd.append(d.ravel());mm.append(m.ravel())
    ij=(np.concatenate(rr),np.concatenate(cc))
    D=sparse.coo_matrix((np.concatenate(dd),ij),shape=(n,n)).tocsc()
    M=sparse.coo_matrix((np.concatenate(mm),ij),shape=(n,n)).tocsr()
    z=mesh.points_m[:,2];lower=np.flatnonzero(abs(z-z.min())<1e-14);upper=np.flatnonzero(abs(z-z.max())<1e-14)
    inside=np.setdiff1d(np.arange(n),np.r_[lower,upper]);di=D[inside][:,inside].tocsc()
    from traxis.quadratic_pzt_sparse import sparse_bytes,factor_bytes
    if 64*sparse_bytes(di)>min(max_factor_mib*2**20,.35*psutil.virtual_memory().available):
        raise MemoryError('Dielectric LU forecast exceeds budget')
    lu=splu(di,permc_spec='MMD_AT_PLUS_A')
    if factor_bytes(lu)>max_factor_mib*2**20:raise MemoryError('Dielectric LU exceeds budget')
    psi=np.zeros(n);psi[upper]=1.;psi[inside]=-lu.solve(D[inside]@psi)
    cap=float(psi@D@psi)
    if not math.isfinite(cap) or cap<=0:raise ArithmeticError('Invalid capacitance')
    g=tensor.electrical(psi)[0]
    c,e,eps=material.stress_charge
    effective=SimpleNamespace(stress_charge=(c+e.T@np.linalg.solve(eps,e),e,eps),density_kg_m3=material.density_kg_m3)
    bound=0.
    for ids in mesh.hexahedra:
        k,m,_,_,_=element_matrices(mesh.points_m[ids],effective,2,6)
        bound=max(bound,float(eigh(k,sparse.block_diag([m]*3).toarray(),eigvals_only=True,subset_by_index=[80,80])[0]))
    return SimpleNamespace(mesh=mesh,n=n,lower=lower,upper=upper,inside=inside,tensor=tensor,
        scalar_mass=M,di=di,electrical_lu=lu,g=g,capacitance_f=cap,
        volume_m3=float(tensor.weight.sum()),body_eigenvalue_bound=bound*(1+1e-8),
        report=dict(element='Q2 spectral hexahedron',nodes=n,hexahedra=ne,rings=rings,sectors=sectors,
            layers=layers,quadrature_order=6,mass='full consistent',material_id=material.id,
            radius_m=geometry.radius_m,thickness_m=geometry.thickness_m,operator='tensor partial assembly'))
