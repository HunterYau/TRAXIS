"""Experimental straight-sided 10-node PZT tetrahedra; opt-in, no LTS changes.

P2 displacement AND potential, full consistent mass, positive quadrature.
Geometry stays affine: edge midpoints are not projected onto a curved circle.
"""
from types import SimpleNamespace
import math
from functools import lru_cache
import numpy as np
import psutil
from scipy import sparse
from scipy.sparse.linalg import splu
from traxis.piezoelectric import PiezoSystem, electrode_port
from traxis.pzt_body import disk_mesh, BodyAttachment
from traxis.pzt_receiver import _point_weights

EDGES=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))

def promote(mesh):
    """One shared midpoint per global edge; preserve the polygon domain."""
    if np.iscomplexobj(mesh.points_m):raise ValueError('Real mesh coordinates required')
    p=np.asarray(mesh.points_m,dtype=float);t=np.asarray(mesh.tetrahedra)
    if p.ndim!=2 or p.shape[1]!=3 or t.ndim!=2 or t.shape[1]!=4:
        raise ValueError('Promotion requires linear tetrahedra')
    pairs=np.stack([t[:,e] for e in EDGES],axis=1)
    edges,inv=np.unique(np.sort(pairs.reshape(-1,2),axis=1),axis=0,return_inverse=True)
    points=np.vstack([p,p[edges].mean(axis=1)])
    return SimpleNamespace(points_m=points,tetrahedra=np.c_[t,len(p)+inv.reshape(-1,6)])

def quadratic_disk(geometry,*,rings=3,sectors=32,layers=2,max_nodes=8192):
    linear,_=disk_mesh(geometry,rings=rings,sectors=sectors,layers=layers,max_nodes=8192,max_elements=65536)
    result=promote(linear)
    if len(result.points_m)>max_nodes:raise MemoryError('Quadratic node budget exceeded')
    return result

def shape(bary):
    l=np.asarray(bary,dtype=float)
    if l.ndim!=2 or l.shape[1]!=4:raise ValueError('Four barycentric coordinates required')
    return np.c_[l*(2*l-1),np.stack([4*l[:,i]*l[:,j] for i,j in EDGES],axis=1)]

def gradients(bary,linear_gradients):
    l=np.asarray(bary)
    return np.concatenate([(4*l[:,None,:]-1)*linear_gradients[None,:,:],
        np.stack([4*(l[:,j,None]*linear_gradients[:,i]+l[:,i,None]*linear_gradients[:,j]) for i,j in EDGES],axis=2)],axis=2)

@lru_cache(maxsize=8)
def quadrature(order=4):
    """Positive Duffy product rule. order=4 exactly integrates P2 mass (degree4).

    Duffy Jacobian raises the maximum polynomial degree in u to six;
    four-point Gauss-Legendre is exact through degree seven.
    """
    x,w=np.polynomial.legendre.leggauss(order);x=(x+1)/2;w=w/2
    u,v,z=np.meshgrid(x,x,x,indexing='ij');wu,wv,wz=np.meshgrid(w,w,w,indexing='ij')
    a=u.ravel();b=((1-u)*v).ravel();c=((1-u)*(1-v)*z).ravel()
    return np.c_[1-a-b-c,a,b,c],(6*wu*wv*wz*(1-u)**2*(1-v)).ravel()

def strain_matrix(g):
    b=np.zeros((len(g),6,30))
    b[:,0,:10]=g[:,0];b[:,1,10:20]=g[:,1];b[:,2,20:]=g[:,2]
    b[:,3,10:20]=g[:,2];b[:,3,20:]=g[:,1]
    b[:,4,:10]=g[:,2];b[:,4,20:]=g[:,0]
    b[:,5,:10]=g[:,1];b[:,5,10:20]=g[:,0]
    return b

def element_matrices(points,material,*,stiffness_order=2,mass_order=4):
    """Affine-element matrices; degree2 stiffness requires order>=3 Duffy.

    Default stiffness uses the symmetric degree2 four-point tetra rule.
    Alternate higher Duffy rule is an independent quadrature cross-check.
    """
    p=np.asarray(points)
    if p.shape!=(10,3) or np.iscomplexobj(p) or not np.isfinite(p).all():
        raise ValueError('Ten finite real node coordinates required')
    jac=(p[1:4]-p[0]).T
    if np.linalg.cond(jac)>1e8:
        raise ValueError('Invalid affine quadratic tetrahedron')
    if not np.allclose(p[4:],[.5*(p[i]+p[j]) for i,j in EDGES],rtol=0,atol=1e-14):
        raise ValueError('Curved geometry requires a different integration implementation')
    vol=abs(np.linalg.det(jac))/6
    if vol<=0:raise ValueError('Zero tetrahedron volume')
    g=np.empty((3,4));g[:,1:]=np.linalg.inv(jac).T;g[:,0]=-g[:,1:].sum(axis=1)
    if stiffness_order==2:
        a=(5+3*np.sqrt(5))/20;b=(5-np.sqrt(5))/20
        bary=np.full((4,4),b);np.fill_diagonal(bary,a);weights=np.full(4,.25)
    elif stiffness_order>=3:bary,weights=quadrature(stiffness_order)
    else:raise ValueError('Underintegration of P2 stiffness is prohibited')
    grad=gradients(bary,g);strain=strain_matrix(grad);c,e,eps=material.stress_charge
    K=vol*np.einsum('qji,jk,qkl,q->il',strain,c,strain,weights,optimize=True)
    B=vol*np.einsum('qji,jk,qkl,q->il',strain,e.T,grad,weights,optimize=True)
    D=vol*np.einsum('qji,jk,qkl,q->il',grad,eps,grad,weights,optimize=True)
    if mass_order<4:raise ValueError('Consistent P2 mass requires degree-four-exact quadrature')
    lm,wm=quadrature(mass_order);N=shape(lm)
    M=material.density_kg_m3*vol*np.kron(np.eye(3),N.T@(wm[:,None]*N))
    return K,M,B,D,vol

def assemble(mesh,material,*,max_nodes=4000,max_workspace_mib=768):
    """Sparse, bounded component assembly. No global plate matrices."""
    p=np.asarray(mesh.points_m);t=np.asarray(mesh.tetrahedra);n=len(p)
    if t.ndim!=2 or t.shape[1]!=10 or t.dtype.kind not in 'iu' or t.min()<0 or t.max()>=n:
        raise ValueError('Ten-node tetrahedral connectivity required')
    if n>max_nodes:raise MemoryError('Quadratic component exceeds declared node budget')
    # Includes triplets and sparse conversions; no quadratic dense allocation.
    forecast=len(t)*(900*2+300+100)*24*2
    if forecast>max_workspace_mib*2**20 or forecast>.5*psutil.virtual_memory().available:
        raise MemoryError('Sparse assembly workspace forecast exceeds budget')
    data=[[],[],[],[]];rows=[[],[],[],[]];cols=[[],[],[],[]];volume=0.
    for tet in t:
        k,m,b,d,v=element_matrices(p[tet],material);volume+=v
        ids=np.concatenate([tet+i*n for i in range(3)])
        for z,(a,rr,cc) in enumerate([(k,ids,ids),(m,ids,ids),(b,ids,tet),(d,tet,tet)]):
            data[z].append(a.ravel());rows[z].append(np.repeat(rr,len(cc)));cols[z].append(np.tile(cc,len(rr)))
    result=[]
    for z,sz in enumerate([(3*n,3*n),(3*n,3*n),(3*n,n),(n,n)]):
        result.append(sparse.coo_matrix((np.concatenate(data[z]),(np.concatenate(rows[z]),np.concatenate(cols[z]))),shape=sz).tocsc())
        data[z].clear();rows[z].clear();cols[z].clear()
    return SimpleNamespace(stiffness=result[0],mass=result[1],coupling=result[2],dielectric=result[3],volume_m3=float(volume))

def electrical_reduction(system,points,*,electrical='open'):
    """Ground lower face; upper face equipotential, open or prescribed zero."""
    if electrical not in ('open','voltage'):raise ValueError('Unknown electrical state')
    z=points[:,2];lower=np.flatnonzero(abs(z-z.min())<1e-14);upper=np.flatnonzero(abs(z-z.max())<1e-14)
    inside=np.setdiff1d(np.arange(len(points)),np.r_[lower,upper]);rr=list(inside);cc=list(range(len(inside)))
    if electrical=='open':rr.extend(upper);cc.extend([len(inside)]*len(upper))
    ne=len(inside)+(electrical=='open')
    T=sparse.coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(points),ne)).tocsc()
    return (system.coupling@T).tocsc(),(T.T@system.dielectric@T).tocsc(),lower,upper

def attach_quadratic_disk(layout,material,geometry,*,first_dof=None,electrical='open',
                          rings=3,sectors=32,layers=2,input_capacitance_f=0.,max_workspace_mib=512):
    """Opt-in perfect tie producing the ORIGINAL BodyAttachment interface.

    Dense local condensation is explicitly budgeted. Larger studies should
    retain sparse component operations; this adapter is for small checks.
    """
    if electrical not in ('open','voltage') or not math.isfinite(input_capacitance_f) or input_capacitance_f<0:
        raise ValueError('Invalid electrical port')
    if not np.array_equal(layout.transform,np.eye(3)) or geometry.in_plane_angle_rad!=0:
        raise ValueError('Aligned axes required')
    center=np.asarray(geometry.center_xy_m);r=geometry.radius_m
    if np.any(center-r<0) or np.any(center+r>[layout.plate.width_m,layout.plate.height_m]):raise ValueError('Disk outside plate')
    mesh=quadratic_disk(geometry,rings=rings,sectors=sectors,layers=layers);n=len(mesh.points_m)
    host=layout.screen['vector_dofs'];first_dof=host if first_dof is None else first_dof
    if first_dof<host:raise ValueError('Body DOFs must follow host')
    z=layout.plate.thickness_m if geometry.surface=='top' else 0.
    contact=np.flatnonzero(abs(mesh.points_m[:,2]-(0. if geometry.surface=='top' else geometry.thickness_m))<1e-14)
    xyz=mesh.points_m.copy();xyz[:,:2]+=center;xyz[:,2]+=z if geometry.surface=='top' else -geometry.thickness_m
    tied=np.concatenate([contact+i*n for i in range(3)]);free=np.setdiff1d(np.arange(3*n),tied)
    rows=[];cols=[];values=[]
    for node in contact:
        ids,w=_point_weights(layout,(*xyz[node,:2],z));nz=layout.cells[2]*layout.order+1
        keep=ids%nz==(nz-1 if geometry.surface=='top' else 0);ids,w=ids[keep],w[keep]
        if abs(w.sum()-1)>1e-11:raise ArithmeticError('Interface partition of unity failed')
        for i in range(3):rows.extend([node+i*n]*len(ids));cols.extend(ids+i*host//3);values.extend(w)
    rows.extend(free);cols.extend(first_dof+np.arange(len(free)));values.extend(np.ones(len(free)))
    indices=np.unique(cols);count=len(indices)
    forecast=(12*(3*n)**2+10*count**2)*8
    if forecast>max_workspace_mib*2**20 or forecast>.5*psutil.virtual_memory().available:raise MemoryError('Dense attachment exceeds budget')
    T=sparse.coo_matrix((values,(rows,np.searchsorted(indices,cols))),shape=(3*n,count)).tocsr()
    a=assemble(mesh,material,max_workspace_mib=max_workspace_mib)
    lower=np.flatnonzero(abs(mesh.points_m[:,2])<1e-14);upper=np.flatnonzero(abs(mesh.points_m[:,2]-geometry.thickness_m)<1e-14)
    port=electrode_port(PiezoSystem(a.stiffness.toarray(),a.mass.toarray(),a.coupling.toarray(),a.dielectric.toarray(),a.volume_m3),lower,upper)
    def project(a):
        x=T.T@(T.T@a.T).T;return (x+x.T)/2
    K=project(port.stiffness_sc);M=project(port.mass);g=np.asarray(T.T@port.coupling_n_v)*geometry.poling_sign
    cap=port.capacitance_f+input_capacitance_f
    if electrical=='open':K+=np.outer(g,g)/cap
    report=dict(element='P2 affine tetrahedron',nodes=n,tetrahedra=len(mesh.tetrahedra),rings=rings,sectors=sectors,layers=layers,
                center_xy_m=center.tolist(),surface=geometry.surface,radius_m=geometry.radius_m,
                thickness_m=geometry.thickness_m,poling_sign=geometry.poling_sign,
                material_id=material.id,material_provenance=material.provenance,input_capacitance_f=input_capacitance_f,
                body_mass_kg=material.density_kg_m3*a.volume_m3,capacitance_f=port.capacitance_f,electrical=electrical,
                bond='perfect tie',geometry='straight polygon; no curved midside nodes',workspace_screen_mib=forecast/2**20)
    return BodyAttachment(indices,K,M,g,port.capacitance_f,cap,first_dof+len(free),T,xyz,free,report)

