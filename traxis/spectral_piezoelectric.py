"""Experimental conforming all-hex PZT disk; GLL nodal Qp fields, FP64.

Four annular blocks surround a square core: no collapsed polar elements.
Geometry is bilinear in-plane and affine in z, preserving a regular polygon.
Gauss integration retains full CONSISTENT mass; diagonal GLL mass is not used.
This component backend is not connected to any plate time integrator.
"""
from functools import lru_cache
from types import SimpleNamespace
import math
import numpy as np
from numpy.polynomial import Polynomial
from scipy import sparse
import psutil
from traxis.tensor_hex import line_rule


def disk_mesh(geometry,*,side_cells=16,radial_cells=1,layers=1,order=2,max_nodes=12500):
    if order not in (2,3,4) or not 1<=side_cells<=24 or not 1<=radial_cells<=8 or not 1<=layers<=4:
        raise ValueError('Invalid experimental spectral mesh')
    predicted=((side_cells*order+1)**2+4*side_cells*radial_cells*order**2)*(layers*order+1)
    if predicted>max_nodes:raise MemoryError('Spectral node forecast exceeds component budget')
    R=geometry.radius_m;T=geometry.thickness_m;a=.45*R
    quads=[];axis=np.linspace(-a,a,side_cells+1)
    for i in range(side_cells):
        for j in range(side_cells):
            quads.append(np.array([[axis[i],axis[j]],[axis[i],axis[j+1]],
                                   [axis[i+1],axis[j]],[axis[i+1],axis[j+1]]]).reshape(2,2,2))
    for side in range(4):
        angle=side*np.pi/2;rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        def boundary(t,r):
            inner=np.array([a,a*t]);outer=R*np.array([np.cos(np.pi*t/4),np.sin(np.pi*t/4)])
            return ((1-r)*inner+r*outer)@rotation.T
        for i in range(radial_cells):
            for j in range(side_cells):
                quads.append(np.array([[boundary(-1+2*v/side_cells,u/radial_cells)
                    for v in (j,j+1)] for u in (i,i+1)]))
    g=(line_rule(order)[0]+1)/2
    uv=np.array(np.meshgrid(g,g,indexing='ij')).reshape(2,-1).T
    points=[];indices={};elements=[]
    for corners in quads:
        xy=np.einsum('qi,qj,ijc->qc',np.c_[1-uv[:,0],uv[:,0]],np.c_[1-uv[:,1],uv[:,1]],corners)
        for k in range(layers):
            z=(k+g)*T/layers
            xyz=np.concatenate([np.broadcast_to(xy[:,None,:],(len(xy),len(z),2)),
                                np.broadcast_to(z[None,:,None],(len(xy),len(z),1))],axis=2).reshape(-1,3)
            local=[]
            for x in xyz:
                key=tuple(np.round(x/R,12))
                if key not in indices:indices[key]=len(points);points.append(x)
                else:
                    if np.linalg.norm(points[indices[key]]-x)>R*1e-11:raise ArithmeticError('Coordinate merge mismatch')
                local.append(indices[key])
            elements.append(local)
    p=np.array(points);h=np.array(elements,dtype=int)
    if len(p)!=predicted:raise ArithmeticError('All-hex topology count mismatch')
    return SimpleNamespace(points_m=p,hexahedra=h,order=order,side_cells=side_cells,
        radial_cells=radial_cells,layers=layers,boundary_sides=4*side_cells,
        polygon_area_m2=2*side_cells*R*R*np.sin(np.pi/(2*side_cells)))


@lru_cache(maxsize=20)
def basis(order,quadrature_order):
    nodes=line_rule(order)[0];x,w=np.polynomial.legendre.leggauss(quadrature_order)
    polys=[Polynomial.fromroots(np.delete(nodes,i))/np.prod(nodes[i]-np.delete(nodes,i)) for i in range(order+1)]
    N=np.array([p(x) for p in polys]).T;G=np.array([p.deriv()(x) for p in polys]).T
    def product(a,b,c):return np.einsum('ai,bj,ck->abcijk',a,b,c).reshape(len(x)**3,-1)
    shape=product(N,N,N);grad=np.stack([product(G,N,N),product(N,G,N),product(N,N,G)],axis=1)
    weights=np.einsum('a,b,c->abc',w,w,w).ravel()
    return shape,grad,weights


def element_matrices(points,material,order=2,quadrature_order=None):
    q=order+2 if quadrature_order is None else quadrature_order
    if q<order+1:raise ValueError('Consistent spectral mass requires sufficient Gauss integration')
    points=np.asarray(points);nl=(order+1)**3
    if points.shape!=(nl,3) or not np.isfinite(points).all():raise ValueError('Invalid hex nodes')
    N,ref,w=basis(order,q);J=ref@points;det=np.linalg.det(J)
    if np.any(det<=0) or np.max(np.linalg.cond(J))>1e6:raise ValueError('Inverted or ill-conditioned hex')
    grad=np.linalg.inv(J)@ref;weight=w*det
    strain=np.zeros((len(w),6,3*nl))
    for i in range(3):strain[:,i,i*nl:(i+1)*nl]=grad[:,i]
    for row,i,j in [(3,1,2),(4,0,2),(5,0,1)]:
        strain[:,row,i*nl:(i+1)*nl]=grad[:,j];strain[:,row,j*nl:(j+1)*nl]=grad[:,i]
    c,e,eps=material.stress_charge
    K=np.einsum('qji,jk,qkl,q->il',strain,c,strain,weight,optimize=True)
    B=np.einsum('qji,jk,qkl,q->il',strain,e.T,grad,weight,optimize=True)
    D=np.einsum('qji,jk,qkl,q->il',grad,eps,grad,weight,optimize=True)
    M=material.density_kg_m3*N.T@(weight[:,None]*N)
    return K,M,B,D,float(weight.sum())


def assemble(mesh,material,*,quadrature_order=None,max_workspace_mib=1024,backend='cpu'):
    n=len(mesh.points_m);nl=(mesh.order+1)**3;ne=len(mesh.hexahedra)
    forecast=ne*14*nl**2*24*2
    if forecast>min(max_workspace_mib*2**20,.45*psutil.virtual_memory().available):
        raise MemoryError('Spectral component assembly forecast exceeds available RAM')
    rows=[[],[],[],[]];cols=[[],[],[],[]];data=[[],[],[],[]];volume=0
    if backend=='gpu':
        from .gpu_spectral_piezoelectric import element_batches
        elements=element_batches(mesh,material,quadrature_order)
    elif backend=='cpu':
        elements=(element_matrices(mesh.points_m[t],material,mesh.order,quadrature_order) for t in mesh.hexahedra)
    else:raise ValueError('Unsupported spectral backend '+str(backend))
    for tet,(K,M,B,D,v) in zip(mesh.hexahedra,elements):
        volume+=float(v)
        ids=np.concatenate([tet+i*n for i in range(3)])
        for k,(matrix,rr,cc) in enumerate([(K,ids,ids),(M,tet,tet),(B,ids,tet),(D,tet,tet)]):
            rows[k].append(np.repeat(rr,len(cc)));cols[k].append(np.tile(cc,len(rr)));data[k].append(matrix.ravel())
    result=[]
    for k,size in enumerate([(3*n,3*n),(n,n),(3*n,n),(n,n)]):
        result.append(sparse.coo_matrix((np.concatenate(data[k]),(np.concatenate(rows[k]),np.concatenate(cols[k]))),shape=size).tocsc())
        rows[k].clear();cols[k].clear();data[k].clear()
    return SimpleNamespace(stiffness=result[0],mass=sparse.kron(sparse.eye(3),result[1],format='csc'),
        coupling=result[2],dielectric=result[3],volume_m3=volume)
