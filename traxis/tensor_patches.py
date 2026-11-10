"""Finite-area circular mechanical covectors on an axis-aligned tensor top face."""
from dataclasses import dataclass
import math
import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import quad_vec
from traxis.tensor_hex import line_rule


def disk_average(a,b):
    if a%2 or b%2:return 0.
    i,j=a//2,b//2
    return math.gamma(i+.5)*math.gamma(j+.5)/(math.pi*(i+j+1)*math.gamma(i+j+1))


def _basis(nodes):
    return [Polynomial.fromroots(np.delete(nodes,i))/np.prod(nodes[i]-np.delete(nodes,i)) for i in range(len(nodes))]


def _rectangle_integrals(xnodes,ynodes,box,guard):
    left,right,bottom,top=box
    lo,hi=max(-1.,left),min(1.,right)
    size=(len(xnodes),len(ynodes))
    if hi<=lo:return np.zeros(size),0.,0
    xpoly=_basis(xnodes);anti=[p.integ() for p in _basis(ynodes)]
    cuts=[]
    for boundary in (bottom,top):
        if abs(boundary)<1:
            root=math.sqrt(1-boundary*boundary)
            cuts.extend(x for x in (-root,root) if lo<x<hi)
    def integrand(x):
        if guard:guard.check()
        edge=math.sqrt(max(0.,1-x*x));a,b=max(bottom,-edge),min(top,edge)
        if b<=a:return np.zeros(size)
        return np.outer([p(x) for p in xpoly],[p(b)-p(a) for p in anti])
    result,error,info=quad_vec(integrand,lo,hi,epsabs=1e-12,epsrel=1e-12,workers=1,points=sorted(set(cuts)),
        cache_size=1024**2,limit=1024,full_output=True)
    if not info.success or not np.isfinite(result).all():
        raise ArithmeticError('circular tensor surface quadrature did not converge')
    return result,float(error),int(info.neval)


@dataclass
class TensorPatch:
    indices: np.ndarray
    weights: np.ndarray
    direction: np.ndarray
    report: dict

    def vector(self,scalar_dofs):
        if type(scalar_dofs) is not int or scalar_dofs<=int(self.indices.max()):
            raise ValueError('patch vector has too few scalar coefficients')
        vector=np.zeros((3,scalar_dofs));vector[:,self.indices]=self.direction[:,None]*self.weights[None,:]
        return vector.ravel()


def circular_patch(layout,center_xy_m,radius_m,direction=(0.,0.,1.),guard=None,*,max_faces=256):
    center,direction=np.asarray(center_xy_m),np.asarray(direction)
    if (center.shape!=(2,) or np.iscomplexobj(center) or not np.isfinite(center).all()
            or direction.shape!=(3,) or np.iscomplexobj(direction) or not np.isfinite(direction).all() or np.linalg.norm(direction)==0
            or not math.isfinite(radius_m) or radius_m<=0 or type(max_faces) is not int or not 1<=max_faces<=1024):
        raise ValueError('circular tensor patch needs finite coordinates, positive radius and nonzero direction')
    if not np.array_equal(layout.transform,np.eye(3)):
        raise ValueError('physical circular top patches currently require an axis-aligned plate')
    center=center.astype(float);direction=direction.astype(float)/np.linalg.norm(direction)
    width,height=layout.plate.width_m,layout.plate.height_m
    if np.any(center-radius_m<0) or np.any(center+radius_m>[width,height]):
        raise ValueError('the complete mechanical disk must be inside the physical plate')
    nx,ny,nz=layout.cells;p=layout.order;nodes=line_rule(p)[0]
    hx,hy=width/nx,height/ny
    xfirst=max(0,int(math.floor((center[0]-radius_m)/hx)));xlast=min(nx-1,int(math.floor((center[0]+radius_m)/hx)))
    yfirst=max(0,int(math.floor((center[1]-radius_m)/hy)));ylast=min(ny-1,int(math.floor((center[1]+radius_m)/hy)))
    if (xlast-xfirst+1)*(ylast-yfirst+1)>max_faces:
        raise MemoryError('circular tensor patch exceeds its surface-cell bound')
    indices=[];integrals=[];error=0.;evaluations=0;faces=0
    for i in range(xfirst,xlast+1):
        for j in range(yfirst,ylast+1):
            if guard:guard.check()
            xleft,ybottom=i*hx,j*hy
            xnodes=(xleft+(nodes+1)*hx/2-center[0])/radius_m
            ynodes=(ybottom+(nodes+1)*hy/2-center[1])/radius_m
            box=((xleft-center[0])/radius_m,(xleft+hx-center[0])/radius_m,
                 (ybottom-center[1])/radius_m,(ybottom+hy-center[1])/radius_m)
            local,estimate,count=_rectangle_integrals(xnodes,ynodes,box,guard)
            if not np.any(local):continue
            ix=i*p+np.arange(p+1);iy=j*p+np.arange(p+1)
            ids=(ix[:,None]*(ny*p+1)+iy[None,:])*(nz*p+1)+nz*p
            indices.append(ids.ravel());integrals.append(local.ravel()*radius_m**2)
            error+=estimate*radius_m**2;evaluations+=count;faces+=1
    if not indices:raise ArithmeticError('circular tensor patch has no surface support')
    flat=np.concatenate(indices);raw=np.concatenate(integrals);unique,inverse=np.unique(flat,return_inverse=True)
    weights=np.zeros(len(unique));np.add.at(weights,inverse,raw)
    area=math.pi*radius_m**2;measured=float(weights.sum());area_error=abs(measured/area-1)
    if area_error>1e-10:raise ArithmeticError('circular tensor patch fails analytic area agreement')
    # Normalize by the analytic physical disk area, never by a computed-area fit.
    weights/=area
    nz_nodes=nz*p+1;ny_nodes=ny*p+1
    ix=unique//(ny_nodes*nz_nodes);iy=(unique//nz_nodes)%ny_nodes
    dx,dy=layout.axes_m[0][ix]-center[0],layout.axes_m[1][iy]-center[1]
    sx,sy=max(radius_m,float(np.max(abs(dx)))),max(radius_m,float(np.max(abs(dy))))
    bounded_error=0.;quadratic_error=0.
    for a in range(p+1):
        for b in range(p+1):
            expected=disk_average(a,b)*(radius_m/sx)**a*(radius_m/sy)**b
            actual=float(weights@((dx/sx)**a*(dy/sy)**b));bounded_error=max(bounded_error,abs(actual-expected))
    for a in range(min(2,p)+1):
        for b in range(min(2,p)+1-a):
            actual=float(weights@((dx/radius_m)**a*(dy/radius_m)**b));quadratic_error=max(quadratic_error,abs(actual-disk_average(a,b)))
    if max(bounded_error,quadratic_error)>1e-10:
        raise ArithmeticError('circular tensor patch fails represented polynomial moments')
    report=dict(center_xy_m=center.tolist(),radius_m=radius_m,direction=direction.tolist(),analytic_area_m2=area,
        integrated_area_m2=measured,relative_area_error=area_error,estimated_integral_error_m2=error,
        bounded_coordinate_moment_maximum_absolute_error=bounded_error,radius_scaled_quadratic_maximum_absolute_error=quadratic_error,
        quadrature_evaluations=evaluations,surface_cells=faces,scalar_support=len(unique),negative_weights=int(np.count_nonzero(weights<0)),
        normalization='analytic disk area',scope='Unit-total-force mechanical covector and work-conjugate area-average displacement probe; not PZT voltage.')
    return TensorPatch(unique,weights,direction,report)
