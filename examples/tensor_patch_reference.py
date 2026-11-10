"""Independent native-basis circle integration with Gaussian inner quadrature."""
import math
import numpy as np
from scipy.integrate import quad_vec
from traxis.tensor_hex import line_rule


def native_circular_covector(grid,center,radius,guard=None):
    import ngsolve as ng
    import netgen.meshing as nm
    ng.SetNumThreads(1)
    raw=nm.Mesh(dim=3);vertices=[raw.Add(nm.MeshPoint(nm.Pnt(*p))) for p in grid.corner_points_m]
    region=raw.AddRegion('solid',dim=3)
    for element in grid.corner_elements:raw.Add(nm.Element3D(region,[vertices[i] for i in element]))
    mesh=ng.Mesh(raw);scalar=ng.H1(mesh,order=grid.order);n=len(grid.points_m)
    if scalar.ndof!=n:raise ArithmeticError('native circle space has unexpected degree count')
    nodes=line_rule(grid.order)[0]
    sample=np.stack(np.meshgrid(*[(nodes+1)/2]*3,indexing='ij'),axis=-1).reshape(-1,3)
    sample_rule=ng.IntegrationRule(points=sample.tolist(),weights=np.ones(len(sample)).tolist())
    transform=np.zeros((n,n));seen=np.zeros(n,dtype=bool);overlap=0.;native=np.zeros(n)
    gx,gw=np.polynomial.legendre.leggauss(grid.order+1)
    total_evaluations=0;cx,cy=center
    for element in mesh.Elements(ng.VOL):
        if guard:guard.check()
        eid=ng.ElementId(ng.VOL,element.nr);fe=scalar.GetFE(eid);trafo=mesh.GetTrafo(eid)
        dofs=np.asarray(scalar.GetDofNrs(eid),int)
        for point,ip in zip(sample,sample_rule):
            physical=np.array(trafo(ip).point);distances=np.linalg.norm(grid.points_m-physical,axis=1);node=int(np.argmin(distances))
            if distances[node]>1e-12:raise ArithmeticError('native footprint point is absent from nodal grid')
            row=np.zeros(n);row[dofs]=fe.CalcShape(*point)
            if seen[node]:overlap=max(overlap,float(np.max(abs(transform[node]-row))))
            else:transform[node]=row;seen[node]=True
        reference_rule=ng.IntegrationRule(points=[(0,0,0),(.5,.5,1.)],weights=[1,1])
        mapped=[trafo(ip) for ip in reference_rule]
        if abs(np.array(mapped[1].point)[2]-grid.plate.thickness_m)>1e-12:continue
        origin=np.array(mapped[0].point);jacobian=np.array(mapped[0].jacobi);inverse=np.linalg.inv(jacobian)
        corners=grid.corner_points_m[[v.nr for v in element.vertices]]
        xmin,ymin=corners[:,:2].min(axis=0);xmax,ymax=corners[:,:2].max(axis=0)
        lo,hi=max(xmin,cx-radius),min(xmax,cx+radius)
        if hi<=lo:continue
        cuts=[]
        for y in (ymin,ymax):
            if abs(y-cy)<radius:
                root=math.sqrt(radius*radius-(y-cy)**2)
                cuts.extend(x for x in (cx-root,cx+root) if lo<x<hi)
        def outer(x):
            if guard:guard.check()
            edge=math.sqrt(max(0.,radius*radius-(x-cx)**2));a,b=max(ymin,cy-edge),min(ymax,cy+edge)
            if b<=a:return np.zeros(len(dofs))
            result=np.zeros(len(dofs))
            for point,weight in zip((a+b)/2+gx*(b-a)/2,gw):
                ref=inverse@(np.array([x,point,grid.plate.thickness_m])-origin)
                result+=weight*np.array(fe.CalcShape(*ref))
            return result*(b-a)/2
        integrated,error,info=quad_vec(outer,lo,hi,epsabs=1e-12*math.pi*radius**2,epsrel=1e-12,
            points=sorted(set(cuts)),workers=1,cache_size=1024**2,limit=1024,full_output=True)
        if not info.success:raise ArithmeticError('native circular trace integration did not converge')
        native[dofs]+=integrated/(math.pi*radius**2);total_evaluations+=info.neval
    if not seen.all() or overlap>1e-10:raise ArithmeticError('native footprint evaluation transform is incomplete')
    return np.linalg.solve(transform.T,native),dict(evaluations=int(total_evaluations),shared_transform_error=overlap,
        transform=transform,native_coefficients=native)
