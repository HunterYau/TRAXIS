"""Finite elastic thin-interface bond; zero geometric thickness approximation."""
import math
import numpy as np
from scipy.sparse import diags
from scipy.spatial import Delaunay


def interface_operators(body_trace, host_trace, points_xy, properties):
    """Positive spring energy and endpoint-lumped adhesive mass.

    Interface kinematics are coincident. Thickness enters stiffness and mass,
    not an explicit geometric layer; no peel failure/viscoelasticity is inferred.
    """
    e=properties['young_pa'];nu=properties['poisson'];h=properties['thickness_m'];rho=properties['density_kg_m3']
    if (not all(math.isfinite(v) and v>0 for v in (e,h,rho))
            or not math.isfinite(nu) or not -1<nu<.5):
        raise ValueError('invalid adhesive properties')
    xy=np.asarray(points_xy);areas=np.zeros(len(xy))
    for triangle in Delaunay(xy).simplices:
        a,b,c=xy[triangle];area=abs(np.linalg.det(np.stack((b-a,c-a))))/2
        areas[triangle]+=area/3
    if np.any(areas<=0):raise ValueError('bond surface has unused nodes')
    shear=e/(2*(1+nu));normal=e*(1-nu)/((1+nu)*(1-2*nu))
    jump=body_trace-host_trace
    weights=np.concatenate((areas*shear/h,areas*shear/h,areas*normal/h))
    stiffness=(jump.T@diags(weights)@jump).toarray()
    mass_weights=np.tile(areas*rho*h/2,3)
    mass=(body_trace.T@diags(mass_weights)@body_trace+
          host_trace.T@diags(mass_weights)@host_trace).toarray()
    return stiffness,mass,dict(**properties,area_m2=float(areas.sum()),
        adhesive_mass_kg=float(areas.sum()*rho*h),
        model='coincident elastic thin interface, area-lumped springs and endpoint-lumped adhesive mass',
        limitations=['not a resolved adhesive volume','no debonding, adhesive loss or geometric thickness offset'])
