"""Circular plane-stress PZT charge probes for the existing tensor plate backend.

This adapter is a one-way, perfectly bonded, thin-layer approximation. The
separate 3D coupled component in piezoelectric.py includes mechanical inertia
and electrical backreaction; this observation adapter does not add PZT mass,
stiffness or an adhesive domain to the plate operator.
"""
from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial.legendre import leggauss

from traxis.gpu_enriched_verlet import sparse_probe_set
from traxis.piezoelectric import loaded_voltage
from traxis.tensor_hex import line_rule
from traxis.tensor_patches import _basis


@dataclass(frozen=True)
class DiskGeometry:
    center_xy_m: tuple
    radius_m: float
    thickness_m: float
    surface: str
    poling_sign: int
    in_plane_angle_rad: float = 0.

    def __post_init__(self):
        xy=np.asarray(self.center_xy_m)
        if (xy.shape!=(2,) or np.iscomplexobj(xy) or not np.isfinite(xy).all()
                or any(not math.isfinite(v) or v<=0 for v in (self.radius_m,self.thickness_m))
                or self.surface not in ('top','bottom') or type(self.poling_sign) is not int
                or self.poling_sign not in (-1,1) or not math.isfinite(self.in_plane_angle_rad)):
            raise ValueError('invalid PZT disk geometry, surface or poling direction')
        object.__setattr__(self,'center_xy_m',tuple(float(x) for x in xy))


def _point_weights(layout,xyz):
    p=layout.order;basis=_basis(line_rule(p)[0]);ids=[];weights=[]
    lengths=(layout.plate.width_m,layout.plate.height_m,layout.plate.thickness_m)
    for x,length,count in zip(xyz,lengths,layout.cells):
        cell=min(int(math.floor(x/length*count)),count-1)
        local=2*(x/length*count-cell)-1
        ids.append(cell*p+np.arange(p+1));weights.append(np.array([f(local) for f in basis]))
    nodes=((ids[0][:,None,None]*(layout.cells[1]*p+1)+ids[1][None,:,None])
           *(layout.cells[2]*p+1)+ids[2][None,None,:]).ravel()
    values=np.einsum('i,j,k->ijk',*weights).ravel()
    keep=values!=0
    return nodes[keep],values[keep]


@dataclass
class DiskReceiver:
    indices: np.ndarray
    charge_weights_c_m: np.ndarray
    vector_dofs: int
    capacitance_f: float
    report: dict

    def strain_charge(self,displacement_m):
        u=np.asarray(displacement_m)
        if (np.iscomplexobj(u) or u.shape!=(self.vector_dofs,) or not np.isfinite(u).all()):
            raise ValueError('receiver requires a finite component-major displacement state')
        return float(self.charge_weights_c_m@u[self.indices])

    def probes(self):
        return sparse_probe_set([(self.indices,self.charge_weights_c_m)],self.vector_dofs)

    def record(self,time_s,strain_charge_c,readout,*,initial_free_charge_c=0.):
        return loaded_voltage(time_s,strain_charge_c,self.capacitance_f,readout,
                              initial_free_charge_c=initial_free_charge_c)


def disk_receiver(layout,material,geometry,*,quadrature=24,max_coefficients=100000,
                  max_workspace_mib=64,guard=None):
    """Integrate q_strain=int_A ebar:[eps_xx,eps_yy,gamma_xy] dA.

    Integration by parts turns the area integral into a rim integral of
    displacement. Split at every element crossing; retain signed FE weights.
    Thus real spatial cancellation is retained rather than replaced by an
    arbitrary gain or a single point displacement.
    """
    if not np.array_equal(layout.transform,np.eye(3)):
        raise ValueError('PZT disk adapter currently requires an axis-aligned tensor plate')
    if type(quadrature) is not int or not 12<=quadrature<=64:
        raise ValueError('rim quadrature must have 12-64 points per segment')
    if type(max_coefficients) is not int or not 1<=max_coefficients<=100000:
        raise ValueError('invalid sparse receiver coefficient budget')
    if not math.isfinite(max_workspace_mib) or not 0<max_workspace_mib<=256:
        raise ValueError('invalid receiver workspace budget')
    center=np.array(geometry.center_xy_m);r=geometry.radius_m
    dimensions=[layout.plate.width_m,layout.plate.height_m]
    if np.any(center-r<0) or np.any(center+r>dimensions):
        raise ValueError('entire PZT must lie on the plate surface')
    z=layout.plate.thickness_m if geometry.surface=='top' else 0.
    ex,ey,exy=material.coupling_c_m2*geometry.poling_sign
    angle=geometry.in_plane_angle_rad;co,si=math.cos(angle),math.sin(angle)
    rotation=np.array([[co,-si],[si,co]])
    coupling=rotation@np.array([[ex,exy],[exy,ey]])@rotation.T
    cuts=[0.,2*math.pi]
    for axis,(length,count) in enumerate(zip(dimensions,layout.cells[:2])):
        for edge in np.linspace(0,length,count+1):
            ratio=(edge-center[axis])/r
            if -1<ratio<1:
                if axis==0:
                    a=math.acos(ratio);cuts.extend((a,2*math.pi-a))
                else:
                    a=math.asin(ratio);cuts.extend((a%(2*math.pi),(math.pi-a)%(2*math.pi)))
    cuts=sorted(set(cuts))
    # Include raw coefficient lists, concatenation, inverse map and sorting.
    forecast=(len(cuts)-1)*quadrature*(layout.order+1)**3*2*96+1024**2
    if forecast>max_workspace_mib*1024**2:
        raise MemoryError('PZT rim integration exceeds its workspace budget')
    nodes,weights=leggauss(quadrature)
    ns=layout.screen['scalar_dofs'];ids=[];values=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        if guard:guard.check()
        for theta,w in zip((nodes+1)*(b-a)/2+a,weights*(b-a)/2*r):
            normal=np.array([math.cos(theta),math.sin(theta)])
            scalar,basis=_point_weights(layout,(*tuple(center+r*normal),z))
            vector=coupling@normal
            for component in (0,1):
                ids.append(scalar+component*ns);values.append(basis*w*vector[component])
    flat=np.concatenate(ids);unique,inverse=np.unique(flat,return_inverse=True)
    coefficients=np.zeros(len(unique));np.add.at(coefficients,inverse,np.concatenate(values))
    keep=coefficients!=0;unique=unique[keep];coefficients=coefficients[keep]
    if len(unique)>max_coefficients:raise MemoryError('PZT rim probe exceeds sparse support budget')
    area=math.pi*r*r;cap=material.permittivity_f_m*area/geometry.thickness_m
    return DiskReceiver(unique,coefficients,3*ns,cap,dict(
        material_id=material.id,provenance=material.provenance,area_m2=area,capacitance_f=cap,
        physical_disc_mass_kg=material.density_kg_m3*area*geometry.thickness_m,
        center_xy_m=list(center),radius_m=r,thickness_m=geometry.thickness_m,
        surface=geometry.surface,poling_sign=geometry.poling_sign,
        angle_rad=angle,quadrature_points_per_segment=quadrature,
        response='area-integrated piezoelectric strain charge; passive RC voltage',
        charge_units='C',voltage_units='V',
        limitations=['one-way observation; no plate mechanical backreaction',
                     'perfect strain transfer; no inferred adhesive/contact model',
                     'quasistatic plane stress; radial resonance not represented',
                     'full face electrodes, no wire/electrode mass or broadband losses',
                     'not calibrated against experimental voltage or accepted for training']))
