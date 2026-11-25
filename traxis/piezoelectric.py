"""Linear small-signal piezoelectric material, electrodes and passive readout.

SI units; engineering Voigt order xx,yy,zz,yz,xz,xy. Axis 3 is the poling
direction. V=phi_top-phi_bottom; Q_top=C*V-q_strain. Lossless material
tensors do not identify bond mechanics or broadband dielectric loss.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import solve

EPSILON_0 = 8.8541878128e-12


def _array(value, shape):
    if np.iscomplexobj(value):
        raise ValueError('material data must be real')
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('invalid material shape or nonfinite values')
    return result


def _positive_matrix(value):
    if not np.allclose(value,value.T,rtol=1e-12,atol=0.):
        raise ValueError('constitutive matrix must be symmetric')
    try: np.linalg.cholesky(value)
    except np.linalg.LinAlgError as exc:
        raise ValueError('constitutive matrix must be positive definite') from exc


def read_profile(path):
    path=Path(path)
    if path.stat().st_size>128*1024:raise MemoryError('PZT profile exceeds 128 KiB')
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema')!=1 or not data.get('source') or not data.get('id'):
        raise ValueError('PZT profile needs a version, identity and provenance')
    return data


@dataclass(frozen=True)
class PlanarPiezo:
    id: str
    coupling_c_m2: np.ndarray
    permittivity_f_m: float
    density_kg_m3: float
    provenance: dict

    def __post_init__(self):
        values=_array(self.coupling_c_m2,(3,));values.setflags(write=False)
        object.__setattr__(self,'coupling_c_m2',values)
        if any(not math.isfinite(x) or x<=0 for x in (self.permittivity_f_m,self.density_kg_m3)):
            raise ValueError('permittivity and density must be positive')


@dataclass(frozen=True)
class PiezoMaterial:
    id: str
    compliance_E_m2_n: np.ndarray
    d_c_n: np.ndarray
    permittivity_T_f_m: np.ndarray
    density_kg_m3: float
    provenance: dict

    def __post_init__(self):
        s=_array(self.compliance_E_m2_n,(6,6));d=_array(self.d_c_n,(3,6))
        eps=_array(self.permittivity_T_f_m,(3,3));_positive_matrix(s);_positive_matrix(eps)
        if not math.isfinite(self.density_kg_m3) or self.density_kg_m3<=0:
            raise ValueError('density must be positive')
        c=np.linalg.inv(s);e=d@c;clamped=eps-d@c@d.T
        _positive_matrix(clamped)
        for name,a in [('compliance_E_m2_n',s),('d_c_n',d),('permittivity_T_f_m',eps)]:
            a.setflags(write=False);object.__setattr__(self,name,a)

    @classmethod
    def from_profile(cls,data):
        if data.get('basis')!='strain-charge-3d':
            raise ValueError('full 3D coefficients are unavailable in this profile')
        if (data.get('voigt_order')!=['xx','yy','zz','yz','xz','xy']
                or data.get('shear_convention')!='engineering' or data.get('poling_axis')!=3):
            raise ValueError('unsupported tensor convention')
        return cls(data['id'],data['compliance_E_m2_n'],data['d_c_n'],
                   np.diag(data['relative_permittivity_T'])*EPSILON_0,
                   data['density_kg_m3'],data['source'])

    @property
    def stress_charge(self):
        c=np.linalg.inv(self.compliance_E_m2_n);e=self.d_c_n@c
        return c,e,self.permittivity_T_f_m-self.d_c_n@c@self.d_c_n.T

    def response(self,strain,electric_field_v_m):
        strain=_array(strain,(6,));field=_array(electric_field_v_m,(3,))
        c,e,eps=self.stress_charge
        return c@strain-e.T@field,e@strain+eps@field

    def planar(self):
        # Eliminate zz,yz,xz stresses, retaining xx,yy,gamma_xy and E3.
        c,e,eps=self.stress_charge;p=[0,1,5];o=[2,3,4]
        eliminated=solve(c[np.ix_(o,o)],c[np.ix_(o,p)],assume_a='pos')
        coupling=e[2,p]-e[2,o]@eliminated
        permittivity=eps[2,2]+e[2,o]@solve(c[np.ix_(o,o)],e[2,o],assume_a='pos')
        return PlanarPiezo(self.id,coupling,float(permittivity),self.density_kg_m3,
                           {**self.provenance,'reduction':'sigma_zz=sigma_yz=sigma_xz=0; E1=E2=0'})


def planar_from_profile(data):
    if data.get('basis')=='strain-charge-3d':return PiezoMaterial.from_profile(data).planar()
    if data.get('basis')!='planar-d31-kp':raise ValueError('unknown PZT material basis')
    d=float(data['d31_c_n']);kp=float(data['planar_coupling_kp'])
    eps=float(data['relative_permittivity_T_33'])*EPSILON_0
    if not math.isfinite(d) or d==0 or not 0<kp<1 or not math.isfinite(eps) or eps<=0:
        raise ValueError('invalid planar piezoelectric properties')
    compliance_sum=2*d*d/(kp*kp*eps)
    coupling=d/compliance_sum
    return PlanarPiezo(data['id'],np.array([coupling,coupling,0.]),eps*(1-kp*kp),
        data['density_kg_m3'],{**data['source'],
        'reduction':'ideal planar d31/kp model; not a full 3D tensor',
        'derived_compliance_sum_m2_n':compliance_sum})


@dataclass(frozen=True)
class Readout:
    """Parallel input resistance and capacitance, specified by the caller.

    None resistance is an ideal open circuit, not an identified instrument.
    A short circuit is represented separately by short_circuit_charge().
    """
    resistance_ohm: float | None
    capacitance_f: float

    def __post_init__(self):
        if self.resistance_ohm is not None and (not math.isfinite(self.resistance_ohm) or self.resistance_ohm<=0):
            raise ValueError('resistance must be positive or None for open circuit')
        if self.resistance_ohm is not None and not math.isfinite(1/self.resistance_ohm):
            raise ValueError('readout conductance overflows')
        if not math.isfinite(self.capacitance_f) or self.capacitance_f<0:
            raise ValueError('input/cable capacitance must be nonnegative')

    @property
    def conductance_s(self):return 0. if self.resistance_ohm is None else 1/self.resistance_ohm


def short_circuit_charge(strain_charge_c):
    q=np.asarray(strain_charge_c)
    if np.iscomplexobj(q) or not np.isfinite(q).all():raise ValueError('charge must be finite and real')
    return -q


def loaded_voltage(time_s,strain_charge_c,capacitance_f,readout,*,initial_free_charge_c=0.):
    """Exact passive RC update for piecewise-linear strain charge.

    (Cp+Cin)*Vdot + G*V = q_strain_dot. Initial free charge includes
    the receiver and parallel input capacitance, with a grounded bottom.
    """
    t=np.asarray(time_s);q=np.asarray(strain_charge_c)
    if (np.iscomplexobj(t) or np.iscomplexobj(q) or t.ndim!=1 or q.shape!=t.shape
            or not 2<=len(t)<=1000000 or not np.isfinite(t).all() or not np.isfinite(q).all()
            or np.any(np.diff(t)<=0) or not math.isfinite(capacitance_f) or capacitance_f<=0
            or not math.isfinite(initial_free_charge_c)):
        raise ValueError('invalid receiver trace or capacitance')
    t=t.astype(float);q=q.astype(float);c=capacitance_f+readout.capacitance_f;g=readout.conductance_s
    if not math.isfinite(c) or not np.isfinite(np.diff(t)).all() or np.any(np.diff(t)<=0):
        raise ValueError('time intervals and total capacitance must be finite and positive')
    v=np.empty(len(t));v[0]=(initial_free_charge_c+q[0])/c
    drained=np.zeros(len(t))
    for i,h in enumerate(np.diff(t),1):
        a=g*h/c;phi=1. if a==0 else -math.expm1(-a)/a
        v[i]=math.exp(-a)*v[i-1]+(q[i]-q[i-1])/c*phi
        drained[i]=drained[i-1]+c*v[i-1]*(-math.expm1(-a))+(q[i]-q[i-1])*(1-phi)
    free=c*v-q
    if not all(np.isfinite(a).all() for a in (v,drained,free)):
        raise ArithmeticError('receiver circuit produced nonfinite values')
    return dict(time_s=t,voltage_v=v,electrode_charge_c=capacitance_f*v-q,
                input_capacitor_charge_c=readout.capacitance_f*v,
                charge_drained_c=drained,charge_balance_c=free-free[0]+drained,
                stored_electrical_energy_j=.5*c*v*v)


def _strain_matrix(g):
    b=np.zeros((6,12));b[0,:4],b[1,4:8],b[2,8:]=g
    b[3,4:8],b[3,8:]=g[2],g[1]
    b[4,:4],b[4,8:]=g[2],g[0]
    b[5,:4],b[5,4:8]=g[1],g[0]
    return b


@dataclass
class PiezoSystem:
    stiffness: np.ndarray
    mass: np.ndarray
    coupling: np.ndarray
    dielectric: np.ndarray
    volume_m3: float


def assemble_piezoelectric(mesh,material,*,max_bytes=256*1024**2,guard=None,
                           max_nodes=1024,max_elements=8192):
    """Consistent 3D linear-tetrahedron coupled component, with all strain terms.

    Bounded component assembly, not a global dense full-plate backend.
    Mechanical DOFs use component-major order. Potential is nodal.
    """
    points=np.asarray(mesh.points_m);tets=np.asarray(mesh.tetrahedra);n=len(points)
    if (type(max_nodes) is not int or not 4<=max_nodes<=8192
            or type(max_elements) is not int or not 1<=max_elements<=65536):
        raise ValueError('invalid explicit component mesh budget')
    if (points.shape!=(n,3) or np.iscomplexobj(points) or not np.isfinite(points).all()
            or tets.ndim!=2 or tets.shape[1]!=4 or not np.issubdtype(tets.dtype,np.integer)
            or not 4<=n<=max_nodes or not 1<=len(tets)<=max_elements
            or np.any(tets<0) or np.any(tets>=n)):
        raise ValueError('coupled PZT component needs a bounded valid tetrahedral mesh')
    if type(max_bytes) is not int or not 0<max_bytes<=16*1024**3 or 176*n*n>max_bytes:
        raise MemoryError('coupled PZT component exceeds its assembly budget')
    c,e,eps=material.stress_charge
    k=np.zeros((3*n,3*n));m=np.zeros_like(k);b=np.zeros((3*n,n));d=np.zeros((n,n));volume=0.
    for tet in tets:
        if guard:guard.check()
        xyz=points[tet];jac=(xyz[1:]-xyz[0]).T
        if len(set(tet))!=4 or np.linalg.cond(jac)>1e8:
            raise ValueError('degenerate or poorly conditioned PZT element')
        v=abs(np.linalg.det(jac))/6
        if not math.isfinite(v) or v<=0:raise ValueError('PZT element volume must be finite and positive')
        g=np.empty((3,4));g[:,1:]=np.linalg.inv(jac).T;g[:,0]=-g[:,1:].sum(axis=1)
        strain=_strain_matrix(g);ids=np.concatenate([tet+i*n for i in range(3)])
        k[np.ix_(ids,ids)]+=v*strain.T@c@strain
        b[np.ix_(ids,tet)]+=v*strain.T@e.T@g
        d[np.ix_(tet,tet)]+=v*g.T@eps@g
        m[np.ix_(ids,ids)]+=np.kron(np.eye(3),material.density_kg_m3*v/20*(np.ones((4,4))+np.eye(4)))
        volume+=v
    return PiezoSystem(k,m,b,d,float(volume))


@dataclass
class ElectrodePort:
    stiffness_sc: np.ndarray
    mass: np.ndarray
    coupling_n_v: np.ndarray
    capacitance_f: float
    potential_per_v: np.ndarray
    potential_per_displacement: np.ndarray

    def charge(self,displacement_m,voltage_v):
        u=_array(displacement_m,self.coupling_n_v.shape)
        if not math.isfinite(voltage_v):raise ValueError('voltage must be finite')
        return self.capacitance_f*voltage_v-self.coupling_n_v@u

    def force(self,displacement_m,voltage_v):
        u=_array(displacement_m,self.coupling_n_v.shape)
        if not math.isfinite(voltage_v):raise ValueError('voltage must be finite')
        return self.stiffness_sc@u+self.coupling_n_v*voltage_v

    def harmonic_response(self,frequency_hz,force_n,readout,*,fixed_dofs):
        """Full mechanical/electrical backreaction, inertia and a passive load.

        Caller supplies mechanical attachment constraints; none are inferred.
        No catalog-Q damping or adhesive layer is invented.
        """
        if not math.isfinite(frequency_hz) or frequency_hz<=0:raise ValueError('positive frequency required')
        force=np.asarray(force_n,dtype=complex);fixed=np.asarray(fixed_dofs)
        if force.shape!=self.coupling_n_v.shape or not np.isfinite(force).all():
            raise ValueError('invalid harmonic force')
        if fixed.size==0:fixed=np.array([],dtype=int)
        n=len(force)
        if (fixed.ndim!=1 or not np.issubdtype(fixed.dtype,np.integer) or len(np.unique(fixed))!=len(fixed)
                or np.any(fixed<0) or np.any(fixed>=n)):
            raise ValueError('invalid mechanical constraints')
        free=np.setdiff1d(np.arange(n),fixed)
        if not len(free):raise ValueError('at least one mechanical DOF must be free')
        k=self.stiffness_sc[np.ix_(free,free)];m=self.mass[np.ix_(free,free)];g=self.coupling_n_v[free]
        w=2*math.pi*frequency_hz;cap=self.capacitance_f+readout.capacitance_f
        a=np.block([[k-w*w*m,g[:,None]],[g[None,:],np.array([[-cap+1j*readout.conductance_s/w]])]])
        scale=np.sqrt(np.r_[np.diag(k)+w*w*np.diag(m),cap])
        x=solve(a/scale[:,None]/scale[None,:],np.r_[force[free],0.]/scale,assume_a='sym')/scale
        u=np.zeros(n,complex);u[free]=x[:-1]
        return dict(displacement_m=u,voltage_v=x[-1],electrode_charge_c=self.capacitance_f*x[-1]-g@x[:-1])


def electrode_port(system,bottom,top):
    """Ground bottom, equipotential top, zero free charge on unelectroded sides."""
    d=system.dielectric;n=len(d);bottom=np.asarray(bottom);top=np.asarray(top)
    for ids in (bottom,top):
        if (ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or not len(ids)
                or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=n)):
            raise ValueError('invalid electrode vertices')
    if np.intersect1d(bottom,top).size:raise ValueError('electrodes overlap')
    inside=np.setdiff1d(np.arange(n),np.r_[bottom,top]);psi=np.zeros(n);psi[top]=1.
    correction=np.zeros((n,len(system.mass)))
    if len(inside):
        di=d[np.ix_(inside,inside)]
        psi[inside]=-solve(di,d[inside]@psi,assume_a='pos')
        correction[inside]=solve(di,system.coupling[:,inside].T,assume_a='pos')
    cap=float(psi@d@psi)
    if not math.isfinite(cap) or cap<=0:raise ValueError('electrode capacitance is not positive')
    return ElectrodePort(system.stiffness+system.coupling@correction,system.mass.copy(),
                         system.coupling@psi,cap,psi,correction)
