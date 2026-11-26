"""Opt-in FP64 sparse quadratic PZT operators; no integration code.

Keep B_i D_ii^-1 B_i.T implicit. Preserve the exact consistent mass,
factoring its scalar Cartesian block rather than storing its inverse.
CPU sparse LU setup is reused by GPU triangular solves (no dropping).
"""
import math
import time
from types import SimpleNamespace
import numpy as np
import psutil
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import splu
from traxis.quadratic_piezoelectric import quadratic_disk, assemble, element_matrices
from traxis.pzt_receiver import _point_weights


def sparse_bytes(a):
    return a.data.nbytes+a.indices.nbytes+a.indptr.nbytes


def factor_bytes(lu):
    return sparse_bytes(lu.L)+sparse_bytes(lu.U)+lu.perm_r.nbytes+lu.perm_c.nbytes


def component(geometry, material, *, rings=9, sectors=64, layers=3,
              max_workspace_mib=768, max_factor_mib=512):
    """Original P2 assembly, exact electrical condensation applied implicitly.

    SuperLU has no hard workspace cap; admission bounds below are planning
    screens. Measured factors are checked after setup; never change precision.
    """
    tick=time.perf_counter()
    mesh=quadratic_disk(geometry,rings=rings,sectors=sectors,layers=layers,max_nodes=20000)
    a=assemble(mesh,material,max_nodes=20000,max_workspace_mib=max_workspace_mib)
    n=len(mesh.points_m);z=mesh.points_m[:,2]
    lower=np.flatnonzero(abs(z-z.min())<1e-14);upper=np.flatnonzero(abs(z-z.max())<1e-14)
    inside=np.setdiff1d(np.arange(n),np.r_[lower,upper]);di=a.dielectric[inside][:,inside].tocsc()
    forecast=64*sparse_bytes(di)
    if forecast>min(max_factor_mib*2**20,.35*psutil.virtual_memory().available):
        raise MemoryError('Electrical LU fill forecast exceeds available component budget')
    lu=splu(di,permc_spec='MMD_AT_PLUS_A')
    if factor_bytes(lu)>max_factor_mib*2**20:raise MemoryError('Measured electrical factors exceed budget')
    psi=np.zeros(n);psi[upper]=1.
    psi[inside]=-lu.solve(a.dielectric[inside]@psi)
    cap=float(psi@a.dielectric@psi)
    if not math.isfinite(cap) or cap<=0:raise ArithmeticError('Invalid electrode capacitance')
    bi=a.coupling[:,inside].tocsr();g=a.coupling@psi
    # Pointwise electrical maximization bounds constrained electrostatic
    # condensation: C_bound = C + e.T eps^-1 e. Element Rayleigh quotients
    # then bound the assembled body against its UNLUMPED consistent mass.
    c,e,eps=material.stress_charge
    effective=SimpleNamespace(stress_charge=(c+e.T@np.linalg.solve(eps,e),e,eps),
                              density_kg_m3=material.density_kg_m3)
    # Only rotations/translation of congruent elements can share this bound;
    # use every tetrahedron; the local dense eigenvalues include a numerical
    # safety margin rather than claiming interval-arithmetic certification.
    bound=0.
    for tet in mesh.tetrahedra:
        k,m,_,_,_=element_matrices(mesh.points_m[tet],effective)
        # Dense small symmetric eigensolve plus backward-roundoff margin.
        value=float(eigh(k,m,eigvals_only=True,subset_by_index=[29,29])[0])
        bound=max(bound,value)
    bound*=1+1e-8
    return SimpleNamespace(mesh=mesh,n=n,lower=lower,upper=upper,inside=inside,
        stiffness=a.stiffness.tocsr(),scalar_mass=a.mass[:n,:n].tocsr(),bi=bi,di=di,
        electrical_lu=lu,g=g,capacitance_f=cap,volume_m3=a.volume_m3,
        body_eigenvalue_bound=bound,seconds=time.perf_counter()-tick,
        report=dict(element='P2 affine tetrahedron',nodes=n,tetrahedra=len(mesh.tetrahedra),
                    rings=rings,sectors=sectors,layers=layers,material_id=material.id,
                    radius_m=geometry.radius_m,thickness_m=geometry.thickness_m))


class SparseAttachment:
    def __init__(self, data, layout, geometry, *, first_dof=None, electrical='open', input_capacitance_f=0.):
        if electrical not in ('open','voltage') or input_capacitance_f<0 or not math.isfinite(input_capacitance_f):
            raise ValueError('Invalid electrical port')
        if not np.array_equal(layout.transform,np.eye(3)) or geometry.in_plane_angle_rad!=0:
            raise ValueError('Aligned geometry required')
        if geometry.radius_m!=data.report['radius_m'] or geometry.thickness_m!=data.report['thickness_m']:
            raise ValueError('Cached component geometry differs from attachment')
        center=np.asarray(geometry.center_xy_m);r=geometry.radius_m
        if np.any(center-r<0) or np.any(center+r>[layout.plate.width_m,layout.plate.height_m]):
            raise ValueError('PZT outside plate')
        self.data=data;self.geometry=geometry;self.electrical=electrical
        n=data.n;ns=layout.screen['scalar_dofs'];host=3*ns
        first_dof=host if first_dof is None else first_dof
        if first_dof<host:raise ValueError('PZT DOFs must follow host')
        contact=data.lower if geometry.surface=='top' else data.upper
        self.free_nodes=np.setdiff1d(np.arange(n),contact);nf=len(self.free_nodes)
        xyz=data.mesh.points_m.copy();xyz[:,:2]+=center
        xyz[:,2]+=layout.plate.thickness_m if geometry.surface=='top' else -geometry.thickness_m
        rr=[];cc=[];vv=[];nz=layout.cells[2]*layout.order+1
        for node in contact:
            ids,w=_point_weights(layout,tuple(xyz[node]))
            keep=ids%nz==(nz-1 if geometry.surface=='top' else 0);ids,w=ids[keep],w[keep]
            if abs(w.sum()-1)>1e-11:raise ArithmeticError('Interface partition failed')
            rr.extend([node]*len(ids));cc.extend(ids);vv.extend(w)
        host_ids=np.unique(cc);nh=len(host_ids);self.nh=nh;self.nf=nf
        scalar_cols=np.r_[np.searchsorted(host_ids,cc),nh+np.arange(nf)]
        self.t=sparse.coo_matrix((np.r_[vv,np.ones(nf)],(np.r_[rr,self.free_nodes],scalar_cols)),shape=(n,nh+nf)).tocsr()
        ids=np.array([np.r_[host_ids+c*ns,first_dof+c*nf+np.arange(nf)] for c in range(3)])
        self.indices=np.unique(ids);self.positions=np.searchsorted(self.indices,ids).T
        self.next_dof=first_dof+3*nf
        self.capacitance_f=data.capacitance_f;self.total_capacitance_f=data.capacitance_f+input_capacitance_f
        self.coupling=self.push(data.g)*geometry.poling_sign
        self.host_scalar_ids=host_ids
        self.report=dict(data.report,center_xy_m=center.tolist(),surface=geometry.surface,
            radius_m=r,thickness_m=geometry.thickness_m,poling_sign=geometry.poling_sign,
            bond='perfect tie',storage='sparse implicit electrical Schur',
            body_mass_kg=data.volume_m3*float(data.scalar_mass.sum()/data.volume_m3),
            capacitance_f=self.capacitance_f,input_capacitance_f=input_capacitance_f)

    def pull(self,x):return (self.t@x[self.positions]).T.ravel()

    def push(self,u):
        y=np.empty(len(self.indices));y[self.positions]=self.t.T@u.reshape(3,self.data.n).T
        return y

    def stiffness_action(self,x):
        d=self.data;u=self.pull(x)
        result=d.stiffness@u+d.bi@d.electrical_lu.solve(d.bi.T@u)
        y=self.push(result)
        if self.electrical=='open':y+=self.coupling*(self.coupling@x)/self.total_capacitance_f
        return y

    def body_mass_action(self,x):
        return self.push((self.data.scalar_mass@self.pull(x).reshape(3,self.data.n).T).T.ravel())


class ConsistentMass:
    """Exact sparse scalar mass factor shared by all three Cartesian axes."""
    def __init__(self, attachment, host_diagonal, *, max_factor_mib=512):
        a=attachment;self.attachment=a
        h=np.asarray(host_diagonal);ns=len(h)//3
        if len(h)%3 or not np.array_equal(h[:ns],h[ns:2*ns]) or not np.array_equal(h[:ns],h[2*ns:]):
            raise ValueError('Scalar mass factor requires identical host Cartesian masses')
        if not np.isfinite(h).all() or np.any(h<=0):raise ValueError('Positive host mass required')
        q=a.t.shape[1]
        self.matrix=(a.t.T@a.data.scalar_mass@a.t+sparse.diags(np.r_[h[a.host_scalar_ids],np.zeros(a.nf)])).tocsc()
        estimate=64*sparse_bytes(self.matrix)
        if estimate>min(max_factor_mib*2**20,.35*psutil.virtual_memory().available):
            raise MemoryError('Mass LU fill forecast exceeds available component budget')
        self.factor=splu(self.matrix,permc_spec='MMD_AT_PLUS_A')
        if factor_bytes(self.factor)>max_factor_mib*2**20:raise MemoryError('Measured mass factors exceed budget')
        self.diagonal=np.empty(len(a.indices));self.diagonal[a.positions]=np.tile(self.matrix.diagonal()[:,None],(1,3))

    def action(self,x):
        out=np.empty_like(x);out[self.attachment.positions]=self.matrix@x[self.attachment.positions];return out

    def solve(self,x):
        out=np.empty_like(x);out[self.attachment.positions]=self.factor.solve(x[self.attachment.positions]);return out


def receipt(a,m):
    d=a.data
    arrays=[d.stiffness,d.scalar_mass,d.bi,d.di,a.t,m.matrix]
    bytes_=sum(sparse_bytes(x) for x in arrays)+factor_bytes(d.electrical_lu)+factor_bytes(m.factor)
    bytes_+=a.indices.nbytes+a.positions.nbytes+3*a.coupling.nbytes+m.diagonal.nbytes
    return dict(patch_dofs=len(a.indices),component_nodes=d.n,
        sparse_arrays_and_factors_bytes=bytes_,dense_three_patch_matrices_bytes=3*len(a.indices)**2*8,
        stiffness_nnz=d.stiffness.nnz,coupling_nnz=d.bi.nnz,mass_nnz=m.matrix.nnz,
        mass_factor_nnz=m.factor.L.nnz+m.factor.U.nnz,
        electrical_factor_nnz=d.electrical_lu.L.nnz+d.electrical_lu.U.nnz,
        action_cost='K/ B/ T sparse products, one dielectric factor solve; one rank-one update for open port. Mass inverse: one scalar factor solve with three RHS; no explicit inverse.',
        setup_seconds=d.seconds)
