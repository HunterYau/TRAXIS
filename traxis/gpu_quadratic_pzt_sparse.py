"""Sparse operator adapter for UNCHANGED original LTS LocalActions.

Subclass BodyStorage only for its adapter interface. Do not call its dense
constructor. CUDA graphs / ReusableActions and cached dense stability
normalization are deliberately unsupported; callers must select the original
unchecked study runner, or provide a separately verified compatible screen.
"""
import math
import numpy as np
from types import SimpleNamespace
from traxis.gpu_pzt_body import BodyStorage
from traxis.gpu_elements import GPUElementOperator
from traxis.quadratic_pzt_sparse import receipt


class DeviceFactor:
    """Exact FP64 factors; explicit optional CPU solve for small components.

    Only component RHS/solutions cross PCIe in CPU mode, never plate vectors.
    GPU mode reuses full LU factors on device; no incomplete factorization.
    """
    def __init__(self, factor, backend='gpu'):
        import cupy as cp
        from cupyx.scipy.sparse.linalg import SuperLU
        if backend not in ('gpu','cpu'):raise ValueError('Unknown factor solve backend')
        self.cp=cp;self.backend=backend;self.factor=SuperLU(factor) if backend=='gpu' else factor

    def solve(self,rhs):
        if self.backend=='gpu':return self.factor.solve(rhs)
        return self.cp.asarray(self.factor.solve(self.cp.asnumpy(rhs)))


class DeviceComponent:
    def __init__(self, data, factor_backend='gpu'):
        import cupy as cp
        from cupyx.scipy import sparse
        self.cp=cp;self.n=data.n
        self.k=sparse.csr_matrix(data.stiffness);self.b=sparse.csr_matrix(data.bi)
        self.bt=self.b.T.tocsr();self.factor=DeviceFactor(data.electrical_lu,factor_backend)


class DevicePatch:
    def __init__(self, attachment, mass, component, factor_backend='gpu'):
        import cupy as cp
        from cupyx.scipy import sparse
        self.cp=cp;self.component=component
        self.ids=cp.asarray(attachment.indices);self.positions=cp.asarray(attachment.positions)
        self.t=sparse.csr_matrix(attachment.t);self.tt=self.t.T.tocsr()
        self.g=cp.asarray(attachment.coupling);self.cap=attachment.total_capacitance_f
        self.open=attachment.electrical=='open'
        self.mass=sparse.csr_matrix(mass.matrix);self.factor=DeviceFactor(mass.factor,factor_backend)

    def stiffness(self,x):
        d=self.component
        u=(self.t@x[self.positions]).T.ravel()
        v=d.k@u+d.b@d.factor.solve(d.bt@u)
        out=self.cp.empty_like(x);out[self.positions]=self.tt@v.reshape(3,d.n).T
        if self.open:out+=self.g*(self.g@x)/self.cap
        return out

    def mass_action(self,x):
        out=self.cp.empty_like(x);out[self.positions]=self.mass@x[self.positions];return out

    def inverse(self,x):
        out=self.cp.empty_like(x);out[self.positions]=self.factor.solve(x[self.positions]);return out


class Action:
    def __init__(self, action):self.action=action


class SparseBodyStorage(BodyStorage):
    def __init__(self,base,diagonal,patches,screen,guard):
        self.base=base;self.cp=base.cp;self.device_id=base.device_id;self.lock=base.lock
        self.guard=guard;self.screen=screen;self.ns=len(diagonal)//3;self.host_dofs=3*base.ns
        self.applications=self.vector_uploads=self.vector_downloads=self.preconditioner_applications=0
        self.input=self.cp.empty(len(diagonal),dtype=self.cp.float64);self.output=self.cp.empty_like(self.input)
        self.diagonal=self.cp.asarray(diagonal);self.device_patches=patches
        self.patches=[(p.ids,Action(p.stiffness),Action(p.mass_action),Action(p.inverse)) for p in patches]

    def matvec(self,matrix,values):return matrix.action(values)

    def _launch(self,values,result,stiffness_scale=0.,mass_scale=0.,*,inverse=False):
        if inverse or mass_scale!=0:raise ValueError('Use coupled mass actions explicitly')
        result.fill(0.);self.base._launch(values[:self.host_dofs],result[:self.host_dofs],stiffness_scale,0.)
        for p in self.device_patches:result[p.ids]+=stiffness_scale*p.stiffness(values[p.ids])
        self.applications+=1

    def inverse_mass_inplace(self,values):
        saved=[values[p.ids] for p in self.device_patches]
        values/=self.diagonal
        for p,x in zip(self.device_patches,saved):values[p.ids]=p.inverse(x)

    def mass_quadratic(self,values,*,inverse=False):
        diagonal=1/self.diagonal if inverse else self.diagonal
        total=self.cp.sum(diagonal*values*values,dtype=self.cp.float64)
        for p in self.device_patches:
            x=values[p.ids];y=p.inverse(x) if inverse else p.mass_action(x)
            total+=self.cp.sum(x*y-diagonal[p.ids]*x*x,dtype=self.cp.float64)
        return total


def couple_sparse(base,attachments,masses,*,guard=None,factor_backend='gpu'):
    """Three disjoint scalar support blocks; reject overlap, never double solve.

    Full merged-support handling remains available through the dense reference.
    No unsupported overlap is silently approximated.
    """
    import cupy as cp
    if len(attachments)!=len(masses) or not attachments:raise ValueError('Matching attachments and masses required')
    next_dof=len(base.mass_diagonal);seen=set()
    for a in attachments:
        if seen.intersection(a.indices.tolist()):raise ValueError('Sparse backend requires disjoint mass supports; use dense reference for overlaps')
        seen.update(a.indices.tolist())
        extra=a.indices[a.indices>=len(base.mass_diagonal)]
        if not np.array_equal(extra,np.arange(next_dof,a.next_dof)):raise ValueError('Nonconsecutive PZT DOFs')
        next_dof=a.next_dof
    diagonal=np.r_[base.mass_diagonal,np.zeros(next_dof-len(base.mass_diagonal))]
    for a,m in zip(attachments,masses):diagonal[a.indices]=m.diagonal
    # Conservative payload forecast: includes duplicate components, transpose
    # copies, LU permutations, local work, and global buffers, before upload.
    payload=sum(2*receipt(a,m)['sparse_arrays_and_factors_bytes'] for a,m in zip(attachments,masses))
    extra=payload+3*len(diagonal)*8
    storage_bytes=base.screen['device_storage_screen_mib']*2**20+extra
    if storage_bytes>base.screen['max_device_mib']*2**20 or extra>.7*cp.cuda.runtime.memGetInfo()[0]:
        raise MemoryError('Sparse PZT device storage forecast exceeds allocation')
    screen=dict(base.screen,vector_dofs=len(diagonal),device_storage_screen_mib=storage_bytes/2**20,
        body_device_storage_mib=extra/2**20,coupled_mass='exact sparse consistent mass, full LU factors',
        body_reports=[a.report for a in attachments],mass_inverse_errors=[],pzt_storage='sparse',factor_backend=factor_backend)
    stable=2/math.sqrt(4/base.stable_dt_s**2+max(a.data.body_eigenvalue_bound for a in attachments))
    screen['stable_dt_s']=stable
    with base.stiffness.storage.lock,cp.cuda.Device(base.stiffness.storage.device_id):
        cache={};patches=[]
        for a,m in zip(attachments,masses):
            if id(a.data) not in cache:cache[id(a.data)]=DeviceComponent(a.data,factor_backend)
            patches.append(DevicePatch(a,m,cache[id(a.data)],factor_backend))
        storage=SparseBodyStorage(base.stiffness.storage,diagonal,patches,screen,guard)
    return SimpleNamespace(stiffness=GPUElementOperator(storage,1.,0.),mass_diagonal=diagonal,
        stable_dt_s=stable,screen=screen,layout=base.layout,device=base.device,
        body_attachments=attachments,body_patches=[SimpleNamespace(indices=a.indices) for a in attachments],
        inverse_mass_inplace=storage.inverse_mass_inplace,mass_quadratic=storage.mass_quadratic)
