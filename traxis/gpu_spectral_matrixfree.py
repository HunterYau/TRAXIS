"""Separate exact PZT operator adapter; compatible with UNCHANGED LocalActions.

Stiffness/coupling: Q2 tensor partial assembly. Electrical inverse: persistent
sparse LU plans. Coupled consistent mass: one scalar matrix/factor for 3 RHS.
Work arrays are owned and reused. Calls must be serialized by the system lock.
"""
import math
from types import SimpleNamespace
import numpy as np
from traxis.gpu_spectral_partial import DeviceTensor
from traxis.gpu_cached_lu import CachedLU
from traxis.gpu_quadratic_pzt_sparse import SparseBodyStorage
from traxis.gpu_elements import GPUElementOperator

KERNELS=r'''
extern "C" __global__ void pull(const int* ptr,const int* col,const double* v,
 const int* pos,const double* x,double* u,int n,int q) {
 int t=blockIdx.x*blockDim.x+threadIdx.x;
 if(t<3*n){int c=t/n,row=t%n;double s=0.;
 for(int j=ptr[row];j<ptr[row+1];++j)s+=v[j]*x[pos[3*col[j]+c]];u[t]=s;}
}
extern "C" __global__ void push(const int* ptr,const int* col,const double* v,
 const int* pos,const double* u,double* x,int n,int q) {
 int t=blockIdx.x*blockDim.x+threadIdx.x;
 if(t<3*q){int c=t/q,row=t%q;double s=0.;
 for(int j=ptr[row];j<ptr[row+1];++j)s+=v[j]*u[c*n+col[j]];x[pos[3*row+c]]=s;}
}
extern "C" __global__ void pack(const int* pos,const double* x,double* rhs,int q) {
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t<3*q)rhs[t]=x[pos[3*(t%q)+t/q]];
}
extern "C" __global__ void unpack(const int* pos,const double* rhs,double* x,int q) {
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t<3*q)x[pos[3*(t%q)+t/q]]=rhs[t];
}
extern "C" __global__ void extract(const int* ids,const double* x,double* y,int n) {
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t<n)y[t]=x[ids[t]];
}
extern "C" __global__ void insert(const int* ids,const double* x,double* y,int n) {
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t<n)y[ids[t]]=x[t];
}
extern "C" __global__ void mass(const int* ptr,const int* col,const double* v,
 const int* pos,const double* x,double* out,int q) {
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t<3*q){int c=t/q,row=t%q;double s=0.;
 for(int j=ptr[row];j<ptr[row+1];++j)s+=v[j]*x[pos[3*col[j]+c]];out[pos[3*row+c]]=s;}
}
extern "C" __global__ void rank_one(const double* g,const double* x,double* out,int n,double cap) {
 __shared__ double sums[256];int t=threadIdx.x;double s=0.;
 for(int i=t;i<n;i+=256)s+=g[i]*x[i];sums[t]=s;__syncthreads();
 for(int d=128;d;d/=2){if(t<d)sums[t]+=sums[t+d];__syncthreads();}
 for(int i=t;i<n;i+=256)out[i]+=g[i]*sums[0]/cap;
}
'''


def csr_arrays(a):
    import cupy as cp
    a=a.tocsr()
    return cp.asarray(a.indptr,dtype=cp.int32),cp.asarray(a.indices,dtype=cp.int32),cp.asarray(a.data)


class DeviceComponent:
    def __init__(self,data):
        import cupy as cp
        self.cp=cp;self.n=data.n;self.tensor=DeviceTensor(data.tensor)
        self.electrical=CachedLU(data.electrical_lu,1)
        self.inside=cp.asarray(data.inside,dtype=cp.int32)
        self.u=cp.empty(3*self.n,dtype=cp.float64)
        self.k=cp.empty_like(self.u);self.correction=cp.empty_like(self.u)
        self.charge=cp.empty(self.n,dtype=cp.float64);self.phi=cp.empty_like(self.charge)
        self.dummy=cp.empty_like(self.charge)


class DevicePatch:
    def __init__(self,a,m,component,module):
        import cupy as cp
        self.cp=cp;self.component=component;self.module=module
        self.ids=cp.asarray(a.indices);self.pos=cp.asarray(a.positions,dtype=cp.int32,order='C')
        self.t=csr_arrays(a.t);self.tt=csr_arrays(a.t.T)
        self.mass=csr_arrays(m.matrix);self.factor=CachedLU(m.factor,3)
        self.g=cp.asarray(a.coupling);self.cap=a.total_capacitance_f;self.open=a.electrical=='open'
        self.q=a.t.shape[1];self.n=a.data.n
        self.stiff_out=cp.empty(len(a.indices),dtype=cp.float64)
        self.mass_out=cp.empty_like(self.stiff_out);self.inverse_out=cp.empty_like(self.stiff_out)
        self.ops={name:module.get_function(name) for name in ('pull','push','pack','unpack','extract','insert','mass','rank_one')}

    def run(self,name,n,args):
        self.ops[name](((n+255)//256,),(256,),args)

    def stiffness(self,x):
        d=self.component
        self.run('pull',3*self.n,(*self.t,self.pos,x,d.u,np.int32(self.n),np.int32(self.q)))
        d.tensor.apply(d.u,d.k,d.charge,0)
        ni=len(d.inside)
        self.run('extract',ni,(d.inside,d.charge,d.electrical.rhs,np.int32(ni)))
        phi=d.electrical.solve();d.phi.fill(0.)
        self.run('insert',ni,(d.inside,phi,d.phi,np.int32(ni)))
        # The dielectric matrix is already factored; only B*phi is needed here.
        d.tensor.apply(d.phi,d.correction,d.dummy,3)
        self.cp.add(d.k,d.correction,out=d.k)
        self.run('push',3*self.q,(*self.tt,self.pos,d.k,self.stiff_out,np.int32(self.n),np.int32(self.q)))
        if self.open:self.ops['rank_one']((1,),(256,),(self.g,x,self.stiff_out,np.int32(len(x)),np.float64(self.cap)))
        return self.stiff_out

    def mass_action(self,x):
        self.run('mass',3*self.q,(*self.mass,self.pos,x,self.mass_out,np.int32(self.q)))
        return self.mass_out

    def inverse(self,x):
        self.run('pack',3*self.q,(self.pos,x,self.factor.rhs,np.int32(self.q)))
        result=self.factor.solve()
        self.run('unpack',3*self.q,(self.pos,result,self.inverse_out,np.int32(self.q)))
        return self.inverse_out

    def bytes(self):
        return sum(v.nbytes for values in (self.t,self.tt,self.mass) for v in values)+sum(v.nbytes for v in
            (self.ids,self.pos,self.g,self.stiff_out,self.mass_out,self.inverse_out))+self.factor.bytes()


def couple_matrixfree(base,attachments,masses,*,guard=None):
    """No dense patches. Existing LTS adapter sees the same K/M/M^-1 actions."""
    import cupy as cp
    if not attachments or len(attachments)!=len(masses):raise ValueError('Matching attachments/masses required')
    next_dof=len(base.mass_diagonal);seen=set()
    for a in attachments:
        if not hasattr(a.data,'tensor'):raise ValueError('Partial-assembly component required')
        if seen.intersection(a.indices.tolist()):raise ValueError('Disjoint mass supports required')
        seen.update(a.indices.tolist())
        if not np.array_equal(a.indices[a.indices>=len(base.mass_diagonal)],np.arange(next_dof,a.next_dof)):
            raise ValueError('Nonconsecutive PZT DOFs')
        next_dof=a.next_dof
    diagonal=np.r_[base.mass_diagonal,np.zeros(next_dof-len(base.mass_diagonal))]
    for a,m in zip(attachments,masses):diagonal[a.indices]=m.diagonal
    # Conservative admission reserve for sparse LU analysis workspace. Check
    # exact retained bytes again after constructing each component/patch.
    from traxis.quadratic_pzt_sparse import sparse_bytes,factor_bytes
    host_sparse=sum(sparse_bytes(m.matrix)+factor_bytes(m.factor) for m in masses)
    unique={id(a.data):a.data for a in attachments}
    host_sparse+=sum(factor_bytes(d.electrical_lu)+d.tensor.inverse.nbytes+d.tensor.weight.nbytes for d in unique.values())
    reserve=8*host_sparse+3*diagonal.nbytes+64*2**20
    free=cp.cuda.runtime.memGetInfo()[0]
    if reserve>.7*free or reserve+base.screen['device_storage_screen_mib']*2**20>base.screen['max_device_mib']*2**20:
        raise MemoryError('Tensor/factor GPU reserve exceeds available allocation')
    screen=dict(base.screen,vector_dofs=len(diagonal),pzt_storage='matrixfree-tensor-cached-lu',
        cuda_graph_capture_supported=False,
        coupled_mass='one sparse scalar consistent mass per patch, 3 FP64 RHS, cached exact LU',
        body_reports=[a.report for a in attachments],mass_inverse_errors=[])
    bound=2/math.sqrt(4/base.stable_dt_s**2+max(d.body_eigenvalue_bound for d in unique.values()))
    screen['stable_dt_s']=bound
    with base.stiffness.storage.lock,cp.cuda.Device(base.stiffness.storage.device_id):
        if guard:guard.check(force=True)
        module=cp.RawModule(code=KERNELS,options=('--std=c++11','--fmad=false'))
        cache={};patches=[]
        for a,m in zip(attachments,masses):
            if id(a.data) not in cache:cache[id(a.data)]=DeviceComponent(a.data)
            patches.append(DevicePatch(a,m,cache[id(a.data)],module))
        payload=sum(p.bytes() for p in patches)
        payload+=sum(d.tensor.bytes()+d.electrical.bytes()+sum(v.nbytes for v in
            (d.inside,d.u,d.k,d.correction,d.charge,d.phi,d.dummy)) for d in cache.values())
        storage_bytes=base.screen['device_storage_screen_mib']*2**20+payload+3*diagonal.nbytes
        if payload+3*diagonal.nbytes>reserve:
            for p in patches:p.factor.close()
            for d in cache.values():d.electrical.close()
            raise MemoryError('Measured tensor/factor workspace exceeds admission reserve')
        screen.update(device_storage_screen_mib=storage_bytes/2**20,body_device_storage_mib=payload/2**20,
            retained_pzt_device_bytes=payload,shared_components=len(cache),cartesian_mass_copies=1,
            dielectric_analysis_calls=sum(d.electrical.analysis_calls for d in cache.values()),
            mass_analysis_calls=sum(p.factor.analysis_calls for p in patches))
        storage=SparseBodyStorage(base.stiffness.storage,diagonal,patches,screen,guard)
    def close():
        with base.stiffness.storage.lock,cp.cuda.Device(base.stiffness.storage.device_id):
            for p in patches:p.factor.close()
            for d in cache.values():d.electrical.close()
    return SimpleNamespace(stiffness=GPUElementOperator(storage,1.,0.),mass_diagonal=diagonal,
        stable_dt_s=bound,screen=screen,layout=base.layout,device=base.device,
        body_attachments=attachments,body_patches=[SimpleNamespace(indices=a.indices) for a in attachments],
        inverse_mass_inplace=storage.inverse_mass_inplace,mass_quadratic=storage.mass_quadratic,close=close)
