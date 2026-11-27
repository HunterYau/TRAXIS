"""Separate exact block-diagonal mass solve; reference operators/LTS untouched.

All disjoint scalar LU factors retain their entries and permutations. A single
SpSM analysis exposes independent components to the GPU at the same time. No
dense inverse, iterative solve, altered mass, or changed numerical recurrence.
"""
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from traxis.gpu_local_aluminum import build_system as reference_build_system

PACK=r'''
extern "C" __global__ void gather_mass(const double* x,const long long* ids,double* rhs,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)rhs[i]=x[ids[i]];}
extern "C" __global__ void scatter_mass(double* x,const long long* ids,const double* rhs,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[ids[i]]=rhs[i];}
extern "C" __global__ void divide_mass(double* x,const double* mass,long long n){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]/=mass[i];}
'''

def combine_factors(factors):
    """Factor the direct sum by retaining each already computed exact LU."""
    offsets=np.r_[0,np.cumsum([f.shape[0] for f in factors])]
    return SimpleNamespace(L=sparse.block_diag([f.L for f in factors],format='csc'),
        U=sparse.block_diag([f.U for f in factors],format='csc'),
        perm_r=np.concatenate([f.perm_r+o for f,o in zip(factors,offsets)]),
        perm_c=np.concatenate([f.perm_c+o for f,o in zip(factors,offsets)]),
        shape=(int(offsets[-1]),int(offsets[-1])))

class BatchedMass:
    def __init__(self,system,factors):
        import cupy as cp
        from traxis.gpu_cached_lu import CachedLU
        from traxis.gpu_native_cusparse import NativeCachedLU,NativeLibrary
        self.cp=cp;storage=system.stiffness.storage;self.diagonal=storage.diagonal
        patches=storage.device_patches
        if len(patches)!=len(factors):raise ValueError('Matching disjoint mass factors required')
        ids=[]
        for p in patches:
            pos=cp.asnumpy(p.pos);index=cp.asnumpy(p.ids)
            if pos.shape!=(p.q,3):raise ValueError('Expected three Cartesian mass RHS')
            ids.append(index[pos])
        packed=np.concatenate(ids,axis=0).T.ravel().astype(np.int64)
        if len(np.unique(packed))!=len(packed):raise ValueError('Mass supports overlap')
        # Retain reference factors for fallback; reject a duplicate payload that
        # would use the available GPU headroom before allocating it.
        factor_bytes=sum(sum(a.nbytes for a in (f.L.data,f.L.indices,f.L.indptr,f.U.data,f.U.indices,f.U.indptr)) for f in factors)
        reserve=4*factor_bytes+12*packed.nbytes+64*2**20
        if reserve>.4*cp.cuda.runtime.memGetInfo()[0]:raise MemoryError('Insufficient GPU headroom for exact batched factors plus fallback')
        combined=combine_factors(factors)
        self.factor=NativeCachedLU(CachedLU(combined,3),NativeLibrary())
        self.ids=cp.asarray(packed);self.n=len(packed)
        self.module=cp.RawModule(code=PACK,options=('--std=c++11','--fmad=false'))
        self.gather=self.module.get_function('gather_mass');self.scatter=self.module.get_function('scatter_mass')
        self.divide=self.module.get_function('divide_mass')
        self.report=dict(scalar_rows=combined.shape[0],independent_blocks=len(factors),
            lower_nnz=combined.L.nnz,upper_nnz=combined.U.nnz,extra_gpu_bytes=self.factor.bytes()+self.ids.nbytes,
            rhs_columns=3,mass='unchanged consistent and projected mass',triangular_solves_per_inverse=2,
            reference_triangular_solves_per_inverse=2*len(factors),dense_inverse=False)
    def apply(self,x):
        self.gather(((self.n+255)//256,),(256,),(x,self.ids,self.factor.rhs,np.int32(self.n)))
        self.divide(((len(x)+255)//256,),(256,),(x,self.diagonal,np.int64(len(x))))
        answer=self.factor.solve()
        self.scatter(((self.n+255)//256,),(256,),(x,self.ids,answer,np.int32(self.n)))

def build_system(base,mesh,attachments=(),masses=(),*,guard=None,native=False):
    if not native:raise ValueError('Batched adapter requires native CUDA Graph-compatible factors; use the reference build for fallback')
    system=reference_build_system(base,mesh,attachments,masses,guard=guard,native=native)
    try:
        batch=BatchedMass(system,[b.factor for b in mesh.mass_blocks]+[m.factor for m in masses])
    except BaseException:
        system.close();raise
    system.reference_inverse_mass_inplace=system.inverse_mass_inplace
    system.inverse_mass_inplace=batch.apply;system.batched_mass=batch
    system.native_factors.append(batch.factor)
    system.screen['batched_mass']=batch.report
    system.screen['device_storage_screen_mib']+=batch.report['extra_gpu_bytes']/2**20
    original_close=system.close
    def close():
        try:batch.factor.close()
        finally:original_close()
    system.close=close
    return system
