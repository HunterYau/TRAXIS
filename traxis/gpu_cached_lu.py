"""FP64 cuSPARSE triangular solves with persistent analysis and fixed buffers.

Uses CuPy's low-level NVIDIA bindings. Requires the pinned CuPy CUDA 12 runtime;
no CPU stepping fallback and no incomplete factorization. Plans are serialized
by their owning operator's lock, and must outlive asynchronous solves/graphs.
"""
import numpy as np

PERMUTE = r'''
extern "C" __global__ void permute(const double* x,const int* p,double* y,int n,int k) {
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t<n*k){int col=t/n,row=t%n;y[t]=x[p[row]+col*n];}
}
'''


class TriangularPlan:
    def __init__(self,matrix,rhs,out,lower):
        import cupy as cp
        from cupyx import cusparse as cs
        self.cp=cp;self.cs=cs;self.lib=cs._cusparse
        self.matrix=matrix;self.rhs=rhs;self.out=out;self.closed=False
        if matrix.dtype!=np.dtype('float64') or rhs.dtype!=matrix.dtype or out.dtype!=matrix.dtype:
            raise TypeError('FP64 triangular buffers required')
        if not rhs.flags.f_contiguous or not out.flags.f_contiguous:
            raise ValueError('Fixed Fortran contiguous buffers required')
        if not cs.check_availability('spsm'):raise RuntimeError('cuSPARSE SpSM is required')
        self.a=cs.SpMatDescriptor.create(matrix);self.b=cs.DnMatDescriptor.create(rhs);self.c=cs.DnMatDescriptor.create(out)
        lib=self.lib
        self.a.set_attribute(lib.CUSPARSE_SPMAT_FILL_MODE,lib.CUSPARSE_FILL_MODE_LOWER if lower else lib.CUSPARSE_FILL_MODE_UPPER)
        self.a.set_attribute(lib.CUSPARSE_SPMAT_DIAG_TYPE,lib.CUSPARSE_DIAG_TYPE_NON_UNIT)
        self.desc=lib.spSM_createDescr();self.alpha=np.array(1.,dtype=np.float64)
        self.dtype=cs._dtype.to_cuda_dtype(matrix.dtype)
        self.algo=lib.CUSPARSE_SPSM_ALG_DEFAULT;self.op=lib.CUSPARSE_OPERATION_NON_TRANSPOSE
        handle=cs._device.get_cusparse_handle()
        size=lib.spSM_bufferSize(*self.arguments(handle))
        self.workspace=cp.empty(size,dtype=cp.int8)
        lib.spSM_analysis(*self.arguments(handle),self.workspace.data.ptr)
        self.analysis_calls=1

    def arguments(self,handle):
        return (handle,self.op,self.op,self.alpha.ctypes.data,self.a.desc,self.b.desc,
                self.c.desc,self.dtype,self.algo,self.desc)

    def solve(self):
        if self.closed:raise RuntimeError('Triangular plan has been closed')
        self.out.fill(0.)
        self.lib.spSM_solve(*self.arguments(self.cs._device.get_cusparse_handle()),self.workspace.data.ptr)

    def close(self):
        if not self.closed:
            self.cp.cuda.get_current_stream().synchronize()
            self.lib.spSM_destroyDescr(self.desc);self.closed=True
            # CuPy matrix descriptors retain their own safe destructors.

    def __del__(self):
        if not getattr(self,'closed',True):
            try:self.close()
            except Exception:pass  # CUDA may already be torn down at interpreter exit.


class CachedLU:
    """Retain one exact scalar factor; handle all Cartesian RHS together."""
    cuda_graph_capture_supported = False  # Pinned CuPy cuSPARSE bindings reject capture.
    def __init__(self,factor,nrhs=1):
        import cupy as cp
        from cupyx.scipy import sparse
        if type(nrhs) is not int or nrhs<1:raise ValueError('Positive fixed RHS count required')
        if factor.L.dtype!=np.dtype('float64'):raise TypeError('FP64 LU required')
        self.cp=cp;self.n=factor.shape[0];self.nrhs=nrhs;self.solve_calls=0
        self.L=sparse.csr_matrix(factor.L.tocsr());self.U=sparse.csr_matrix(factor.U.tocsr())
        self.perm_r=cp.asarray(np.argsort(factor.perm_r).astype(np.int32))
        self.perm_c=cp.asarray(factor.perm_c.astype(np.int32))
        self.rhs=cp.zeros((self.n,nrhs),dtype=cp.float64,order='F')
        self.b=cp.zeros_like(self.rhs,order='F');self.y=cp.zeros_like(self.rhs,order='F')
        self.z=cp.zeros_like(self.rhs,order='F');self.result=cp.zeros_like(self.rhs,order='F')
        self.permute=cp.RawKernel(PERMUTE,'permute',options=('--std=c++11',))
        self.lower=TriangularPlan(self.L,self.b,self.y,True)
        self.upper=TriangularPlan(self.U,self.y,self.z,False)

    @property
    def analysis_calls(self):return self.lower.analysis_calls+self.upper.analysis_calls

    def solve(self):
        """Caller fills self.rhs; result remains valid until the next call."""
        args=(np.int32(self.n),np.int32(self.nrhs));grid=((self.n*self.nrhs+255)//256,)
        self.permute(grid,(256,),(self.rhs,self.perm_r,self.b,*args))
        self.lower.solve();self.upper.solve()
        self.permute(grid,(256,),(self.z,self.perm_c,self.result,*args))
        self.solve_calls+=1
        return self.result

    def close(self):
        self.lower.close();self.upper.close()

    def bytes(self):
        return sum(a.nbytes for a in (self.rhs,self.b,self.y,self.z,self.result,self.perm_r,self.perm_c,
            self.lower.workspace,self.upper.workspace))+sum(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes for a in (self.L,self.U))
