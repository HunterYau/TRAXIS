"""FP64 batched spectral element integration; CPU sparse assembly remains exact.

Same fields, geometry, quadrature and consistent mass as the CPU backend.
No changes to plate coupling or LTS time integration.
"""
import numpy as np


def element_batches(mesh, material, quadrature_order=None):
    import cupy as cp
    from .spectral_piezoelectric import basis
    p=mesh.order;nl=(p+1)**3;q=p+2 if quadrature_order is None else quadrature_order
    if q<p+1:raise ValueError('Consistent spectral mass requires sufficient Gauss integration')
    N,ref,w=[cp.asarray(v,dtype=cp.float64) for v in basis(p,q)]
    c,e,eps=[cp.asarray(v,dtype=cp.float64) for v in material.stress_charge]
    free,_=cp.cuda.runtime.memGetInfo()
    # Bound temporary batches conservatively; never densify global matrices.
    bytes_per_element=q**3*(6*3*nl*4+nl*8+256)*8
    batch=min(8,int(.1*free/bytes_per_element))
    if batch<1:raise MemoryError('Insufficient free GPU memory for spectral element batch')
    for first in range(0,len(mesh.hexahedra),batch):
        ids=mesh.hexahedra[first:first+batch]
        points=cp.asarray(mesh.points_m[ids],dtype=cp.float64)
        J=ref[None]@points[:,None];det=cp.linalg.det(J)
        if bool(cp.any(det<=0)) or float(cp.max(cp.linalg.cond(J)))>1e6:
            raise ValueError('Inverted or ill-conditioned hex')
        grad=cp.linalg.inv(J)@ref[None];weight=w[None]*det
        strain=cp.zeros((len(ids),q**3,6,3*nl),dtype=cp.float64)
        for i in range(3):strain[:,:,i,i*nl:(i+1)*nl]=grad[:,:,i]
        for row,i,j in [(3,1,2),(4,0,2),(5,0,1)]:
            strain[:,:,row,i*nl:(i+1)*nl]=grad[:,:,j]
            strain[:,:,row,j*nl:(j+1)*nl]=grad[:,:,i]
        def integrate(left,tensor,right):
            products=(left.swapaxes(-1,-2)@tensor)@right
            return cp.sum(products*weight[:,:,None,None],axis=1)
        K=integrate(strain,c,strain);B=integrate(strain,e.T,grad);D=integrate(grad,eps,grad)
        M=material.density_kg_m3*(N.T[None]@(weight[:,:,None]*N[None]))
        host=[cp.asnumpy(v) for v in (K,M,B,D,cp.sum(weight,axis=1))]
        for j in range(len(ids)):
            yield tuple(v[j] for v in host)


class GpuFactor:
    """Exact CPU SuperLU factorization, GPU FP64 triangular solves/products."""
    def __init__(self,matrix):
        import cupy as cp
        from cupyx.scipy import sparse
        from cupyx.scipy.sparse.linalg import SuperLU
        from scipy.sparse.linalg import splu
        self.cp=cp
        self.matrix=sparse.csr_matrix(matrix)
        self.factor=SuperLU(splu(matrix,permc_spec='MMD_AT_PLUS_A'))

    def solve(self,rhs):
        cp=self.cp
        rhs=cp.asarray(rhs,dtype=cp.complex128)
        value=self.factor.solve(rhs.real.copy())+1j*self.factor.solve(rhs.imag.copy())
        residual=float(cp.linalg.norm(self.matrix@value-rhs)/cp.linalg.norm(rhs))
        return cp.asnumpy(value),residual
