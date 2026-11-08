"""Optional float64 CUDA application of native H1 element operators.

Local stiffness and scalar consistent-mass matrices are streamed to the GPU.
No global sparse matrix is constructed. Public coefficients keep the checked
component-major H1 layout, including native edge/face orientations.
"""

from dataclasses import dataclass
import hashlib
import math
from numbers import Real
import threading

import numpy as np
from scipy.sparse.linalg import LinearOperator

from traxis.highorder import _native_space, highorder_dofs, highorder_forms
from traxis.matrixfree import MatrixFreeSettings


KERNEL = r'''
extern "C" __global__ void element_apply(
    const double* k, const double* m, const int* dofs,
    const double* x, double* y, int ne, int nl, int ns,
    double ks, double ms) {
    long long index = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    int nv = 3*nl;
    if (index >= (long long)ne*nv) return;
    int e = (int)(index/nv), row = (int)(index%nv);
    double value = 0.0;
    if (ks != 0.0) {
        double sum = 0.0;
        for (int c=0; c<3; ++c)
            for (int j=0; j<nl; ++j)
                sum += k[((long long)e*nv+row)*nv+c*nl+j]
                    *x[dofs[(long long)e*nl+j]+c*ns];
        value += ks*sum;
    }
    if (ms != 0.0) {
        double sum = 0.0;
        for (int j=0; j<nl; ++j)
            sum += m[((long long)e*nl+row%nl)*nl+j]
                *x[dofs[(long long)e*nl+j]+(row/nl)*ns];
        value += ms*sum;
    }
    atomicAdd(y+dofs[(long long)e*nl+row%nl]+(row/nl)*ns,value);
}
'''


SCHWARZ_KERNEL = r'''
extern "C" __global__ void schwarz_apply(
    const double* inverse, const double* weight, const int* dofs,
    const double* x, double* y, int ne, int nl, int ns) {
    long long index = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    int nv = 3*nl;
    if (index >= (long long)ne*nv) return;
    int e = (int)(index/nv), row = (int)(index%nv);
    double sum = 0.0;
    for (int c=0; c<3; ++c)
        for (int j=0; j<nl; ++j) {
            int scalar = dofs[(long long)e*nl+j];
            sum += inverse[((long long)e*nv+row)*nv+c*nl+j]*weight[scalar]
                *x[scalar+c*ns];
        }
    int scalar = dofs[(long long)e*nl+row%nl];
    atomicAdd(y+scalar+(row/nl)*ns,weight[scalar]*sum);
}
'''


def element_inverse(stiffness, mass, scale):
    """Cholesky inverse of M_e + scale*K_e, with a checked local identity."""
    if not isinstance(scale,Real) or not math.isfinite(scale) or scale < 0:
        raise ValueError('element inverse requires a finite nonnegative stiffness scale')
    k,m = np.asarray(stiffness),np.asarray(mass)
    if (k.ndim != 3 or m.ndim != 3 or k.shape[0] != m.shape[0] or m.shape[1] != m.shape[2]
            or k.shape[1:] != (3*m.shape[1],3*m.shape[1]) or not len(k)
            or np.iscomplexobj(k) or np.iscomplexobj(m) or not np.isfinite(k).all() or not np.isfinite(m).all()):
        raise ValueError('element inverse requires matching finite real local matrices')
    n = m.shape[1]
    a = np.array(k*scale,dtype=np.float64)
    for c in range(3):
        a[:,c*n:(c+1)*n,c*n:(c+1)*n] += m
    skew = np.linalg.norm(a-a.swapaxes(-1,-2),axis=(-2,-1))
    if np.any(skew > 1e-12*np.linalg.norm(a,axis=(-2,-1))):
        raise ArithmeticError('local effective matrix is not symmetric')
    a = (a+a.swapaxes(-1,-2))*.5
    factor = np.linalg.cholesky(a)
    inverse_factor = np.linalg.solve(factor,np.broadcast_to(np.eye(3*n),factor.shape))
    inverse = inverse_factor.swapaxes(-1,-2)@inverse_factor
    error = float(np.max(np.linalg.norm(a@inverse-np.eye(3*n),axis=(-2,-1))/math.sqrt(3*n)))
    if not np.isfinite(inverse).all() or not math.isfinite(error) or error > 1e-9:
        raise ArithmeticError('local effective inverse failed its identity check')
    return inverse,error


def storage_screen(mesh, settings, *, max_device_mib=3072, chunk_elements=128, device_id=0,
                   preconditioner_scale=None):
    if (not math.isfinite(max_device_mib) or not 0 < max_device_mib <= 6144
            or type(chunk_elements) is not int or not 1 <= chunk_elements <= 1024
            or type(device_id) is not int or not 0 <= device_id <= 31):
        raise ValueError('invalid GPU element storage/chunk/device limits')
    if preconditioner_scale is not None and (not isinstance(preconditioner_scale,Real)
            or not math.isfinite(preconditioner_scale) or preconditioner_scale < 0):
        raise ValueError('invalid GPU preconditioner stiffness scale')
    screen = highorder_dofs(mesh,settings.order,max_vertices=settings.max_vertices,
        max_tetrahedra=settings.max_tetrahedra,max_topology_mib=settings.max_topology_mib)
    local = math.comb(settings.order+3,3)
    element_bytes = screen['tetrahedra']*((3*local)**2+local**2)*8
    index_bytes = screen['tetrahedra']*local*4
    vector_bytes = screen['vector_dofs']*8*16
    inverse_bytes = 0 if preconditioner_scale is None else screen['tetrahedra']*(3*local)**2*8+screen['scalar_dofs']*8
    device_mib = (element_bytes+index_bytes+vector_bytes+inverse_bytes)/1024**2
    host_chunk_mib = chunk_elements*((3*local)**2+local**2)*8/1024**2
    if preconditioner_scale is not None:
        host_chunk_mib += chunk_elements*6*(3*local)**2*8/1024**2
    if (screen['vector_dofs'] > settings.max_dofs or vector_bytes/1024**2 > settings.max_vector_mib
            or device_mib > max_device_mib or host_chunk_mib > 128):
        raise MemoryError('GPU element/vector storage exceeds its declared screen')
    screen.update(local_scalar_dofs=local,device_storage_screen_mib=device_mib,
        element_matrices_mib=element_bytes/1024**2,host_chunk_screen_mib=host_chunk_mib,
        preconditioner_storage_mib=inverse_bytes/1024**2,preconditioner_scale=preconditioner_scale,
        kernel_source_sha256=hashlib.sha256(KERNEL.encode()).hexdigest(),
        schwarz_kernel_source_sha256=hashlib.sha256(SCHWARZ_KERNEL.encode()).hexdigest())
    return screen


class _Storage:
    def __init__(self, cp, screen, device_id, guard):
        self.cp,self.screen,self.device_id,self.guard = cp,screen,device_id,guard
        self.lock = threading.RLock()
        self.applications = 0
        self.preconditioner_applications = 0
        self.vector_uploads = self.vector_downloads = 0
        self.ne,self.nl,self.ns = screen['tetrahedra'],screen['local_scalar_dofs'],screen['scalar_dofs']
        self.stiffness = cp.empty((self.ne,3*self.nl,3*self.nl),dtype=cp.float64)
        self.mass = cp.empty((self.ne,self.nl,self.nl),dtype=cp.float64)
        self.indices = cp.empty((self.ne,self.nl),dtype=cp.int32)
        self.input = cp.empty(3*self.ns,dtype=cp.float64)
        self.output = cp.empty_like(self.input)
        self.kernel = cp.RawKernel(KERNEL,'element_apply',options=('--std=c++11',))
        # Compile/initialize without reading unfilled element arrays.
        self.kernel((1,),(1,),(self.stiffness,self.mass,self.indices,self.input,self.output,
            np.int32(0),np.int32(self.nl),np.int32(self.ns),np.float64(0),np.float64(0)))
        cp.cuda.get_current_stream().synchronize()
        self.inverse = self.weight = self.schwarz_kernel = None
        if screen['preconditioner_scale'] is not None:
            self.inverse = cp.empty_like(self.stiffness)
            self.weight = cp.empty(self.ns,dtype=cp.float64)
            self.schwarz_kernel = cp.RawKernel(SCHWARZ_KERNEL,'schwarz_apply',options=('--std=c++11',))
            self.schwarz_kernel((1,),(1,),(self.inverse,self.weight,self.indices,self.input,self.output,
                np.int32(0),np.int32(self.nl),np.int32(self.ns)))
            cp.cuda.get_current_stream().synchronize()

    def _launch(self, values, result, stiffness_scale=0., mass_scale=0., *, inverse=False):
        """Apply to device buffers; caller owns the storage lock/device context."""
        result.fill(0.)
        count = self.ne*3*self.nl
        if inverse:
            if self.inverse is None:
                raise ValueError('GPU element preconditioner was not constructed')
            self.schwarz_kernel(((count+127)//128,),(128,),
                (self.inverse,self.weight,self.indices,values,result,np.int32(self.ne),
                 np.int32(self.nl),np.int32(self.ns)))
            self.preconditioner_applications += 1
        else:
            self.kernel(((count+127)//128,),(128,),
                (self.stiffness,self.mass,self.indices,values,result,np.int32(self.ne),np.int32(self.nl),
                 np.int32(self.ns),np.float64(stiffness_scale),np.float64(mass_scale)))
            self.applications += 1

    def apply(self, values, stiffness_scale=0., mass_scale=0., *, inverse=False):
        values = np.asarray(values)
        if (np.iscomplexobj(values) or values.shape not in ((3*self.ns,),(3*self.ns,1))
                or not np.isfinite(values).all()):
            raise ValueError('GPU elasticity needs a finite real coefficient vector')
        with self.lock,self.cp.cuda.Device(self.device_id):
            if self.guard:
                self.guard.check()
            self.input.set(np.ascontiguousarray(values.reshape(-1),dtype=np.float64))
            self.vector_uploads += 1
            self._launch(self.input,self.output,stiffness_scale,mass_scale,inverse=inverse)
            answer = self.cp.asnumpy(self.output)
            self.vector_downloads += 1
            if self.guard:
                self.guard.check()
        if not np.isfinite(answer).all():
            raise ArithmeticError('GPU element application produced nonfinite coefficients')
        return answer


class GPUElementOperator(LinearOperator):
    def __init__(self, storage, stiffness_scale, mass_scale):
        self.storage = storage
        self.stiffness_scale,self.mass_scale = float(stiffness_scale),float(mass_scale)
        if not all(math.isfinite(v) for v in (self.stiffness_scale,self.mass_scale)):
            raise ValueError('GPU operator scales must be finite')
        self.applications = 0
        super().__init__(dtype=np.dtype('float64'),shape=(3*storage.ns,3*storage.ns))

    def _matvec(self, values):
        answer = self.storage.apply(values,self.stiffness_scale,self.mass_scale)
        self.applications += 1
        return answer

    def _rmatvec(self, values):
        return self._matvec(values)

    def __add__(self, other):
        if isinstance(other,GPUElementOperator) and other.storage is self.storage:
            return GPUElementOperator(self.storage,self.stiffness_scale+other.stiffness_scale,
                self.mass_scale+other.mass_scale)
        return super().__add__(other)

    def __mul__(self, other):
        if isinstance(other,Real):
            return GPUElementOperator(self.storage,self.stiffness_scale*other,self.mass_scale*other)
        return super().__mul__(other)

    def __rmul__(self, other):
        if isinstance(other,Real):
            return self.__mul__(other)
        return super().__rmul__(other)


class GPUElementPreconditioner(LinearOperator):
    """Symmetric multiplicity-weighted element additive Schwarz inverse.

    B = D sum_e R_e.T A_e^-1 R_e D. Positive local mass makes every A_e SPD;
    complete coefficient coverage and positive D make B SPD as well.
    """
    def __init__(self, storage):
        self.storage = storage
        self.applications = 0
        super().__init__(dtype=np.dtype('float64'),shape=(3*storage.ns,3*storage.ns))

    def _matvec(self, values):
        answer = self.storage.apply(values,inverse=True)
        self.applications += 1
        return answer

    def _rmatvec(self, values):
        return self._matvec(values)


@dataclass
class GPUElementSystem:
    stiffness: GPUElementOperator
    mass: GPUElementOperator
    mass_diagonal: np.ndarray
    stiffness_diagonal: np.ndarray
    mesh: object
    scalar_space: object
    space: object
    screen: dict
    device: dict
    preconditioner: object = None


def build_gpu_elements(mesh, material, settings=None, *, max_device_mib=3072, chunk_elements=128,
                       device_id=0, guard=None, progress=None, preconditioner_scale=None):
    settings = settings or MatrixFreeSettings()
    screen = storage_screen(mesh,settings,max_device_mib=max_device_mib,chunk_elements=chunk_elements,
        device_id=device_id,preconditioner_scale=preconditioner_scale)
    if guard:
        guard.check(force=True)
    import cupy as cp
    import ngsolve as ng
    if cp.cuda.runtime.getDeviceCount() <= device_id:
        raise ValueError('requested CUDA device is unavailable')
    with cp.cuda.Device(device_id):
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        if properties['major'] < 6:
            raise ValueError('float64 element assembly requires native double atomic addition')
        free_bytes,total_bytes = cp.cuda.runtime.memGetInfo()
        if screen['device_storage_screen_mib']*1024**2 > .85*free_bytes:
            raise MemoryError('GPU element screen exceeds 85% of currently available device memory')
        native_mesh,scalar,space = _native_space(mesh,settings.order,screen,guard)
        stiffness,_ = highorder_forms(space,material,matrix_free=True)
        u,v = scalar.TnT()
        mass = ng.BilinearForm(scalar,nonassemble=True)
        mass += material.density_kg_m3*u*v*ng.dx(bonus_intorder=2)
        storage = _Storage(cp,screen,device_id,guard)
        nl,nv = screen['local_scalar_dofs'],3*screen['local_scalar_dofs']
        k_buffer = np.empty((chunk_elements,nv,nv))
        m_buffer = np.empty((chunk_elements,nl,nl))
        indices = np.empty((chunk_elements,nl),dtype=np.int32)
        diagonal_k,diagonal_m = np.zeros(space.ndof),np.zeros(scalar.ndof)
        coverage = np.zeros(scalar.ndof,dtype=np.int32) if preconditioner_scale is not None else None
        inverse_error = 0.
        start,count = 0,0
        def upload():
            nonlocal inverse_error
            if preconditioner_scale is not None:
                inverse,error = element_inverse(k_buffer[:count],m_buffer[:count],preconditioner_scale)
                inverse_error = max(inverse_error,error)
                storage.inverse[start:start+count].set(inverse)
            storage.stiffness[start:start+count].set(k_buffer[:count])
            storage.mass[start:start+count].set(m_buffer[:count])
            storage.indices[start:start+count].set(indices[:count])
            cp.cuda.get_current_stream().synchronize()
            if progress:
                progress({'elements_uploaded':start+count,'total_elements':screen['tetrahedra']})
        for element in native_mesh.Elements(ng.VOL):
            if guard:
                guard.check()
            eid = ng.ElementId(ng.VOL,element.nr)
            scalar_dofs = np.asarray(scalar.GetDofNrs(eid),dtype=np.int32)
            vector_dofs = np.asarray(space.GetDofNrs(eid),dtype=np.int32)
            expected = np.concatenate([scalar_dofs+c*scalar.ndof for c in range(3)])
            if (len(scalar_dofs) != nl or np.any(scalar_dofs < 0) or len(np.unique(scalar_dofs)) != nl
                    or not np.array_equal(vector_dofs,expected)):
                raise ValueError('native element coefficient layout changed')
            transformation = native_mesh.GetTrafo(eid)
            k = sum(np.asarray(i.CalcElementMatrix(space.GetFE(eid),transformation)) for i in stiffness.integrators)
            m = sum(np.asarray(i.CalcElementMatrix(scalar.GetFE(eid),transformation)) for i in mass.integrators)
            if k.shape != (nv,nv) or m.shape != (nl,nl) or not np.isfinite(k).all() or not np.isfinite(m).all():
                raise ArithmeticError('invalid native element stiffness/mass')
            k_buffer[count],m_buffer[count],indices[count] = k,m,scalar_dofs
            np.add.at(diagonal_k,vector_dofs,np.diag(k))
            np.add.at(diagonal_m,scalar_dofs,np.diag(m))
            if coverage is not None:
                np.add.at(coverage,scalar_dofs,1)
            count += 1
            if count == chunk_elements:
                upload()
                start,count = start+count,0
        if count:
            upload()
            start += count
        if start != screen['tetrahedra'] or np.any(diagonal_m <= 0) or np.any(diagonal_k < 0):
            raise ArithmeticError('GPU element count or positive diagonal check failed')
        preconditioner = None
        if coverage is not None:
            if np.any(coverage <= 0):
                raise ArithmeticError('GPU preconditioner has an uncovered coefficient')
            storage.weight.set(1./coverage)
            cp.cuda.get_current_stream().synchronize()
            screen.update(maximum_local_inverse_identity_error=inverse_error,
                minimum_coefficient_coverage=int(coverage.min()),maximum_coefficient_coverage=int(coverage.max()))
            preconditioner = GPUElementPreconditioner(storage)
        if guard:
            guard.check(force=True)
        device = {'name':properties['name'].decode(),'compute_capability':[properties['major'],properties['minor']],
            'total_bytes':total_bytes,'free_bytes_before_allocation':free_bytes,'cupy_version':cp.__version__,
            'cuda_runtime_version':cp.cuda.runtime.runtimeGetVersion(),'cuda_driver_version':cp.cuda.runtime.driverGetVersion(),
            'precision':'float64','global_sparse_assembly':False}
        return GPUElementSystem(GPUElementOperator(storage,1.,0.),GPUElementOperator(storage,0.,1.),
            np.tile(diagonal_m,3),diagonal_k,native_mesh,scalar,space,screen,device,preconditioner)
