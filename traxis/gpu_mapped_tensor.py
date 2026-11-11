"""Float64 CUDA elasticity on the explicit, curved recess connectivity."""
from dataclasses import dataclass
import hashlib
import math
import threading

import numpy as np

from traxis.gpu_elements import _Storage, GPUElementOperator
from traxis.mapped_tensor import MappedTensorSystem
from traxis.tensor_hex import line_rule


KERNEL = r'''
extern "C" __global__ void mapped_tensor_apply(
    const double* D, const int* ids, const double* inverse, const double* volume,
    const double* x, double* y, int order, long long ns,
    double lam, double mu, double scale, int ne) {
    int e=blockIdx.x, t=threadIdx.x;
    if(e>=ne) return;
    int q=order+1, n=q*q*q;
    __shared__ double u[3*LOCAL_NODES], flux[9*LOCAL_NODES];
    int i=t/(q*q), j=(t/q)%q, k=t%q;
    long long node=0;
    if(t<n) {
        node=ids[(long long)e*n+t];
        for(int c=0;c<3;++c) u[c*n+t]=x[node+c*ns];
    }
    __syncthreads();
    if(t<n) {
        const double* J=inverse+((long long)e*n+t)*9;
        double g[9];
        for(int c=0;c<3;++c) {
            double dx=0., dy=0., dz=0.;
            for(int l=0;l<q;++l) {
                dx+=D[i*q+l]*u[c*n+(l*q+j)*q+k];
                dy+=D[j*q+l]*u[c*n+(i*q+l)*q+k];
                dz+=D[k*q+l]*u[c*n+(i*q+j)*q+l];
            }
            for(int b=0;b<3;++b)
                g[c*3+b]=dx*J[b]+dy*J[3+b]+dz*J[6+b];
        }
        double trace=g[0]+g[4]+g[8], weight=volume[(long long)e*n+t];
        for(int c=0;c<3;++c) for(int a=0;a<3;++a) {
            double value=0.;
            for(int b=0;b<3;++b) {
                double stress=mu*(g[c*3+b]+g[b*3+c]);
                if(c==b) stress+=lam*trace;
                value+=stress*J[a*3+b];
            }
            flux[(c*3+a)*n+t]=weight*value;
        }
    }
    __syncthreads();
    if(t<n) for(int c=0;c<3;++c) {
        double value=0.;
        for(int l=0;l<q;++l) {
            value+=D[l*q+i]*flux[(c*3)*n+(l*q+j)*q+k];
            value+=D[l*q+j]*flux[(c*3+1)*n+(i*q+l)*q+k];
            value+=D[l*q+k]*flux[(c*3+2)*n+(i*q+j)*q+l];
        }
        atomicAdd(y+node+c*ns,scale*value);
    }
}
'''


def storage_screen(system, *, max_device_mib=512, max_host_mib=512):
    """Check counts and host/device forecasts before importing CUDA."""
    if not isinstance(system, MappedTensorSystem):
        raise ValueError('mapped CUDA elasticity needs a checked CPU mapped system')
    if (not math.isfinite(max_device_mib) or not 0 < max_device_mib <= 6144
            or not math.isfinite(max_host_mib) or not 0 < max_host_mib <= 4096):
        raise ValueError('invalid mapped CUDA memory limits')
    mesh = system.mesh
    line_rule(mesh.order)
    ns, ne, q = len(mesh.points_m), len(mesh.element_nodes), mesh.order+1
    n = q**3
    # Geometry and connectivity plus sixteen global vectors for the resident
    # time integrator, including the two operator buffers. Samples are extra.
    device_bytes = ne*n*(10*8+4) + q*q*8 + 16*3*ns*8 + 4096
    host_bytes = system.screen['operator_storage_screen_bytes'] + mesh.allocation_upper_bytes
    if (3*ns > 12000000 or ne > 1000000 or device_bytes > max_device_mib*1024**2
            or host_bytes > max_host_mib*1024**2):
        raise MemoryError('mapped CUDA operator exceeds its preallocation screen')
    if (mesh.element_nodes.shape != (ne, n) or mesh.element_nodes.dtype.kind not in 'iu'
            or not ne or not ns or np.any(mesh.element_nodes < 0) or np.any(mesh.element_nodes >= ns)
            or system.inverse_jacobians.shape != (ne, n, 3, 3)
            or system.volume_weights_m3.shape != (ne, n) or system.mass_kg.shape != (3*ns,)
            or np.iscomplexobj(system.inverse_jacobians) or np.iscomplexobj(system.volume_weights_m3)
            or np.iscomplexobj(system.mass_kg) or not np.isfinite(system.inverse_jacobians).all()
            or not np.isfinite(system.volume_weights_m3).all() or not np.isfinite(system.mass_kg).all()
            or np.any(system.volume_weights_m3 <= 0) or np.any(system.mass_kg <= 0)
            or not math.isfinite(system.stable_dt_s) or system.stable_dt_s <= 0):
        raise ValueError('mapped CUDA geometry, connectivity or positive mass is invalid')
    return {**system.screen, 'scalar_dofs':ns, 'vector_dofs':3*ns, 'elements':ne, 'local_scalar_dofs':n,
        'device_storage_screen_mib':device_bytes/1024**2, 'host_storage_screen_mib':host_bytes/1024**2,
        'max_device_mib':max_device_mib, 'max_host_mib':max_host_mib,
        'precision':'float64', 'global_sparse_assembly':False,
        'dense_element_matrices_stored_on_device':False, 'element_index_table_stored':True,
        'inverse_jacobian_per_gll_point':True, 'stable_dt_s':system.stable_dt_s,
        'eigenvalue_upper_s2':system.eigenvalue_upper_s2,
        'kernel_source_sha256':hashlib.sha256(KERNEL.encode()).hexdigest(),
        'kernel_defines':{'LOCAL_NODES':n}, 'kernel_threads':128 if n <= 128 else 256,
        'kernel_shared_bytes':12*n*8}


class MappedTensorStorage(_Storage):
    def __init__(self, cp, system, screen, device_id, guard):
        self.cp, self.screen, self.device_id, self.guard = cp, screen, device_id, guard
        self.lock = threading.RLock()
        self.applications = self.preconditioner_applications = self.vector_uploads = self.vector_downloads = 0
        self.ns, self.ne, self.order = screen['scalar_dofs'], screen['elements'], system.mesh.order
        self.derivative = cp.asarray(np.ascontiguousarray(line_rule(self.order)[2]), dtype=cp.float64)
        self.indices = cp.asarray(system.mesh.element_nodes, dtype=cp.int32, order='C')
        self.inverse = cp.asarray(system.inverse_jacobians, dtype=cp.float64, order='C')
        self.volume = cp.asarray(system.volume_weights_m3, dtype=cp.float64, order='C')
        self.lam, self.mu = system.material.lame
        self.input = cp.empty(3*self.ns, dtype=cp.float64)
        self.output = cp.empty_like(self.input)
        self.kernel = cp.RawKernel(KERNEL, 'mapped_tensor_apply',
            options=('--std=c++11', f'-DLOCAL_NODES={screen["local_scalar_dofs"]}'))
        self._kernel(self.input, self.output, 0., 0)
        cp.cuda.get_current_stream().synchronize()

    def _kernel(self, values, result, scale, elements):
        self.kernel((max(1, elements),), (self.screen['kernel_threads'],),
            (self.derivative, self.indices, self.inverse, self.volume, values, result,
             np.int32(self.order), np.int64(self.ns), np.float64(self.lam), np.float64(self.mu),
             np.float64(scale), np.int32(elements)))

    def _launch(self, values, result, stiffness_scale=0., mass_scale=0., *, inverse=False):
        if inverse or mass_scale != 0.:
            raise ValueError('mapped CUDA storage exposes stiffness; use the positive mass diagonal separately')
        result.fill(0.)
        self._kernel(values, result, stiffness_scale, self.ne)
        self.applications += 1


@dataclass
class GPUMappedTensorSystem:
    stiffness: object
    mass_diagonal: np.ndarray
    cpu_system: MappedTensorSystem
    stable_dt_s: float
    screen: dict
    device: dict


def build_gpu_mapped(system, *, device_id=0, guard=None, max_device_mib=512, max_host_mib=512):
    if type(device_id) is not int or not 0 <= device_id <= 31:
        raise ValueError('invalid mapped CUDA device')
    screen = storage_screen(system, max_device_mib=max_device_mib, max_host_mib=max_host_mib)
    if guard:
        guard.check(force=True)
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() <= device_id:
        raise ValueError('requested mapped CUDA device is unavailable')
    with cp.cuda.Device(device_id):
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        free, total = cp.cuda.runtime.memGetInfo()
        if (properties['major'] < 6 or screen['device_storage_screen_mib']*1024**2 > .85*free
                or screen['kernel_shared_bytes'] > properties['sharedMemPerBlock']):
            raise MemoryError('mapped float64 atomics, shared-memory or free-device-memory screen failed')
        storage = MappedTensorStorage(cp, system, screen, device_id, guard)
        raw_name = properties['name']
        device = dict(name=raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name),
            compute_capability=[properties['major'], properties['minor']], total_bytes=int(total),
            free_bytes_before_allocation=int(free), cupy_version=cp.__version__, precision='float64')
    if guard:
        guard.check(force=True)
    return GPUMappedTensorSystem(GPUElementOperator(storage, 1., 0.), system.mass_kg, system,
                                system.stable_dt_s, screen, device)
