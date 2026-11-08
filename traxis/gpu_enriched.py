"""Float64 GPU elasticity from shared reference gradients and affine geometry."""

from dataclasses import dataclass
import hashlib
import math
import threading

import numpy as np

from traxis.enriched_factor import gradient_factor,local_stability_data
from traxis.enriched_tet import EDGES,FACES,NODES,WEIGHTS,build_topology
from traxis.gpu_elements import GPUElementOperator,_Storage
from traxis.highorder import _native_space,highorder_dofs
from traxis.matrixfree import MatrixFreeSettings


KERNEL = r'''
extern "C" __global__ void enriched_apply(
    const double* factor, const double* geometry, const int* dofs,
    const double* x, double* y, int ne, int ns, double lam, double mu, double scale) {
    int e=blockIdx.x, t=threadIdx.x;
    if (e>=ne) return;
    __shared__ double u[45], gradient[117], physical[117], stress[117];
    const double* g=geometry+(long long)e*10;
    const int* ids=dofs+(long long)e*15;
    if (t<45) u[t]=x[ids[t%15]+(t/15)*ns];
    __syncthreads();
    if (t<117) {
        int q=t/9, c=(t%9)/3, a=t%3;
        double sum=0.;
        for (int j=0;j<15;++j) sum+=factor[(q*3+a)*15+j]*u[c*15+j];
        gradient[t]=sum;
    }
    __syncthreads();
    if (t<117) {
        int q=t/9, c=(t%9)/3, b=t%3;
        double sum=0.;
        for (int a=0;a<3;++a) sum+=gradient[(q*3+c)*3+a]*g[a*3+b];
        physical[t]=sum;
    }
    __syncthreads();
    if (t<117) {
        int q=t/9, c=(t%9)/3, a=t%3;
        double trace=physical[q*9]+physical[q*9+4]+physical[q*9+8], sum=0.;
        for (int b=0;b<3;++b) {
            double value=mu*(physical[(q*3+c)*3+b]+physical[(q*3+b)*3+c]);
            if (c==b) value+=lam*trace;
            sum+=value*g[a*3+b];
        }
        stress[t]=sum;
    }
    __syncthreads();
    if (t<45) {
        int c=t/15, i=t%15;
        double sum=0.;
        for (int q=0;q<13;++q)
            for (int a=0;a<3;++a) sum+=factor[(q*3+a)*15+i]*stress[(q*3+c)*3+a];
        atomicAdd(y+ids[i]+c*ns,scale*g[9]*sum);
    }
}
'''


class FactorStorage(_Storage):
    def __init__(self,cp,screen,material,device_id,guard):
        self.cp,self.screen,self.device_id,self.guard = cp,screen,device_id,guard
        self.lock = threading.RLock()
        self.applications = self.preconditioner_applications = self.vector_uploads = self.vector_downloads = 0
        self.ne,self.nl,self.ns = screen['tetrahedra'],15,screen['scalar_dofs']
        self.indices = cp.empty((self.ne,15),dtype=cp.int32)
        self.geometry = cp.empty((self.ne,10),dtype=cp.float64)
        self.factor = cp.asarray(np.ascontiguousarray(gradient_factor()[0]))
        self.input = cp.empty(3*self.ns,dtype=cp.float64)
        self.output = cp.empty_like(self.input)
        self.lam,self.mu = material.lame
        self.kernel = cp.RawKernel(KERNEL,'enriched_apply',options=('--std=c++11',))
        self.kernel((1,),(128,),(self.factor,self.geometry,self.indices,self.input,self.output,
            np.int32(0),np.int32(self.ns),np.float64(self.lam),np.float64(self.mu),np.float64(0)))
        cp.cuda.get_current_stream().synchronize()

    def _launch(self,values,result,stiffness_scale=0.,mass_scale=0.,*,inverse=False):
        if inverse or mass_scale!=0.:
            raise ValueError('factorized elasticity exposes stiffness; use the positive mass diagonal separately')
        result.fill(0.)
        self.kernel((self.ne,),(128,),(self.factor,self.geometry,self.indices,values,result,
            np.int32(self.ne),np.int32(self.ns),np.float64(self.lam),np.float64(self.mu),np.float64(stiffness_scale)))
        self.applications += 1


@dataclass
class GPUEnrichedSystem:
    stiffness: GPUElementOperator
    mass_diagonal: np.ndarray
    stiffness_diagonal: np.ndarray
    native_to_topology: np.ndarray
    points_m: np.ndarray
    stable_dt_s: float
    mesh: object
    scalar_space: object
    space: object
    screen: dict
    device: dict


def build_gpu_enriched(mesh,material,settings=None,*,max_device_mib=1024,chunk_elements=128,
                       device_id=0,guard=None,progress=None):
    settings = settings or MatrixFreeSettings()
    if (not math.isfinite(max_device_mib) or not 0<max_device_mib<=6144
            or type(chunk_elements) is not int or not 1<=chunk_elements<=1024
            or type(device_id) is not int or not 0<=device_id<=31):
        raise ValueError('invalid enriched GPU device/storage/chunk limits')
    topology = build_topology(mesh,max_vertices=settings.max_vertices,max_tetrahedra=settings.max_tetrahedra,
        max_dofs=settings.max_dofs,max_workspace_mib=settings.max_topology_mib)
    screen = topology.screen.copy()
    n,ne = screen['scalar_dofs'],screen['tetrahedra']
    vector_mib = 16*3*n*8/1024**2
    device_mib = (ne*(15*4+10*8)+13*45*8)/1024**2+vector_mib
    host_chunk_mib = chunk_elements*45**2*8*8/1024**2
    if vector_mib>settings.max_vector_mib or device_mib>max_device_mib or host_chunk_mib>128:
        raise MemoryError('enriched GPU operator exceeds its declared vector/storage/workspace screen')
    factor,error = gradient_factor()
    screen.update(device_storage_screen_mib=device_mib,host_chunk_screen_mib=host_chunk_mib,
        reference_gradient_rank=13,gradient_factor_relative_error=error,
        kernel_source_sha256=hashlib.sha256(KERNEL.encode()).hexdigest(),global_sparse_assembly=False,
        local_dense_stiffness_stored=False)
    if guard:
        guard.check(force=True)
    import cupy as cp
    import ngsolve as ng
    if cp.cuda.runtime.getDeviceCount()<=device_id:
        raise ValueError('requested enriched CUDA device is unavailable')
    with cp.cuda.Device(device_id):
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        free,total = cp.cuda.runtime.memGetInfo()
        if properties['major']<6 or device_mib*1024**2>.85*free:
            raise MemoryError('enriched float64 atomic support or 85% free-VRAM screen failed')
        linear_screen = highorder_dofs(mesh,1,max_vertices=settings.max_vertices,max_tetrahedra=settings.max_tetrahedra,
            max_topology_mib=settings.max_topology_mib)
        native,_,_ = _native_space(mesh,1,linear_screen,guard)
        positions = np.array([tuple(v.point) for v in native.vertices])
        tolerance = 1e-12*np.ptp(mesh.points_m,axis=0).max()
        if not np.allclose(positions,mesh.points_m,rtol=0,atol=tolerance):
            raise ArithmeticError('native geometry vertex numbering changed')
        scalar = ng.H1LumpingFESpace(native,order=2)
        space = ng.FESpace([scalar,scalar,scalar])
        if scalar.ndof!=n or space.ndof!=3*n:
            raise ArithmeticError('native enriched coefficient counts changed')
        for c in range(3):
            if space.Range(c).start!=c*n or space.Range(c).stop!=(c+1)*n:
                raise ArithmeticError('native enriched component ordering changed')
        storage = FactorStorage(cp,screen,material,device_id,guard)
        permutation = np.full(n,-1,dtype=np.int64)
        coordinates = np.full((n,3),np.nan)
        mass,diagonal = np.zeros(n),np.zeros(3*n)
        owners = {tuple(sorted(t)):i for i,t in enumerate(mesh.tetrahedra)}
        indices = np.empty((chunk_elements,15),dtype=np.int32)
        geometry = np.empty((chunk_elements,10))
        start,count,upper,coordinate_error = 0,0,0.,0.
        def upload():
            nonlocal upper
            bounds,local_diagonal = local_stability_data(geometry[:count,:9].reshape(-1,3,3),material)
            upper = max(upper,float(bounds.max()))
            for component in range(3):
                np.add.at(diagonal,indices[:count]+component*n,
                    local_diagonal[:,component*15:(component+1)*15]*geometry[:count,9,None])
            storage.indices[start:start+count].set(indices[:count])
            storage.geometry[start:start+count].set(geometry[:count])
            cp.cuda.get_current_stream().synchronize()
            if progress:
                progress(dict(elements_ready=start+count,total_elements=ne))
        for element in native.Elements(ng.VOL):
            if guard:
                guard.check()
            vertices = [v.nr for v in element.vertices]
            owner = owners[tuple(sorted(vertices))]
            original = list(mesh.tetrahedra[owner])
            vertex_permutation = [original.index(v) for v in vertices]
            entity_order = vertex_permutation.copy()
            entity_order += [4+EDGES.index(tuple(sorted(vertex_permutation[i] for i in edge))) for edge in EDGES]
            entity_order += [10+FACES.index(tuple(sorted(vertex_permutation[i] for i in face))) for face in FACES]
            entity_order += [14]
            expected_ids = topology.element_nodes[owner,entity_order]
            eid = ng.ElementId(ng.VOL,element.nr)
            fe = scalar.GetFE(eid)
            values = np.array([fe.CalcShape(*p[:3]) for p in NODES])
            columns = values.argmax(axis=1)
            if values.shape!=(15,15) or len(np.unique(columns))!=15 or not np.allclose(values[:,columns],np.eye(15),rtol=0,atol=1e-12):
                raise ArithmeticError('native enriched functions are not the checked nodal permutation')
            scalar_ids = np.asarray(scalar.GetDofNrs(eid),dtype=np.int32)
            vector_ids = np.asarray(space.GetDofNrs(eid),dtype=np.int64)
            if (len(scalar_ids)!=15 or np.any(scalar_ids<0)
                    or not np.array_equal(vector_ids,np.concatenate([scalar_ids+c*n for c in range(3)]))):
                raise ArithmeticError('native enriched local/component map changed')
            ids = scalar_ids[columns]
            if np.any((permutation[ids]!=-1)&(permutation[ids]!=expected_ids)):
                raise ArithmeticError('native enriched map violates shared entity continuity')
            permutation[ids] = expected_ids
            corners = mesh.points_m[vertices]
            physical = NODES@corners
            coordinate_error = max(coordinate_error,float(np.max(abs(physical-topology.points_m[expected_ids]))))
            coordinates[ids] = physical
            jacobian = (corners[:3]-corners[3]).T
            determinant = abs(float(np.linalg.det(jacobian)))
            mass[ids] += material.density_kg_m3*determinant*WEIGHTS
            indices[count] = ids
            geometry[count,:9] = np.linalg.inv(jacobian).ravel()
            geometry[count,9] = determinant
            count += 1
            if count==chunk_elements:
                upload()
                start,count = start+count,0
        if count:
            upload()
            start += count
        if (start!=ne or not np.array_equal(np.sort(permutation),np.arange(n)) or coordinate_error>tolerance
                or not np.isfinite(coordinates).all() or np.any(mass<=0) or not np.isfinite(mass).all()
                or np.any(diagonal<0) or not np.isfinite(diagonal).all()):
            raise ArithmeticError('enriched coverage, physical map or positive diagonal check failed')
        screen.update(native_topology_permutation_verified=True,maximum_coordinate_error_m=coordinate_error,
            maximum_local_normalized_row_sum_bound=upper,
            stability_policy='maximum_element_lumped_mass_gershgorin',scalar_total_mass_kg=float(mass.sum()))
        device = dict(name=properties['name'].decode(),compute_capability=[properties['major'],properties['minor']],
            total_bytes=total,free_bytes_before_allocation=free,cupy_version=cp.__version__,precision='float64')
        if guard:
            guard.check(force=True)
        return GPUEnrichedSystem(GPUElementOperator(storage,1.,0.),np.tile(mass,3),diagonal,permutation,coordinates,
            2/math.sqrt(upper),native,scalar,space,screen,device)
