"""Float64 GLL tensor elasticity with implicit structured gather/scatter."""
from dataclasses import dataclass
import hashlib
import math
import threading
import numpy as np
from traxis.tensor_hex import line_rule,element_matrices
from traxis.gpu_elements import _Storage,GPUElementOperator


KERNEL=r'''
extern "C" __global__ void tensor_apply(
    const double* D,const double* w,const double* inverse,const double* x,double* y,
    int nx,int ny,int nz,int order,long long ns,double determinant,double lam,double mu,double scale,int ne) {
    int e=blockIdx.x,t=threadIdx.x;
    if(e>=ne) return;
    int q=order+1,n=q*q*q;
    int ex=e/(ny*nz),ey=(e/nz)%ny,ez=e%nz;
    __shared__ double u[3*LOCAL_NODES],flux[9*LOCAL_NODES];
    int i=t/(q*q),j=(t/q)%q,k=t%q;
    long long node=0;
    if(t<n) {
        node=((long long)(ex*order+i)*(ny*order+1)+ey*order+j)*(nz*order+1)+ez*order+k;
        for(int c=0;c<3;++c) u[c*n+t]=x[node+c*ns];
    }
    __syncthreads();
    if(t<n) {
        double g[9];
        for(int c=0;c<3;++c) {
            double dx=0.,dy=0.,dz=0.;
            for(int l=0;l<q;++l) {
                dx+=D[i*q+l]*u[c*n+(l*q+j)*q+k];
                dy+=D[j*q+l]*u[c*n+(i*q+l)*q+k];
                dz+=D[k*q+l]*u[c*n+(i*q+j)*q+l];
            }
            for(int b=0;b<3;++b)
                g[c*3+b]=dx*inverse[b]+dy*inverse[3+b]+dz*inverse[6+b];
        }
        double trace=g[0]+g[4]+g[8],volume=determinant*w[i]*w[j]*w[k];
        for(int c=0;c<3;++c) for(int a=0;a<3;++a) {
            double value=0.;
            for(int b=0;b<3;++b) {
                double stress=mu*(g[c*3+b]+g[b*3+c]);
                if(c==b) stress+=lam*trace;
                value+=stress*inverse[a*3+b];
            }
            flux[(c*3+a)*n+t]=volume*value;
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


@dataclass
class TensorLayout:
    plate: object
    cells: tuple
    order: int
    transform: np.ndarray
    axes_m: tuple
    screen: dict

    @property
    def jacobian(self):
        lengths=np.array([self.plate.width_m,self.plate.height_m,self.plate.thickness_m])/self.cells
        return self.transform@np.diag(lengths/2)

    def points(self):
        points=np.stack(np.meshgrid(*self.axes_m,indexing='ij'),axis=-1).reshape(-1,3)
        return points@self.transform.T

    def element_nodes(self,first,stop):
        if type(first) is not int or type(stop) is not int or not 0<=first<stop<=self.screen['elements'] or stop-first>512:
            raise ValueError('tensor gather needs a bounded valid element chunk')
        p=self.order;q=p+1;nx,ny,nz=self.cells
        e=np.arange(first,stop,dtype=np.int64)
        ex,ey,ez=e//(ny*nz),(e//nz)%ny,e%nz
        i,j,k=np.indices((q,q,q)).reshape(3,-1)
        return ((ex[:,None]*p+i)*(ny*p+1)+ey[:,None]*p+j)*(nz*p+1)+ez[:,None]*p+k


def make_layout(plate,cells,order,transform=None,*,max_dofs=12000000,max_device_mib=1536,max_host_mib=1024):
    nodes=line_rule(order)[0]
    if (len(cells)!=3 or any(type(n) is not int or not 1<=n<=512 for n in cells)
            or type(max_dofs) is not int or not 24<=max_dofs<=12000000
            or not math.isfinite(max_device_mib) or not 0<max_device_mib<=6144
            or not math.isfinite(max_host_mib) or not 0<max_host_mib<=4096):
        raise ValueError('invalid tensor layout counts or memory limits')
    n=math.prod(count*order+1 for count in cells)
    device=(16*3*n*8+4096)/1024**2
    host=(6*3*n*8+3*n*8+32*1024**2)/1024**2
    if 3*n>max_dofs or device>max_device_mib or host>max_host_mib:
        raise MemoryError('tensor layout exceeds declared coefficient/vector/memory screens')
    transform=np.eye(3) if transform is None else np.asarray(transform)
    if (transform.shape!=(3,3) or np.iscomplexobj(transform) or not np.isfinite(transform).all()
            or np.linalg.det(transform)<=0 or np.linalg.cond(transform)>1e4):
        raise ValueError('tensor layout needs a regular positive affine map')
    lengths=(plate.width_m,plate.height_m,plate.thickness_m)
    axes=tuple(np.r_[np.concatenate([(cell+(nodes[:-1]+1)/2)*length/count for cell in range(count)]),length]
               for count,length in zip(cells,lengths))
    screen=dict(order=order,cells=list(cells),scalar_dofs=n,vector_dofs=3*n,elements=math.prod(cells),
        local_scalar_dofs=(order+1)**3,device_storage_screen_mib=device,host_storage_screen_mib=host,
        max_device_mib=max_device_mib,max_host_mib=max_host_mib,stiffness_rule='gll',precision='float64',
        kernel_source_sha256=hashlib.sha256(KERNEL.encode()).hexdigest(),global_sparse_assembly=False,
        dense_element_matrices_stored_on_device=False,element_index_table_stored=False)
    screen.update(kernel_defines={'LOCAL_NODES':(order+1)**3},kernel_threads=128 if (order+1)**3<=128 else 256)
    return TensorLayout(plate,tuple(cells),order,np.array(transform,copy=True),axes,screen)


def mass_diagonal(layout,material):
    weights=line_rule(layout.order)[1];factors=[]
    lengths=(layout.plate.width_m,layout.plate.height_m,layout.plate.thickness_m)
    for count,length in zip(layout.cells,lengths):
        factor=np.zeros(count*layout.order+1)
        for cell in range(count):
            factor[cell*layout.order:cell*layout.order+layout.order+1]+=weights*length/(2*count)
        factors.append(factor)
    scalar=(factors[0][:,None,None]*factors[1][None,:,None]*factors[2][None,None,:]).ravel()
    scalar*=material.density_kg_m3*np.linalg.det(layout.transform)
    if np.any(scalar<=0) or not np.isfinite(scalar).all():
        raise ArithmeticError('structured tensor mass must be finite and positive')
    return np.tile(scalar,3)


def streamed_action(layout,material,values,guard=None,*,chunk_elements=64):
    values=np.asarray(values);ns=layout.screen['scalar_dofs'];q=layout.order+1
    if (values.shape!=(3*ns,) or np.iscomplexobj(values) or not np.isfinite(values).all()
            or type(chunk_elements) is not int or not 1<=chunk_elements<=512):
        raise ValueError('invalid streamed tensor state or chunk')
    derivative=line_rule(layout.order)[2];weights=line_rule(layout.order)[1]
    inverse=np.linalg.inv(layout.jacobian);det=float(np.linalg.det(layout.jacobian))
    volume=weights[:,None,None]*weights[None,:,None]*weights[None,None,:]*det
    lam,mu=material.lame;answer=np.zeros((3,ns));state=values.reshape(3,ns)
    for first in range(0,layout.screen['elements'],chunk_elements):
        if guard: guard.check()
        ids=layout.element_nodes(first,min(first+chunk_elements,layout.screen['elements']))
        u=state[:,ids].transpose(1,0,2).reshape(len(ids),3,q,q,q)
        derivatives=np.stack([np.einsum('il,bcljk->bcijk',derivative,u),
            np.einsum('jl,bcilk->bcijk',derivative,u),np.einsum('kl,bcijl->bcijk',derivative,u)],axis=2)
        gradient=np.einsum('bcixyz,ij->bcjxyz',derivatives,inverse)
        stress=mu*(gradient+gradient.swapaxes(1,2));trace=sum(gradient[:,i,i] for i in range(3))
        for i in range(3):stress[:,i,i]+=lam*trace
        flux=np.einsum('bcjxyz,ij->bcixyz',stress,inverse)*volume[None,None,None,:,:,:]
        result=(np.einsum('il,bcijk->bcljk',derivative,flux[:,:,0])+
                np.einsum('jl,bcijk->bcilk',derivative,flux[:,:,1])+
                np.einsum('kl,bcijk->bcijl',derivative,flux[:,:,2]))
        result=result.reshape(len(ids),3,-1)
        for c in range(3):np.add.at(answer[c],ids.ravel(),result[:,c].ravel())
    return answer.ravel()


class TensorStorage(_Storage):
    def __init__(self,cp,layout,material,device_id,guard):
        self.cp,self.layout,self.screen,self.device_id,self.guard=cp,layout,layout.screen,device_id,guard
        self.lock=threading.RLock();self.applications=self.vector_uploads=self.vector_downloads=0
        self.preconditioner_applications=0;self.ns=layout.screen['scalar_dofs'];self.ne=layout.screen['elements']
        _,weights,derivative,*_=line_rule(layout.order)
        self.derivative=cp.asarray(np.ascontiguousarray(derivative));self.weights=cp.asarray(weights)
        self.inverse=cp.asarray(np.ascontiguousarray(np.linalg.inv(layout.jacobian)))
        self.determinant=float(np.linalg.det(layout.jacobian));self.lam,self.mu=material.lame
        self.input=cp.empty(3*self.ns,dtype=cp.float64);self.output=cp.empty_like(self.input)
        self.kernel=cp.RawKernel(KERNEL,'tensor_apply',options=('--std=c++11',f'-DLOCAL_NODES={(layout.order+1)**3}'))
        self._kernel(self.input,self.output,0.,0)
        cp.cuda.get_current_stream().synchronize()

    def _kernel(self,values,result,scale,elements):
        nx,ny,nz=self.layout.cells
        self.kernel((max(1,elements),),(self.screen['kernel_threads'],),(self.derivative,self.weights,self.inverse,values,result,
            np.int32(nx),np.int32(ny),np.int32(nz),np.int32(self.layout.order),np.int64(self.ns),
            np.float64(self.determinant),np.float64(self.lam),np.float64(self.mu),np.float64(scale),np.int32(elements)))

    def _launch(self,values,result,stiffness_scale=0.,mass_scale=0.,*,inverse=False):
        if inverse or mass_scale!=0.:
            raise ValueError('tensor storage exposes GLL stiffness; use its positive mass diagonal separately')
        result.fill(0.);self._kernel(values,result,stiffness_scale,self.ne);self.applications+=1


@dataclass
class GPUTensorSystem:
    stiffness: object
    mass_diagonal: np.ndarray
    layout: TensorLayout
    stable_dt_s: float
    screen: dict
    device: dict


def build_gpu_tensor(layout,material,*,device_id=0,guard=None):
    if type(device_id) is not int or not 0<=device_id<=31:
        raise ValueError('invalid tensor CUDA device')
    if guard:guard.check(force=True)
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount()<=device_id:
        raise ValueError('requested tensor CUDA device is unavailable')
    k,_,mass=element_matrices(layout.order,layout.jacobian,material,'gll')
    local_mass=np.tile(mass,3);invsqrt=1/np.sqrt(local_mass)
    upper=float(np.max(invsqrt*(abs(k)@invsqrt)))
    if not math.isfinite(upper) or upper<=0:
        raise ArithmeticError('invalid local tensor stability bound')
    # K and M assemble as sums of the local quadratic forms. The largest
    # local generalized bound therefore also bounds the global Rayleigh quotient.
    stable=2/math.sqrt(upper)
    diagonal=mass_diagonal(layout,material)
    with cp.cuda.Device(device_id):
        properties=cp.cuda.runtime.getDeviceProperties(device_id);free,total=cp.cuda.runtime.memGetInfo()
        if properties['major']<6 or layout.screen['device_storage_screen_mib']*1024**2>.85*free:
            raise MemoryError('tensor float64 atomics or free-device-memory screen failed')
        storage=TensorStorage(cp,layout,material,device_id,guard)
        raw_name=properties['name'];name=raw_name.decode() if isinstance(raw_name,bytes) else str(raw_name)
        device=dict(name=name,compute_capability=[properties['major'],properties['minor']],
            total_bytes=int(total),free_bytes_before_allocation=int(free),cupy_version=cp.__version__,precision='float64')
    if guard:guard.check(force=True)
    screen={**layout.screen,'local_generalized_gershgorin_upper_s2':upper,'stable_dt_s':stable}
    storage.screen=screen
    return GPUTensorSystem(GPUElementOperator(storage,1.,0.),diagonal,layout,stable,screen,device)
