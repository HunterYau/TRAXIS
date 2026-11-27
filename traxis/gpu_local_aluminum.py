"""Separate GPU mesh adapter; original tensor/PZT/LTS implementations retained."""
from types import SimpleNamespace,FunctionType
import math,threading
import numpy as np
from scipy import sparse
from traxis.gpu_tensor import KERNEL,TensorStorage
from traxis.gpu_elements import _Storage,GPUElementOperator
from traxis.local_aluminum import Block
from traxis.gpu_local_time import selected_elements,LocalActions
from traxis.gpu_lts_optimized import ReusableActions,VECTOR_KERNELS
from traxis.gpu_spectral_matrixfree import csr_arrays,KERNELS,DeviceComponent,DevicePatch
from traxis.gpu_quadratic_pzt_sparse import SparseBodyStorage
from traxis.gpu_cached_lu import CachedLU
from traxis.tensor_hex import line_rule

CSR=r'''
extern "C" __global__ void csr3(const int* ptr,const int* col,const double* val,
 const double* x,double* y,int rows,int cols){int t=blockIdx.x*blockDim.x+threadIdx.x;
 if(t<3*rows){int c=t/rows,r=t%rows;double s=0.;for(int j=ptr[r];j<ptr[r+1];++j)s+=val[j]*x[c*cols+col[j]];y[t]=s;}}
'''


class DeviceBlock:
    def __init__(self,block,T,support,ns,material):
        import cupy as cp
        self.cp=cp;self.block=block;self.n=block.n;self.q=T.shape[1]
        self.ids=cp.asarray(np.concatenate([support+c*ns for c in range(3)]))
        self.t=csr_arrays(T);self.tt=csr_arrays(T.T)
        self.input=cp.empty(3*self.n);self.output=cp.empty_like(self.input);self.result=cp.empty(3*self.q)
        self.derivative=cp.asarray(line_rule(block.order)[2]);self.weights=cp.asarray(line_rule(block.order)[1])
        self.indices=cp.asarray(block.elements,dtype=cp.int32)
        inv=np.array([np.diag(v) for v in block.inverse]);self.inverse=cp.asarray(inv)
        self.det=cp.asarray(block.determinant);self.lam,self.mu=material.lame
        code=KERNEL.replace('tensor_apply(', 'indexed_tensor_apply(').replace('int ne) {','int ne,const int* indices,const double* determinants) {')
        code=code.replace('int ex=e/(ny*nz),ey=(e/nz)%ny,ez=e%nz;','const double* gi=inverse+(long long)e*9;')
        code=code.replace('node=((long long)(ex*order+i)*(ny*order+1)+ey*order+j)*(nz*order+1)+ez*order+k;', 'node=indices[(long long)e*n+t];')
        code=code.replace('inverse[b]','gi[b]').replace('inverse[3+b]','gi[3+b]').replace('inverse[6+b]','gi[6+b]').replace('inverse[a*3+b]','gi[a*3+b]')
        code=code.replace('volume=determinant*w[i]*w[j]*w[k]','volume=determinants[e]*w[i]*w[j]*w[k]')
        self.kernel=cp.RawKernel(code,'indexed_tensor_apply',options=('--std=c++11',f'-DLOCAL_NODES={(block.order+1)**3}'))
        self.csr=cp.RawKernel(CSR,'csr3',options=('--std=c++11',))
        self.apply(cp.zeros(3*ns));cp.cuda.get_current_stream().synchronize()
    def apply(self,x):
        self.csr(((3*self.n+255)//256,),(256,),(*self.t,x[self.ids],self.input,np.int32(self.n),np.int32(self.q)))
        self.output.fill(0.);b=self.block
        self.kernel((len(b.elements),),(128 if (b.order+1)**3<=128 else 256,),
            (self.derivative,self.weights,self.inverse,self.input,self.output,np.int32(1),np.int32(1),np.int32(1),np.int32(b.order),
             np.int64(b.n),np.float64(1.),np.float64(self.lam),np.float64(self.mu),np.float64(1.),np.int32(len(b.elements)),self.indices,self.det))
        self.csr(((3*self.q+255)//256,),(256,),(*self.tt,self.output,self.result,np.int32(self.q),np.int32(self.n)))
        return self.result


class HostStorage(_Storage):
    def __init__(self,base,mesh):
        import cupy as cp
        self.cp=cp;self.device_id=base.device_id;self.lock=base.lock;self.guard=base.guard
        self.base=base;self.mesh=mesh;self.ns=mesh.n;self.screen=dict(base.screen,scalar_dofs=mesh.n,vector_dofs=3*mesh.n)
        self.applications=self.vector_uploads=self.vector_downloads=self.preconditioner_applications=0
        self.input=cp.empty(3*self.ns);self.output=cp.empty_like(self.input)
        self.expanded=cp.empty(3*mesh.old_n);self.coarse_out=cp.empty_like(self.expanded);self.keep=cp.asarray(mesh.keep)
        self.blocks=[]
        for q in mesh.patches:
            self.blocks.append((1.,DeviceBlock(q.block,q.t,q.support,mesh.n,mesh.material)))
            edges=tuple(e[(e>=lo-1e-12)&(e<=hi+1e-12)] for e,(lo,hi) in zip(mesh.coarse_knots[:2],q.bounds))
            block=Block((*edges,mesh.coarse_knots[2]),mesh.base.order)
            self.blocks.append((-1.,DeviceBlock(block,q.ct,q.support,mesh.n,mesh.material)))
        code=KERNEL.replace('tensor_apply(','tensor_local_apply(').replace('int ne) {','int ne,const int* selected) {').replace(
            'int q=order+1,n=q*q*q;','e=selected[e];\n int q=order+1,n=q*q*q;')
        self.selected_kernel=cp.RawKernel(code,'tensor_local_apply',options=('--std=c++11',f'-DLOCAL_NODES={(mesh.base.order+1)**3}'))
        self.selected=None
    def select(self,fine):
        coarse=np.zeros((3,self.mesh.old_n),bool);coarse[:,self.mesh.keep]=fine[:3*self.ns].reshape(3,self.ns)[:,:len(self.mesh.keep)]
        ids=selected_elements(self.mesh.base,coarse.ravel());self.selected=self.cp.asarray(ids);return ids
    def launch(self,values,result,local=False):
        m=self.mesh;b=self.base;self.expanded.fill(0.)
        self.expanded.reshape(3,m.old_n)[:,self.keep]=values.reshape(3,m.n)[:,:len(m.keep)]
        if not local:b._launch(self.expanded,self.coarse_out,1.,0.)
        else:
            self.coarse_out.fill(0.);nx,ny,nz=b.layout.cells
            if self.selected is None:raise RuntimeError('Select fine elements before local actions')
            if len(self.selected):self.selected_kernel((len(self.selected),),(b.screen['kernel_threads'],),
                (b.derivative,b.weights,b.inverse,self.expanded,self.coarse_out,np.int32(nx),np.int32(ny),np.int32(nz),
                 np.int32(b.layout.order),np.int64(b.ns),np.float64(b.determinant),np.float64(b.lam),np.float64(b.mu),
                 np.float64(1.),np.int32(len(self.selected)),self.selected))
        result.fill(0.);result.reshape(3,m.n)[:,:len(m.keep)]=self.coarse_out.reshape(3,m.old_n)[:,self.keep]
        for sign,block in self.blocks:result[block.ids]+=sign*block.apply(values)
    def _launch(self,values,result,stiffness_scale=0.,mass_scale=0.,*,inverse=False):
        if inverse or mass_scale:raise ValueError('Use exact constrained mass actions')
        self.launch(values,result);result*=stiffness_scale;self.applications+=1


class BoundaryMass:
    def __init__(self,b,ns,module):
        import cupy as cp
        self.cp=cp;self.ids=cp.asarray(np.concatenate([b.ids+c*ns for c in range(3)]));self.q=len(b.ids)
        self.pos=cp.asarray(np.arange(3*self.q).reshape(3,self.q).T,dtype=cp.int32,order='C')
        self.mass=csr_arrays(b.matrix);self.factor=CachedLU(b.factor,3);self.module=module
        self.mass_out=cp.empty(3*self.q);self.inverse_out=cp.empty_like(self.mass_out);self.zero=cp.zeros_like(self.mass_out)
    def stiffness(self,x):return self.zero
    def mass_action(self,x):
        self.module.get_function('mass')(((3*self.q+255)//256,),(256,),(*self.mass,self.pos,x,self.mass_out,np.int32(self.q)))
        return self.mass_out
    def inverse(self,x):
        self.module.get_function('pack')(((3*self.q+255)//256,),(256,),(self.pos,x,self.factor.rhs,np.int32(self.q)))
        value=self.factor.solve()
        self.module.get_function('unpack')(((3*self.q+255)//256,),(256,),(self.pos,value,self.inverse_out,np.int32(self.q)))
        return self.inverse_out


def build_system(base,mesh,attachments=(),masses=(),*,guard=None,native=False):
    import cupy as cp
    reserve=(18*(3*mesh.n)+6*(3*mesh.old_n))*8
    if reserve>.7*cp.cuda.runtime.memGetInfo()[0]:raise MemoryError('Insufficient GPU headroom for local mesh and LTS buffers')
    host=HostStorage(base.stiffness.storage,mesh);module=cp.RawModule(code=KERNELS,options=('--std=c++11','--fmad=false'))
    patches=[BoundaryMass(b,mesh.n,module) for b in mesh.mass_blocks]
    diagonal=np.tile(mesh.diagonal,3);n=len(diagonal);seen=set()
    for b in mesh.mass_blocks:seen.update(np.concatenate([b.ids+c*mesh.n for c in range(3)]).tolist())
    components={}
    for a,m in zip(attachments,masses):
        if seen.intersection(a.indices.tolist()):raise ValueError('Mass supports overlap')
        seen.update(a.indices.tolist())
        if a.next_dof>n:diagonal=np.r_[diagonal,np.zeros(a.next_dof-n)];n=a.next_dof
        diagonal[a.indices]=m.diagonal
        if id(a.data) not in components:components[id(a.data)]=DeviceComponent(a.data)
        patches.append(DevicePatch(a,m,components[id(a.data)],module))
    factors=[]
    if native:
        from traxis.gpu_native_cusparse import NativeLibrary,NativeCachedLU
        lib=NativeLibrary()
        for p in patches:p.factor=NativeCachedLU(p.factor,lib);factors.append(p.factor)
        for d in components.values():d.electrical=NativeCachedLU(d.electrical,lib);factors.append(d.electrical)
    screen=dict(base.screen,scalar_dofs=mesh.n,vector_dofs=n,local_aluminum_refinement=mesh.report,
        pzt_storage='local-aluminum-tensor-matrixfree',cuda_graph_capture_supported=bool(native),
        device_storage_screen_mib=base.screen['device_storage_screen_mib']+reserve/2**20,
        stability_qualification='New local mesh: previous uniform-grid stability limits do not certify it')
    storage=SparseBodyStorage(host,diagonal,patches,screen,guard)
    # A conservative bound for the smaller host cells, not a certified LTS screen.
    from traxis.tensor_hex import element_matrices
    upper=4/base.stable_dt_s**2
    geometries={tuple(np.round(v,10)) for q in mesh.patches for v in q.block.inverse}
    for inverse in geometries:
        k,_,mass=element_matrices(mesh.base.order,np.diag(1/np.array(inverse)),mesh.material,'gll')
        root=1/np.sqrt(np.tile(mass,3));upper=max(upper,float(np.max(root*(abs(k)@root))))
    stable=2/math.sqrt(upper*(1+1e-8))
    if attachments:stable=2/math.sqrt(4/stable**2+max(a.data.body_eigenvalue_bound for a in attachments))
    screen['stable_dt_s']=stable
    system=SimpleNamespace(stiffness=GPUElementOperator(storage,1.,0.),mass_diagonal=diagonal,stable_dt_s=stable,
        screen=screen,layout=mesh.base,device=base.device,body_attachments=list(attachments),local_mesh=mesh,native_factors=factors,
        body_patches=[SimpleNamespace(indices=p.ids.get()) for p in patches],
        inverse_mass_inplace=storage.inverse_mass_inplace,mass_quadratic=storage.mass_quadratic)
    def close():
        for p in patches:p.factor.close()
        for d in components.values():d.electrical.close()
    system.close=close;return system


class HybridActions(LocalActions):
    def __init__(self,system,fine):
        self.system=system;self.storage=system.stiffness.storage;self.base=self.storage.base
        self.fine=self.storage.cp.asarray(fine);self.mass=self.storage.cp.asarray(system.mass_diagonal)
        for factor in system.native_factors:factor.bind_stream()
        self.ids=self.base.select(fine);self.full_count=self.local_count=0
        for q in system.local_mesh.patches:
            ids=np.concatenate([q.support+c*system.local_mesh.n for c in range(3)])
            if not fine[ids].all():raise ValueError('Local mesh support must be wholly in the LTS fine region')
    def local(self,x):
        cp=self.storage.cp;projected=cp.where(self.fine,x,0.);result=cp.zeros_like(x)
        self.base.launch(projected[:3*self.base.ns],result[:3*self.base.ns],local=True)
        for p in self.storage.device_patches:result[p.ids]+=p.stiffness(projected[p.ids])
        self.local_count+=1;return self.inverse(result)


class HybridReusable(HybridActions,ReusableActions):
    def __init__(self,system,fine):
        HybridActions.__init__(self,system,fine);cp=self.storage.cp
        self.module=cp.RawModule(code=VECTOR_KERNELS,options=('--std=c++11','--fmad=false'))
        self.ops={name:self.module.get_function(name) for name in ('gather','scatter','divide_mass','project','update','sample','advance')}
    def inverse_into(self,x):self.system.inverse_mass_inplace(x)
    def full_into(self,x,out):self.storage._launch(x,out,1.,0.);self.inverse_into(out)
    def local_into(self,x,out):
        out.fill(0.);self.base.launch(x[:3*self.base.ns],out[:3*self.base.ns],local=True)
        for p in self.storage.device_patches:out[p.ids]+=p.stiffness(x[p.ids])
        self.inverse_into(out)


def clone(function,**replacements):
    ns=dict(function.__globals__);ns.update(replacements)
    f=FunctionType(function.__code__,ns,function.__name__,function.__defaults__,function.__closure__)
    f.__kwdefaults__=dict(function.__kwdefaults__ or {});return f

def gpu_lts_simulate(system,*args,**kwargs):
    """Original optimized recurrence bytecode, with separate mesh actions only."""
    from traxis import gpu_lts_optimized as executor
    if kwargs.get('substeps',2)!=2:raise ValueError('Local mesh adapter preserves the existing p=2 LTS executor')
    if kwargs.get('screen_stability',True):raise NotImplementedError('Select an explicitly qualified local-mesh stability policy')
    if kwargs.get('use_graphs',True) and not system.native_factors:raise ValueError('Native factors required for graph capture; use_graphs=False retains cached fallback')
    execute=clone(executor._execute,ReusableActions=HybridReusable,LocalActions=HybridActions)
    simulate=clone(executor.gpu_lts_simulate,_execute=execute)
    return simulate(system,*args,**kwargs)

def fine_partition(system,load,boxes=(),edge_width_m=0.):
    from traxis.local_time import fine_partition as original
    mesh=system.local_mesh;coarse=SimpleNamespace(mass_diagonal=np.ones(3*mesh.old_n),layout=mesh.base)
    old=original(coarse,np.zeros(3*mesh.old_n),boxes,edge_width_m).reshape(3,mesh.old_n)
    fine=np.asarray(load)!=0;view=fine[:3*mesh.n].reshape(3,mesh.n);view[:,:len(mesh.keep)]|=old[:,mesh.keep]
    for q in mesh.patches:view[:,q.support]=True
    for b in system.body_patches:fine[b.indices]=True
    return fine
