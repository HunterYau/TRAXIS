"""Optional execution optimizations for the existing FP64 LTS-LF2(p=2).

Plate stiffness and consistent PZT mass/stiffness kernels are reused verbatim.
Only vector arithmetic, workspace ownership and launch scheduling differ.
"""
import math
import time
import numpy as np
from traxis.gpu_local_time import LocalActions, lts_stability_screen
from traxis.gpu_local_time import gpu_lts_simulate as original_simulate
from traxis.gpu_enriched_verlet import SparseProbeSet, sparse_probe_set


VECTOR_KERNELS = r'''
extern "C" __global__ void gather(const double* x,const long long* ids,double* y,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=x[ids[i]];
}
extern "C" __global__ void scatter(double* x,const long long* ids,const double* y,int n,int add) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){if(add)x[ids[i]]+=y[i];else x[ids[i]]=y[i];}
}
extern "C" __global__ void divide_mass(double* x,const double* m,long long n) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]/=m[i];
}
extern "C" __global__ void project(const double* a,const double* drive,const bool* fine,
    const double* source,const int* step,double* out,long long n) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)out[i]=fine[i] ? a[i]-drive[i]*source[2*(long long)step[0]] : 0.;
}
extern "C" __global__ void update(double* previous,double* current,const double* a,
    const double* b,const double* drive,const double* source,const int* step,
    long long n,double h2,double h4) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){double old=current[i];
        double next=2.*old-previous[i]-h2*a[i]+h4*b[i]
                    +h2*drive[i]*source[2*(long long)step[0]+1];
        previous[i]=old;current[i]=next;}
}
extern "C" __global__ void sample(const int* starts,const int* ids,const double* weights,
    const double* u,double* output,int probes,const int* step) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=probes)return;
    double value=0.;for(int j=starts[p];j<starts[p+1];++j)value+=weights[j]*u[ids[j]];
    output[((long long)step[0]+1)*probes+p]=value;
}
extern "C" __global__ void advance(int* step){step[0]+=1;}
'''


class ReusableActions(LocalActions):
    """Allocation-free full/local matvec path with exact consistent mass blocks."""
    def __init__(self, system, fine):
        super().__init__(system,fine)
        cp=self.storage.cp
        self.module=cp.RawModule(code=VECTOR_KERNELS,options=('--std=c++11','--fmad=false'))
        self.ops={name:self.module.get_function(name) for name in
                  ('gather','scatter','divide_mass','project','update','sample','advance')}
        self.work=[]
        for ids,k,m,inv in getattr(self.storage,'patches',()):
            # Explicit int64 indices; each merged mass block has unique indices.
            self.work.append((ids.astype(cp.int64),k,inv,cp.empty(len(ids),dtype=cp.float64),
                              cp.empty(len(ids),dtype=cp.float64)))

    def vector(self,name,n,args):
        self.ops[name](((n+255)//256,),(256,),args)

    def gather(self,x,ids,out):
        self.vector('gather',len(ids),(x,ids,out,np.int32(len(ids))))

    def scatter(self,x,ids,y,add):
        self.vector('scatter',len(ids),(x,ids,y,np.int32(len(ids)),np.int32(add)))

    def dense(self,matrix,x,out):
        # Original kernel and reduction order, now using caller-owned output.
        self.storage.dense_kernel((len(x),),(128,),(matrix,x,out,np.int32(len(x))))

    def inverse_into(self,x):
        for ids,_,_,saved,_ in self.work:self.gather(x,ids,saved)
        self.vector('divide_mass',len(x),(x,self.mass,np.int64(len(x))))
        for ids,_,inv,saved,out in self.work:
            self.dense(inv,saved,out);self.scatter(x,ids,out,False)

    def full_into(self,x,out):
        out.fill(0.)
        host=3*self.base.ns
        self.base._kernel(x[:host],out[:host],1.,self.base.ne)
        for ids,k,_,saved,tmp in self.work:
            self.gather(x,ids,saved);self.dense(k,saved,tmp);self.scatter(out,ids,tmp,True)
        self.inverse_into(out)

    def local_into(self,projected,out):
        out.fill(0.);b=self.base;nx,ny,nz=b.layout.cells
        if len(self.ids):
            self.kernel((len(self.ids),),(b.screen['kernel_threads'],),(
                b.derivative,b.weights,b.inverse,projected,out,np.int32(nx),np.int32(ny),np.int32(nz),
                np.int32(b.layout.order),np.int64(b.ns),np.float64(b.determinant),np.float64(b.lam),
                np.float64(b.mu),np.float64(1.),np.int32(len(self.ids)),self.device_ids))
        for ids,k,_,saved,tmp in self.work:
            self.gather(projected,ids,saved);self.dense(k,saved,tmp);self.scatter(out,ids,tmp,True)
        self.inverse_into(out)


def _execute(system,steps,H,probes,load,fine,source,source_sampling,energy_stride,
             progress,guard,forecast,use_graphs,graph_steps):
    cp=system.stiffness.storage.cp
    # Evaluate the same samples as the original solver, before capture. No
    # fitted drive, interpolation or time shift. Source callables must be pure.
    samples=np.empty((steps,2))
    for j in range(steps):
        t=j*H;f=source(t)
        samples[j]=(f,f if source_sampling=='coarse' else .5*f+.25*(source(t+H/2)+source(t-H/2)))
    cp.cuda.get_current_stream().synchronize()
    stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        actions=ReusableActions(system,fine)
        drive=actions.inverse(cp.asarray(load,dtype=cp.float64))
        if bool(cp.any(drive[~actions.fine]!=0).item()):raise ValueError('Mass inverse moves source outside fine region')
        n=len(load)
        previous=cp.zeros(n,dtype=cp.float64);current=cp.zeros_like(previous)
        a=cp.empty_like(current);b=cp.empty_like(current);projected=cp.empty_like(current)
        step=cp.zeros(1,dtype=cp.int32);sources=cp.asarray(samples)
        count=len(probes.indices);traces=cp.zeros((steps+1,count),dtype=cp.float64)
        starts=cp.asarray(np.r_[0,np.cumsum([len(x) for x in probes.indices])].astype(np.int32))
        ids=cp.asarray(np.concatenate(probes.indices).astype(np.int32))
        weights=cp.asarray(np.concatenate(probes.weights))
        def one_step():
            actions.full_into(current,a)
            actions.vector('project',n,(a,drive,actions.fine,sources,step,projected,np.int64(n)))
            actions.local_into(projected,b)
            actions.vector('update',n,(previous,current,a,b,drive,sources,step,np.int64(n),
                                      np.float64(H**2),np.float64(H**4/16)))
            actions.vector('sample',count,(starts,ids,weights,current,traces,np.int32(count),step))
            actions.ops['advance']((1,),(1,),(step,))
        # Compile all kernels/ufuncs outside capture, then erase the warm-up.
        one_step();stream.synchronize()
        previous.fill(0.);current.fill(0.);traces.fill(0.);step.fill(0)
        stream.synchronize()
        graphs={};energies=[];energy_times=[];completed=0;launches=0
        # Use the original diagnostic implementation; never capture its host read.
        diagnostic=LocalActions(system,fine)
        tick=time.perf_counter();capture_seconds=0.
        while completed<steps:
            # Preserve the original every-16-step checks and every-stride energy
            # observations, including awkward non-divisible final chunks.
            chunk=min(graph_steps,steps-completed,16-completed%16,
                      energy_stride-completed%energy_stride)
            if use_graphs:
                if chunk not in graphs:
                    start=time.perf_counter()
                    stream.begin_capture()
                    try:
                        for _ in range(chunk):one_step()
                    finally:
                        graph=stream.end_capture()
                    graphs[chunk]=graph;capture_seconds+=time.perf_counter()-start
                graphs[chunk].launch(stream=stream);launches+=1
            else:
                for _ in range(chunk):one_step()
            completed+=chunk
            if completed%energy_stride==0 or completed==steps:
                value=diagnostic.discrete_energy(previous,current,H,2)
                if not math.isfinite(value):raise ArithmeticError('Nonfinite LTS discrete energy')
                energies.append(value);energy_times.append((completed-.5)*H)
            if completed%16==0 or completed==steps:
                stream.synchronize()
                if guard:guard.check()
                if not bool(cp.isfinite(current).all().item()):raise ArithmeticError('Nonfinite LTS state')
                if progress:progress(dict(steps_completed=completed,requested_steps=steps,
                    time_s=completed*H,elapsed_seconds=time.perf_counter()-tick))
        stream.synchronize();answer=cp.asnumpy(traces)
        if not np.isfinite(answer).all():raise ArithmeticError('Nonfinite LTS trace')
        if int(step.item())!=steps:raise ArithmeticError('CUDA graph replay step count mismatch')
        return np.arange(steps+1)*H,answer,dict(method='LTS-LF2',substeps=2,source_sampling=source_sampling,
            coarse_dt_s=H,fine_dt_s=H/2,precision='float64',fine_dofs=int(fine.sum()),total_dofs=n,
            local_elements=len(actions.ids),total_elements=system.layout.screen['elements'],
            full_actions=steps+diagnostic.full_count,local_actions=steps+diagnostic.local_count,
            workspace_bytes=forecast,energy_time_s=energy_times,modified_energy_j=energies,
            energy_diagnostic='Original M-weighted LTS-LF2 invariant; no claim while forced',
            solver_seconds=time.perf_counter()-tick,execution_backend='optimized',memory_reuse=True,
            fused_vector_updates=True,cuda_graphs=use_graphs,graph_launches=launches,
            graph_variants=len(graphs),graph_capture_seconds=capture_seconds,
            workspace_note='Conservative bound includes original diagnostic temporaries; graph nodes add runtime-managed metadata.')


def gpu_lts_simulate(system,steps,dt_s,probes,load,*,fine,substeps=2,force=None,
                     progress=None,guard=None,max_workspace_mib=4096,
                     mass_damping_per_s=0.,electrical_ports=(),energy_stride=256,
                     source_sampling='fine',use_graphs=True,graph_steps=16,screen_stability=True):
    """Opt-in optimized execution; the historical solver remains the fallback.

    p=2 uses fixed-buffer updates. Other ratios dispatch to the original.
    CUDA graph capture failures are raised, never silently ignored.
    """
    if type(use_graphs) is not bool or type(graph_steps) is not int or not 1<=graph_steps<=16:
        raise ValueError('Boolean graphs option and chunk size 1..16 required')
    if type(screen_stability) is not bool:raise ValueError('Boolean stability policy required')
    if substeps!=2:
        if not screen_stability:raise ValueError('Unchecked optimized execution supports ratio 2 only')
        return original_simulate(system,steps,dt_s,probes,load,fine=fine,substeps=substeps,
            force=force,progress=progress,guard=guard,max_workspace_mib=max_workspace_mib,
            mass_damping_per_s=mass_damping_per_s,electrical_ports=electrical_ports,
            energy_stride=energy_stride,source_sampling=source_sampling)
    mass,load,fine = np.asarray(system.mass_diagonal),np.asarray(load),np.asarray(fine)
    if source_sampling not in ('fine','coarse'):
        raise ValueError('Choose fine-source LF-LTS or coarse-source split-LFC')
    n = len(mass)
    if (type(steps) is not int or not 1<=steps<=500000 or type(substeps) is not int
            or not 1<=substeps<=16 or not math.isfinite(dt_s) or dt_s<=0
            or not math.isfinite(system.stable_dt_s) or system.stable_dt_s<=0
            or type(energy_stride) is not int
            or not 1<=energy_stride<=1024):
        raise ValueError('Invalid LTS steps or system stability bound')
    if mass_damping_per_s != 0 or electrical_ports:
        raise ValueError('LTS-LF2 requires lossless mechanics and condensed ideal-open ports')
    if (mass.shape!=(n,) or fine.shape!=(n,) or fine.dtype!=bool or load.shape!=(n,)
            or np.iscomplexobj(load) or not np.isfinite(load).all()
            or not np.isfinite(mass).all() or np.any(mass<=0) or np.any(load[~fine]!=0)):
        raise ValueError('Finite load entirely in fine region and positive mass required')
    for patch in getattr(system,'body_patches',()):
        if not fine[patch.indices].all(): raise ValueError('Fine projector splits a coupled mass block')
    if not isinstance(probes,SparseProbeSet): raise ValueError('Sparse probe set required')
    if len(probes.indices)!=len(probes.weights): raise ValueError('Probe rows and weights differ')
    probes=sparse_probe_set(zip(probes.indices,probes.weights),n)
    if sum(len(i) for i in probes.indices)>100000: raise MemoryError('Probe support exceeds 100000 coefficients')
    # Includes peak recurrence temporaries, local/full products, states, probes.
    forecast = (32*n*8+(steps+1)*len(probes.indices)*8+fine.nbytes
                +4*system.layout.screen['elements']+steps*16
                +sum(len(p.indices)*24 for p in getattr(system,'body_patches',())))
    if (not math.isfinite(max_workspace_mib) or not 0<max_workspace_mib<=65536
            or forecast>max_workspace_mib*1024**2
            or forecast/1024**2+system.screen['device_storage_screen_mib']>system.screen['max_device_mib']):
        raise MemoryError('LTS workspace exceeds declared device budget')
    storage = system.stiffness.storage
    cp = storage.cp
    def source(t):
        value = 0. if t<0 else float(force(t)) if force else 0.
        if not math.isfinite(value): raise ValueError('Nonfinite LTS source')
        return value
    if source(0.)!=0.: raise ValueError('Zero-state LTS requires force(0)=0')
    with storage.lock,cp.cuda.Device(storage.device_id):
        if guard: guard.check(force=True)
        if forecast>.8*cp.cuda.runtime.memGetInfo()[0]: raise MemoryError('Insufficient currently free GPU memory')
        stability=(lts_stability_screen(system,fine,dt_s,substeps) if screen_stability else
            dict(accepted=None,performed=False,kind='disabled_by_user',coarse_cfl_relaxed=None))
        t,y,stats=_execute(system,steps,dt_s,probes,load,fine,source,source_sampling,
            energy_stride,progress,guard,forecast,use_graphs,graph_steps)
        stats.update(stability_screen=stability,coarse_cfl_relaxed=stability['coarse_cfl_relaxed'])
        return t,y,stats
