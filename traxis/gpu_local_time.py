"""GPU LTS-LF2 with exact selected-element fine stiffness actions."""
import math
import time
import numpy as np
from traxis.local_time import lf2_step
from traxis.gpu_tensor import KERNEL, TensorStorage
from traxis.gpu_pzt_body import BodyStorage
from traxis.gpu_enriched_verlet import SparseProbeSet, PROBE_KERNEL, sparse_probe_set


def selected_elements(layout, fine):
    ns = layout.screen['scalar_dofs']
    support = fine[:3*ns].reshape(3,ns).any(axis=0)
    selected = []
    for first in range(0, layout.screen['elements'], 512):
        stop = min(first+512, layout.screen['elements'])
        ids = layout.element_nodes(first,stop)
        selected.extend((first+np.flatnonzero(support[ids].any(axis=1))).tolist())
    return np.array(selected, dtype=np.int32)


class LocalActions:
    def __init__(self, system, fine):
        self.system = system
        self.storage = system.stiffness.storage
        self.base = self.storage.base if isinstance(self.storage, BodyStorage) else self.storage
        if not isinstance(self.base, TensorStorage):
            raise ValueError('LTS adapter requires the affine tensor plate operator')
        cp = self.storage.cp
        self.fine = cp.asarray(fine)
        self.mass = cp.asarray(system.mass_diagonal)
        self.ids = selected_elements(system.layout, fine)
        self.device_ids = cp.asarray(self.ids)
        # Same proven element arithmetic; only the element lookup changes.
        code = KERNEL.replace('tensor_apply(', 'tensor_local_apply(').replace(
            'int ne) {', 'int ne,const int* selected) {').replace(
            'int q=order+1,n=q*q*q;', 'e=selected[e];\n    int q=order+1,n=q*q*q;')
        self.kernel = cp.RawKernel(code, 'tensor_local_apply', options=(
            '--std=c++11',f'-DLOCAL_NODES={(system.layout.order+1)**3}'))
        self.full_count = self.local_count = 0

    def inverse(self, x):
        if hasattr(self.system,'inverse_mass_inplace'): self.system.inverse_mass_inplace(x)
        else: x /= self.mass
        return x

    def full(self, x):
        result = self.storage.cp.empty_like(x)
        self.storage._launch(x,result,1.,0.)
        self.full_count += 1
        return self.inverse(result)

    def local(self, x):
        cp, b = self.storage.cp, self.base
        projected = cp.where(self.fine,x,0.)
        result = cp.zeros_like(x)
        nx,ny,nz = b.layout.cells
        if len(self.ids):
            self.kernel((len(self.ids),),(b.screen['kernel_threads'],),(
                b.derivative,b.weights,b.inverse,projected,result,np.int32(nx),np.int32(ny),np.int32(nz),
                np.int32(b.layout.order),np.int64(b.ns),np.float64(b.determinant),np.float64(b.lam),
                np.float64(b.mu),np.float64(1.),np.int32(len(self.ids)),self.device_ids))
        for ids, stiffness, _, _ in getattr(self.storage,'patches',()):
            result[ids] += self.storage.matvec(stiffness,projected[ids])
        self.local_count += 1
        return self.inverse(result)

    def mass_action(self, x):
        result=self.mass*x
        for ids,_,mass,_ in getattr(self.storage,'patches',()):
            result[ids] += self.storage.matvec(mass,x[ids])-self.mass[ids]*x[ids]
        return result

    def discrete_energy(self, previous, current, H, p):
        # In physical coordinates A_p is M-self-adjoint, not generally symmetric.
        # q(H)=2u-H^2*A_p*u; previous=current gives next=u-H^2*A_p*u.
        unforced=lf2_step(current,current,H,p,self.full,self.local,self.fine,
                          0.,lambda _:0.,0.)
        effective=(current-unforced)/(H*H)
        velocity=(current-previous)/H
        cp=self.storage.cp
        energy=.5*(cp.sum(velocity*self.mass_action(velocity),dtype=cp.float64)
                   +cp.sum(previous*self.mass_action(effective),dtype=cp.float64))
        return float(energy.item())


def gpu_lts_simulate(system,steps,dt_s,probes,load,*,fine,substeps=2,force=None,
                     progress=None,guard=None,max_workspace_mib=4096,
                     mass_damping_per_s=0.,electrical_ports=(),energy_stride=256):
    """Zero-state lossless response on coarse nodes, all dynamics in FP64.

    Until a full-plate stability study authorizes relaxed coarse CFL values,
    retain the existing global stability bound on the COARSE step. This is
    conservative, not a claim that fine PZT CFL limits have been removed.
    """
    mass,load,fine = np.asarray(system.mass_diagonal),np.asarray(load),np.asarray(fine)
    n = len(mass)
    if (type(steps) is not int or not 1<=steps<=500000 or type(substeps) is not int
            or not 1<=substeps<=16 or not math.isfinite(dt_s) or dt_s<=0
            or not math.isfinite(system.stable_dt_s) or system.stable_dt_s<=0
            or dt_s>.8*system.stable_dt_s or type(energy_stride) is not int
            or not 1<=energy_stride<=1024):
        raise ValueError('Invalid LTS steps or conservative coarse stability limit exceeded')
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
    forecast = 32*n*8+(steps+1)*len(probes.indices)*8+fine.nbytes+4*system.layout.screen['elements']
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
        actions = LocalActions(system,fine)
        drive = actions.inverse(cp.asarray(load,dtype=cp.float64))
        # Mass block closure guarantees M^-1 does not move drive outside P.
        if bool(cp.any(drive[~actions.fine]!=0).item()): raise ValueError('Mass inverse moves source outside fine region')
        starts = np.r_[0,np.cumsum([len(x) for x in probes.indices])].astype(np.int32)
        pi=cp.asarray(np.concatenate(probes.indices));pw=cp.asarray(np.concatenate(probes.weights))
        ps=cp.asarray(starts); count=len(probes.indices)
        sample=cp.RawKernel(PROBE_KERNEL,'sample_probes',options=('--std=c++11',))
        previous=cp.zeros(n,dtype=cp.float64);current=cp.zeros_like(previous)
        traces=cp.zeros((steps+1,count),dtype=cp.float64)
        energy_times=[];energies=[]
        tick=time.perf_counter()
        for j in range(steps):
            following=lf2_step(previous,current,dt_s,substeps,actions.full,actions.local,
                               actions.fine,drive,source,j*dt_s)
            previous,current=current,following
            if (j+1)%energy_stride==0 or j+1==steps:
                value=actions.discrete_energy(previous,current,dt_s,substeps)
                if not math.isfinite(value): raise ArithmeticError('Nonfinite LTS discrete energy')
                energy_times.append((j+.5)*dt_s);energies.append(value)
            sample((1,),(64,),(ps,pi,pw,current,traces,np.int32(count),np.int32(j+1)))
            if (j+1)%16==0 or j+1==steps:
                cp.cuda.get_current_stream().synchronize()
                if guard: guard.check()
                if not bool(cp.isfinite(current).all().item()): raise ArithmeticError('Nonfinite LTS state')
                if progress: progress(dict(steps_completed=j+1,requested_steps=steps,
                    time_s=(j+1)*dt_s,elapsed_seconds=time.perf_counter()-tick))
        answer=cp.asnumpy(traces)
        if not np.isfinite(answer).all(): raise ArithmeticError('Nonfinite LTS trace')
        return np.arange(steps+1)*dt_s,answer,dict(method='LTS-LF2',substeps=substeps,
            coarse_dt_s=dt_s,fine_dt_s=dt_s/substeps,precision='float64',
            fine_dofs=int(fine.sum()),total_dofs=n,local_elements=len(actions.ids),
            total_elements=system.layout.screen['elements'],full_actions=actions.full_count,
            local_actions=actions.local_count,workspace_bytes=forecast,
            energy_time_s=energy_times,modified_energy_j=energies,
            energy_diagnostic='M-weighted LTS-LF2 invariant; no energy-conservation claim while forced',
            solver_seconds=time.perf_counter()-tick,coarse_cfl_relaxed=False)
