"""GPU LTS-LF2 with exact selected-element fine stiffness actions."""
import math
import time
import hashlib
import numpy as np
from traxis.local_time import lf2_step,split_lf2_step
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


def lts_stability_screen(system, fine, H, p, progress=None):
    """Screen the actual LF2 effective operator before relaxed-coarse runs.

    Grote & Mitkova (2012), Prop. 3.3: 0 <= H^2 A_p < 4 I in
    mass-normalized coordinates. A converged Ritz estimate is a numerical
    screen, NOT a rigorous upper eigenvalue bound. No automatic step fallback.
    """
    if not math.isfinite(H) or H<=0 or type(p) is not int or p<1:
        raise ValueError('Positive coarse step and integer ratio required')
    fine=np.asarray(fine)
    if fine.dtype!=bool or fine.shape!=np.asarray(system.mass_diagonal).shape:
        raise ValueError('Fine partition does not match system')
    for patch in getattr(system,'body_patches',()):
        if not fine[patch.indices].all(): raise ValueError('Fine projector splits mass block')
    if H<=.8*system.stable_dt_s:
        return dict(accepted=True,kind='conservative_full_system_bound',coarse_dt_s=H,
                    fine_dt_s=H/p,coarse_cfl_relaxed=False)
    if p!=2 or H/p>=system.stable_dt_s:
        raise ValueError('Relaxed coarse stepping requires p=2 and stable fine step')
    # For p=2, A_p=A-H^2 A P A/16 is PSD when H^2 lambda(PAP)<16.
    # Experimental opt-in: require the fine step strictly below the full-system bound (no extra 20% margin).
    key=(H,p,hashlib.sha256(fine.tobytes()).hexdigest())
    cache=getattr(system,'_lts_stability_cache',{})
    if key in cache: return dict(cache[key])
    cp=system.stiffness.storage.cp
    from scipy.sparse.linalg import LinearOperator,eigsh
    from scipy.linalg import solve_triangular
    import psutil
    ncv=min(32,len(fine))
    # ARPACK basis plus work vectors; no dense full-plate matrices.
    if (ncv+12)*len(fine)*8>.7*psutil.virtual_memory().available:
        raise MemoryError('Insufficient host RAM for bounded stability-screen basis')
    actions=LocalActions(system,fine)
    root=np.sqrt(system.mass_diagonal)
    # M=L L^T; z=L^T u. Whole patch blocks commute with P. Thus
    # L^T A_p L^-T is Euclidean symmetric, even for consistent PZT mass.
    blocks=[(b.indices,np.linalg.cholesky(b.mass))
            for b in getattr(system,'body_patches',())]
    def physical(z):
        u=z/root
        for ids,L in blocks: u[ids]=solve_triangular(L.T,z[ids],lower=False)
        return u
    def normalized(u):
        z=root*u
        for ids,L in blocks: z[ids]=L.T@u[ids]
        return z
    calls=0
    tick=time.perf_counter()
    def matvec(z):
        nonlocal calls
        calls+=1
        if calls%32==0 and progress:
            progress(dict(phase='checking_stability',operator_evaluations=calls,
                          elapsed_seconds=time.perf_counter()-tick))
        a=actions.full(cp.asarray(physical(z)))
        return H**2*normalized(cp.asnumpy(a-H**2/16*actions.local(a)))
    n=len(root); op=LinearOperator((n,n),matvec=matvec,dtype=np.float64)
    estimates=[];residuals=[]
    # Two independent starts and bounded workspace. Let ARPACK complete its
    # iterations rather than aborting at an unrelated operator-call count.
    for seed in (2718,3141):
        v0=np.random.default_rng(seed).standard_normal(n)
        vals,vecs=eigsh(op,k=1,which='LA',v0=v0,ncv=ncv,maxiter=1000,tol=1e-8)
        value=float(vals[0])
        residual=float(np.linalg.norm(matvec(vecs[:,0])-vals[0]*vecs[:,0]))
        if not math.isfinite(value) or not math.isfinite(residual) or residual>1e-6:
            raise ArithmeticError('LTS spectral screen failed to converge')
        estimates.append(value);residuals.append(residual)
        if progress: progress(dict(phase='stability_start_finished',start=len(estimates),
            estimated_H2_lambda_max=value,residual=residual,operator_evaluations=calls,
            elapsed_seconds=time.perf_counter()-tick))
    if max(estimates)-min(estimates)>1e-5 or max(estimates)+max(residuals)>=3.8:
        raise ArithmeticError('LTS coarse step fails effective-operator stability screen')
    report=dict(accepted=True,kind='two_start_mass_normalized_ritz_screen',
        formal_upper_bound=False,estimated_H2_lambda_max=estimates,residuals=residuals,
        rejection_threshold=3.8,theoretical_limit=4.,positivity='fine-step full-system bound',
        operator_evaluations=calls,host_basis_vectors=ncv,
        backend='CPU ARPACK iteration, GPU stiffness/mass actions',
        coarse_dt_s=H,fine_dt_s=H/p,coarse_cfl_relaxed=True)
    cache[key]=report; system._lts_stability_cache=cache
    return dict(report)


def gpu_lts_simulate(system,steps,dt_s,probes,load,*,fine,substeps=2,force=None,
                     progress=None,guard=None,max_workspace_mib=4096,
                     mass_damping_per_s=0.,electrical_ports=(),energy_stride=256,
                     source_sampling='fine'):
    """Zero-state lossless response on coarse nodes, all dynamics in FP64.

    Relaxed coarse p=2 runs must pass the actual effective-operator screen;
    the fine step retains the original full-system stability safety factor.
    """
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
        stability=lts_stability_screen(system,fine,dt_s,substeps)
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
        stepper=split_lf2_step if source_sampling=='coarse' else lf2_step
        for j in range(steps):
            following=stepper(previous,current,dt_s,substeps,actions.full,actions.local,
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
            source_sampling=source_sampling,
            coarse_dt_s=dt_s,fine_dt_s=dt_s/substeps,precision='float64',
            fine_dofs=int(fine.sum()),total_dofs=n,local_elements=len(actions.ids),
            total_elements=system.layout.screen['elements'],full_actions=actions.full_count,
            local_actions=actions.local_count,workspace_bytes=forecast,
            energy_time_s=energy_times,modified_energy_j=energies,
            energy_diagnostic='M-weighted LTS-LF2 invariant; no energy-conservation claim while forced',
            solver_seconds=time.perf_counter()-tick,stability_screen=stability,
            coarse_cfl_relaxed=stability['coarse_cfl_relaxed'])
