"""Optional stability screening; the original LF2 solver is never modified.

The numerical certificate remains a two-start Ritz screen, not a rigorous
eigenvalue upper bound. Exact-model cache hits reuse that same certificate.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
from traxis import gpu_local_time_requested as original

CRITERIA = dict(schema=1, precision='float64', tol=1e-8, residual=1e-6,
                agreement=1e-5, threshold=3.8, ncv=32, maxiter=1000,
                seeds=[2718, 3141])


def conservative_upper(H, stable_dt, all_fine=False):
    """p=2: B=T-TPT/16 <= T; P=I also admits a polynomial bound.

    T=H^2 M^-1/2 K M^-1/2, lambda(T)<=4(H/stable_dt)^2.
    Positivity is checked separately using the unchanged fine-step bound.
    """
    x=4*(H/stable_dt)**2
    bound=(x-x*x/16 if x<=8 else 4.) if all_fine else x
    return float(bound+128*np.finfo(float).eps*max(1.,abs(bound)))


def _array_hash(h, value, cp=None):
    # Bound staging memory, including device PZT matrices and full plate mass.
    h.update(str(value.dtype).encode()); h.update(str(value.shape).encode())
    flat=value.reshape(-1)
    chunk=max(1,8*1024**2//value.dtype.itemsize)
    for first in range(0,flat.size,chunk):
        part=flat[first:first+chunk]
        a=cp.asnumpy(part) if cp is not None else np.asarray(part)
        h.update(np.ascontiguousarray(a).tobytes())


def model_fingerprint(system, fine):
    """Hash actual operator coefficients, not a user-supplied config label.

    Only the existing affine TensorStorage/BodyStorage adapter is supported.
    Unknown adapters fail rather than reusing an unverifiable cache entry.
    """
    storage=system.stiffness.storage
    base=storage.base if isinstance(storage, original.BodyStorage) else storage
    if type(base) is not original.TensorStorage or type(storage) not in (original.TensorStorage,original.BodyStorage):
        raise ValueError('Fingerprint requires the original tensor/PZT operator')
    if (system.stiffness.stiffness_scale,system.stiffness.mass_scale)!=(1.,0.):
        raise ValueError('Unexpected stiffness scaling')
    h=hashlib.sha256()
    h.update(json.dumps(CRITERIA,sort_keys=True).encode())
    # Invalidate across changes to solver, operator, mass or screening code.
    for file in sorted(Path(__file__).parent.glob('*.py')):
        h.update(file.name.encode());h.update(file.read_bytes())
    h.update(repr((system.layout.cells,system.layout.order,base.determinant,
                   base.lam,base.mu,base.ns,base.ne,system.stable_dt_s)).encode())
    h.update(repr((base.kernel.code,base.kernel.options,original.KERNEL)).encode())
    for a in (system.mass_diagonal,fine,system.layout.transform): _array_hash(h,np.asarray(a))
    for a in system.layout.axes_m: _array_hash(h,np.asarray(a))
    for a in (base.derivative,base.weights,base.inverse): _array_hash(h,a,storage.cp)
    if isinstance(storage,original.BodyStorage):
        h.update(repr((storage.dense_kernel.code,storage.dense_kernel.options)).encode())
        _array_hash(h,storage.diagonal,storage.cp)
        for patch in storage.patches:
            for a in patch: _array_hash(h,a,storage.cp)
    # The normalization uses the consistent host patch mass, so hash it too.
    for patch in getattr(system,'body_patches',()):
        _array_hash(h,np.asarray(patch.indices));_array_hash(h,np.asarray(patch.mass))
    import scipy
    h.update(repr((np.__version__,scipy.__version__,storage.cp.__version__)).encode())
    return h.hexdigest()


def _valid_report(report):
    if not isinstance(report,dict) or report.get('accepted') is not True: return False
    if report.get('kind')=='conservative_full_system_bound': return True
    if report.get('formal_upper_bound') is True:
        b=report.get('effective_upper_bound',float('nan'))
        return math.isfinite(b) and 0<=b<CRITERIA['threshold']
    vals=report.get('estimated_H2_lambda_max',[]);res=report.get('residuals',[])
    return (len(vals)==len(res)==2 and all(math.isfinite(x) for x in vals+res)
            and all(0<=x<=CRITERIA['residual'] for x in res)
            and min(vals)>=-CRITERIA['residual']
            and max(vals)-min(vals)<=CRITERIA['agreement']
            and max(vals)+max(res)<CRITERIA['threshold'])


def _atomic_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f: json.dump(value,f,sort_keys=True)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def _read_report(path, key):
    try:
        data=json.loads(path.read_text())
        report=data['report']
        digest=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()
        if data['key']==key and data['criteria']==CRITERIA and data['sha256']==digest and _valid_report(report):
            return report
    except (OSError,ValueError,KeyError,TypeError): pass
    return None


def _gpu_available(cp):
    """Check optional cuBLAS availability BEFORE selecting a GPU eigensolver."""
    from cupyx.scipy.sparse.linalg import eigsh,LinearOperator
    # The original custom stiffness kernels do not require cuBLAS; eigsh does.
    cp.dot(cp.ones(2,dtype=cp.float64),cp.ones(2,dtype=cp.float64)).item()
    return eigsh,LinearOperator


class NormalizedOperator:
    """Same Cholesky mass similarity as the original screen, kept on device."""
    def __init__(self,system,fine,H,backend,progress):
        from scipy.linalg import solve_triangular
        self.actions=original.LocalActions(system,fine)
        self.cp=system.stiffness.storage.cp;self.H=H;self.backend=backend
        self.calls=0;self.tick=time.perf_counter();self.progress=progress
        self.root=np.sqrt(system.mass_diagonal);self.blocks=[]
        for patch in getattr(system,'body_patches',()):
            L=np.linalg.cholesky(patch.mass)
            invLT=solve_triangular(L.T,np.eye(len(L)),lower=False)
            if backend=='gpu':
                self.blocks.append((self.cp.asarray(patch.indices),self.cp.asarray(L.T,order='C'),
                                    self.cp.asarray(invLT,order='C')))
            else: self.blocks.append((patch.indices,L.T,invLT))
        if backend=='gpu': self.root=self.cp.asarray(self.root)

    def __call__(self,z):
        self.calls+=1;cp=self.cp
        physical=z/self.root
        for ids,LT,invLT in self.blocks:
            physical[ids]=(self.actions.storage.matvec(invLT,z[ids]) if self.backend=='gpu' else invLT@z[ids])
        a=self.actions.full(physical if self.backend=='gpu' else cp.asarray(physical))
        out=a-self.H**2/16*self.actions.local(a)
        if self.backend!='gpu': out=cp.asnumpy(out)
        normalized=self.root*out
        for ids,LT,_ in self.blocks:
            normalized[ids]=(self.actions.storage.matvec(LT,out[ids]) if self.backend=='gpu' else LT@out[ids])
        if self.calls%32==0 and self.progress:
            self.progress(dict(phase='checking_stability',backend=self.backend,
                operator_evaluations=self.calls,elapsed_seconds=time.perf_counter()-self.tick))
        return self.H**2*normalized


def _warm_vector(path, model, n):
    try:
        with np.load(path,allow_pickle=False) as data:
            if str(data['model'].item())!=model: return None
            v=data['vector']
            if v.dtype!=np.float64 or v.shape!=(n,) or not np.isfinite(v).all(): return None
            if str(data['sha256'].item())!=hashlib.sha256(v.tobytes()).hexdigest(): return None
            norm=np.linalg.norm(v)
            return v/norm if norm>0 else None
    except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile): return None


def _save_warm(path,model,vector):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as f:
            np.savez(f,model=model,vector=vector,sha256=hashlib.sha256(vector.tobytes()).hexdigest())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def optimized_stability_screen(system,fine,H,p,progress=None,*,backend='auto',
                               cache_dir=None,max_warm_bytes=256*1024**2):
    storage=system.stiffness.storage
    with storage.lock,storage.cp.cuda.Device(storage.device_id):
        return _screen(system,fine,H,p,progress,backend=backend,cache_dir=cache_dir,
                       max_warm_bytes=max_warm_bytes)


def _screen(system,fine,H,p,progress=None,*,backend='auto',
                               cache_dir=None,max_warm_bytes=256*1024**2):
    """Original criteria, optional GPU Lanczos, exact cache and one warm start.

    backend='original' delegates directly to the unchanged original screen.
    This function primes the original solver's existing per-system cache so its
    own pre-integration call does not repeat an already successful screen.
    """
    if backend not in ('auto','gpu','cpu','original'): raise ValueError('Unknown stability backend')
    if backend=='original': return original.lts_stability_screen(system,fine,H,p,progress=progress)
    fine=np.asarray(fine);mass=np.asarray(system.mass_diagonal)
    if (not math.isfinite(H) or H<=0 or type(p) is not int or p<1
            or not math.isfinite(system.stable_dt_s) or system.stable_dt_s<=0
            or mass.ndim!=1 or not np.isfinite(mass).all() or np.any(mass<=0)
            or fine.dtype!=bool or fine.shape!=mass.shape):
        raise ValueError('Invalid stability inputs')
    for patch in getattr(system,'body_patches',()):
        if not fine[patch.indices].all(): raise ValueError('Fine projector splits mass block')
    if H<=.8*system.stable_dt_s:
        return original.lts_stability_screen(system,fine,H,p,progress=progress)
    if p!=2 or H/p>=system.stable_dt_s:
        raise ValueError('Relaxed coarse stepping requires p=2 and stable fine step')
    started=time.perf_counter();cp=system.stiffness.storage.cp;n=len(fine)
    model=model_fingerprint(system,fine)
    key=hashlib.sha256((model+repr((H,p))).encode()).hexdigest()
    legacy_key=(H,p,hashlib.sha256(fine.tobytes()).hexdigest())
    directory=Path(cache_dir) if cache_dir is not None else None
    def finish(report,reused=False):
        answer=copy.deepcopy(report);answer.update(cache_hit=reused,model_fingerprint=model,
            cache_key=key,screen_wall_seconds=time.perf_counter()-started)
        if reused:
            answer['certificate_operator_evaluations']=report.get('operator_evaluations',0)
            answer['operator_evaluations']=0
        if not _valid_report(answer): raise ArithmeticError('Invalid stability certificate')
        cache=getattr(system,'_optimized_lts_stability_cache',{})
        cache[key]=copy.deepcopy(answer);system._optimized_lts_stability_cache=cache
        # No monkey-patching: original solver already has this cache hook.
        cache=getattr(system,'_lts_stability_cache',{})
        cache[legacy_key]=copy.deepcopy(answer);system._lts_stability_cache=cache
        return answer
    saved=getattr(system,'_optimized_lts_stability_cache',{}).get(key)
    if saved is not None and _valid_report(saved): return finish(saved,True)
    if directory is not None:
        saved=_read_report(directory/(key+'.json'),key)
        if saved is not None: return finish(saved,True)
    common=dict(accepted=True,coarse_dt_s=H,fine_dt_s=H/p,coarse_cfl_relaxed=True,
                rejection_threshold=3.8,theoretical_limit=4.,positivity='fine-step full-system bound')
    upper=conservative_upper(H,system.stable_dt_s,bool(fine.all()))
    if upper<CRITERIA['threshold']:
        report=dict(common,kind='conservative_effective_operator_bound',formal_upper_bound=True,
                    effective_upper_bound=upper,operator_evaluations=0,backend='analytic',
                    bound='all-fine polynomial' if fine.all() else 'B <= H^2 S')
        vector=None
    else:
        import psutil
        ncv=min(CRITERIA['ncv'],n)
        if ncv<=1: raise ValueError('Spectral screen needs at least two DOFs')
        # Basis, extra recurrence/staging vectors, and small consistent-mass blocks.
        patch_bytes=sum(3*b.mass.nbytes for b in getattr(system,'body_patches',()))
        workspace=(ncv+16)*n*8+patch_bytes
        free=cp.cuda.runtime.memGetInfo()[0]
        storage_bytes=system.screen['device_storage_screen_mib']*1024**2
        budget=system.screen['max_device_mib']*1024**2
        fits=(workspace<.7*free and workspace+storage_bytes<budget)
        reason=None;selected=backend
        if backend in ('auto','gpu'):
            if not fits:
                if backend=='gpu': raise MemoryError('GPU eigensolver basis exceeds memory budget')
                selected='cpu';reason='GPU basis exceeds screened memory budget'
            else:
                try:
                    gpu_eigsh,gpu_operator=_gpu_available(cp);selected='gpu'
                except (ImportError,OSError,RuntimeError) as exc:
                    if backend=='gpu': raise
                    selected='cpu';reason=f'Optional GPU eigensolver unavailable: {type(exc).__name__}'
        if selected=='cpu':
            if workspace>.7*psutil.virtual_memory().available:
                raise MemoryError('CPU eigensolver basis exceeds available RAM')
            from scipy.sparse.linalg import eigsh,LinearOperator
            xp=np
        else: eigsh,LinearOperator,xp=gpu_eigsh,gpu_operator,cp
        operation=NormalizedOperator(system,fine,H,selected,progress)
        op=LinearOperator((n,n),matvec=operation,dtype=np.float64)
        warm=None
        if directory is not None and n*8<=max_warm_bytes:
            warm=_warm_vector(directory/'warm-latest.npz',model,n)
        estimates=[];residuals=[];vector=None
        for index,seed in enumerate(CRITERIA['seeds']):
            v0=np.random.default_rng(seed).standard_normal(n);v0/=np.linalg.norm(v0)
            # Only first start uses a hint. Perturb it to avoid a trapped exact
            # lower eigenvector; second start is independently random as before.
            if index==0 and warm is not None:
                v0=warm+1e-3*v0;v0/=np.linalg.norm(v0)
            vals,vecs=eigsh(op,k=1,which='LA',v0=xp.asarray(v0),ncv=ncv,
                maxiter=CRITERIA['maxiter'],tol=CRITERIA['tol'])
            v=vecs[:,0];norm=float(xp.sqrt(xp.sum(v*v)));v=v/norm
            value=float(vals[0]);error=operation(v)-value*v
            residual=float(xp.sqrt(xp.sum(error*error)))
            if not math.isfinite(value) or not math.isfinite(residual) or residual>CRITERIA['residual']:
                raise ArithmeticError('LTS spectral screen failed to converge')
            # CuPy may return its last Ritz pair at maxiter without raising.
            # Explicitly enforce its requested absolute convergence tolerance;
            # never accept a merely approximate max-iteration return.
            if selected=='gpu' and residual>CRITERIA['tol']:
                raise ArithmeticError('GPU eigensolver did not reach the requested tolerance')
            estimates.append(value);residuals.append(residual)
            if index==0 and directory is not None and n*8<=max_warm_bytes:
                vector=cp.asnumpy(v) if selected=='gpu' else np.array(v,copy=True)
            if progress: progress(dict(phase='stability_start_finished',start=index+1,
                backend=selected,estimated_H2_lambda_max=value,residual=residual,
                operator_evaluations=operation.calls,elapsed_seconds=time.perf_counter()-started))
        report=dict(common,kind='two_start_mass_normalized_ritz_screen',formal_upper_bound=False,
            estimated_H2_lambda_max=estimates,residuals=residuals,
            operator_evaluations=operation.calls,backend=selected,warm_start_used=warm is not None,
            basis_vectors=ncv,workspace_forecast_bytes=workspace,fallback_reason=reason)
        if not _valid_report(report): raise ArithmeticError('LTS coarse step fails effective-operator stability screen')
    answer=finish(report)
    if directory is not None:
        try:
            _atomic_json(directory/(key+'.json'),dict(key=key,criteria=CRITERIA,report=report,
                sha256=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()))
            if vector is not None: _save_warm(directory/'warm-latest.npz',model,vector)
        except OSError as exc:
            # Optional cache failure cannot invalidate a newly completed screen.
            answer['cache_write_warning']=type(exc).__name__
    return answer
