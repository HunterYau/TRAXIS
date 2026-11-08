"""Float64 CG with device-resident coefficient vectors and checked residuals."""

from dataclasses import dataclass
import math

import numpy as np

from traxis.gpu_elements import GPUElementOperator,GPUElementPreconditioner
from traxis.iterative_fem import CheckedCGResult,IterativeSettings


@dataclass
class GPUCGResult(CheckedCGResult):
    statistics: dict


def gpu_checked_cg(operator, rhs, guess, diagonal, settings=None, *, guard=None,
                   progress=None, preconditioner=None):
    """Solve the original SPD operator without per-iteration vector transfers.

    Only scalar reductions synchronize during CG. Convergence requires the
    recurrence target and a fresh final operator residual, as on the CPU path.
    Optional Schwarz storage must belong to the same checked element system.
    """
    settings = settings or IterativeSettings()
    if not isinstance(operator,GPUElementOperator):
        raise ValueError('device CG requires a native GPU element operator')
    n = operator.shape[0]
    rhs,guess,diagonal = (np.asarray(v) for v in (rhs,guess,diagonal))
    if (not 1 <= n <= settings.max_dofs
            or any(np.iscomplexobj(v) or v.shape != (n,) or not np.isfinite(v).all() for v in (rhs,guess,diagonal))
            or np.any(diagonal <= 0) or (progress is not None and not callable(progress))
            or (preconditioner is not None and (not isinstance(preconditioner,GPUElementPreconditioner)
                or preconditioner.storage is not operator.storage))):
        raise ValueError('device CG requires bounded real vectors, a positive diagonal and matching SPD preconditioner')
    workspace = 12*n*8/1024**2
    if workspace > settings.max_workspace_mib:
        raise MemoryError('device CG workspace exceeds its declared screen')
    storage = operator.storage
    cp = storage.cp
    count,scalar_reads = 0,0
    def emit(**fields):
        if progress:
            progress(fields)
    def scalar(value):
        nonlocal scalar_reads
        answer = float(value.item())
        scalar_reads += 1
        if not math.isfinite(answer):
            raise ArithmeticError('device CG reduction is nonfinite')
        return answer
    def dot(a,b):
        return scalar(cp.sum(a*b,dtype=cp.float64))
    def apply(a,b):
        storage._launch(a,b,operator.stiffness_scale,operator.mass_scale)
        operator.applications += 1
    def precondition(a,b):
        if preconditioner is None:
            cp.divide(a,device_diagonal,out=b)
        else:
            storage._launch(a,b,inverse=True)
            preconditioner.applications += 1
        rho = dot(a,b)
        if rho <= 0:
            raise ArithmeticError('device CG preconditioner produced nonpositive work')
        return rho
    with storage.lock,cp.cuda.Device(storage.device_id):
        if guard:
            guard.check(force=True)
        uploads_before,downloads_before = storage.vector_uploads,storage.vector_downloads
        b,x = cp.asarray(rhs,dtype=cp.float64),cp.asarray(guess,dtype=cp.float64)
        storage.vector_uploads += 2
        device_diagonal = None
        if preconditioner is None:
            device_diagonal = cp.asarray(diagonal,dtype=cp.float64)
            storage.vector_uploads += 1
        r,z,p,q = (cp.empty(n,dtype=cp.float64) for _ in range(4))
        b_squared = dot(b,b)
        if b_squared == 0.:
            if np.any(rhs):
                raise ValueError('device CG right-hand-side norm underflowed')
            x.fill(0.)
        emit(phase='solve_start',solver='gpu_cg',rhs_norm=math.sqrt(b_squared),
            relative_tolerance=settings.relative_tolerance)
        apply(x,q)
        cp.subtract(b,q,out=r)
        r_squared = dot(r,r)
        target = settings.relative_tolerance**2*b_squared
        converged = r_squared <= target
        rho = precondition(r,z) if not converged else 0.
        cp.copyto(p,z) if not converged else p.fill(0.)
        for iteration in range(1,settings.max_iterations+1):
            if converged:
                break
            if guard:
                guard.check()
            apply(p,q)
            curvature = dot(p,q)
            if curvature <= 0:
                raise ArithmeticError('device CG operator produced nonpositive curvature')
            alpha = rho/curvature
            x += alpha*p
            r -= alpha*q
            count = iteration
            r_squared = dot(r,r)
            emit(phase='cg_iteration',iteration=count,
                recurrence_relative_residual=math.sqrt(r_squared/b_squared))
            converged = r_squared <= target
            if not converged:
                following_rho = precondition(r,z)
                p *= following_rho/rho
                p += z
                rho = following_rho
        # Recompute against the original operator, independently of the recurrence.
        apply(x,q)
        cp.subtract(q,b,out=r)
        relative = math.sqrt(dot(r,r))/max(math.sqrt(b_squared),1e-30)
        emit(phase='solve_end',solver='gpu_cg',iterations=count,status=0 if converged else settings.max_iterations,
            true_relative_residual=relative)
        if not converged or relative > settings.residual_limit:
            raise ArithmeticError(f'device CG solve failed: iterations={count}, relative residual={relative:.6g}')
        answer = cp.asnumpy(x)
        storage.vector_downloads += 1
        if not np.isfinite(answer).all():
            raise ArithmeticError('device CG solution is nonfinite')
        if guard:
            guard.check(force=True)
        statistics = dict(solver='gpu_cg',precision='float64',iterations=count,
            device_workspace_screen_mib=workspace,scalar_readbacks=scalar_reads,
            host_vector_uploads=storage.vector_uploads-uploads_before,
            host_vector_downloads=storage.vector_downloads-downloads_before,
            preconditioner='element_schwarz' if preconditioner is not None else 'diagonal')
        return GPUCGResult(answer,count,relative,statistics)
