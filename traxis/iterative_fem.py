"""Average-acceleration Newmark using checked matrix-free CG solves.

Callers must establish symmetric positive mass and positive-semidefinite
elastic stiffness. The preconditioner uses the positive operator diagonal;
the full consistent mass remains in every operator equation and energy.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator, cg

from traxis.config import positive


@dataclass(frozen=True)
class IterativeSettings:
    relative_tolerance: float = 1e-11
    residual_limit: float = 1e-9
    max_iterations: int = 200
    max_dofs: int = 2000000
    max_workspace_mib: float = 512

    def __post_init__(self):
        if not (math.isfinite(self.relative_tolerance) and math.isfinite(self.residual_limit)
                and 1e-15 <= self.relative_tolerance <= self.residual_limit <= 1e-9):
            raise ValueError('iterative tolerances must lie in [1e-15,1e-9] and target the checked residual')
        if type(self.max_iterations) is not int or not 1 <= self.max_iterations <= 10000:
            raise ValueError('invalid CG iteration budget')
        if type(self.max_dofs) is not int or not 1 <= self.max_dofs <= 5000000:
            raise ValueError('invalid iterative degree-of-freedom limit')
        if not math.isfinite(self.max_workspace_mib) or not 0 < self.max_workspace_mib <= 2048:
            raise ValueError('invalid iterative workspace screen')


@dataclass
class IterativeResult:
    time_s: np.ndarray
    displacement_m: np.ndarray
    energy_time_s: np.ndarray
    energy_j: np.ndarray
    final_displacement_m: np.ndarray
    final_velocity_m_s: np.ndarray
    maximum_relative_residual: float
    iterations: np.ndarray
    workspace_screen_mib: float


@dataclass
class CheckedCGResult:
    solution: np.ndarray
    iterations: int
    relative_residual: float


def checked_cg(operator, rhs, guess, diagonal, settings=None, *, guard=None, progress=None):
    """Solve an SPD equation with strict status and a fresh final residual.

    Optional progress contains iteration counts/norms, not residual estimates.
    Only the final operator application provides a checked relative residual.
    """
    settings = settings or IterativeSettings()
    operator = aslinearoperator(operator)
    rhs, guess, diagonal = (np.asarray(value) for value in (rhs,guess,diagonal))
    n = operator.shape[0]
    if (operator.shape != (n,n) or not 1 <= n <= settings.max_dofs
            or operator.dtype is None or np.issubdtype(operator.dtype,np.complexfloating)
            or any(np.iscomplexobj(value) or value.shape != (n,) or not np.isfinite(value).all()
                   for value in (rhs,guess,diagonal)) or np.any(diagonal <= 0)
            or (progress is not None and not callable(progress))):
        raise ValueError('CG needs bounded real operators/vectors and a positive diagonal')
    if 12*n*8/1024**2 > settings.max_workspace_mib:
        raise MemoryError('CG vector workspace exceeds its declared screen')
    count = 0
    if guard:
        guard.check()
    def emit(**fields):
        if progress:
            progress(fields)
    emit(phase='solve_start',rhs_norm=float(np.linalg.norm(rhs)),relative_tolerance=settings.relative_tolerance)
    def counted(values):
        nonlocal count
        count += 1
        if progress:
            emit(phase='cg_iteration',iteration=count,iterate_norm=float(np.linalg.norm(values)))
        if guard:
            guard.check()
    preconditioner = LinearOperator((n,n),matvec=lambda x:x/diagonal,dtype=float)
    answer,info = cg(operator,rhs,x0=guess.copy(),M=preconditioner,rtol=settings.relative_tolerance,
        atol=0.,maxiter=settings.max_iterations,callback=counted)
    relative = float(np.linalg.norm(operator@answer-rhs)/max(float(np.linalg.norm(rhs)),1e-30))
    emit(phase='solve_end',iterations=count,status=int(info),
        true_relative_residual=relative if math.isfinite(relative) else None)
    if info != 0 or not math.isfinite(relative) or relative > settings.residual_limit or not np.isfinite(answer).all():
        raise ArithmeticError(f'CG solve failed: status={info}, iterations={count}, relative residual={relative:.6g}')
    return CheckedCGResult(answer,count,relative)


def iterative_simulate(stiffness, mass, mass_diagonal, steps, dt_s, probes, *, force=None,
                       initial_displacement=None, initial_velocity=None, output_stride=1,
                       stiffness_diagonal=None, settings=None, guard=None):
    settings = settings or IterativeSettings()
    positive('time step', dt_s)
    if (type(steps) is not int or not 1 <= steps <= 50000 or type(output_stride) is not int
            or not 1 <= output_stride <= 64 or steps % output_stride):
        raise ValueError('invalid iterative step count or output stride')
    k, m = aslinearoperator(stiffness), aslinearoperator(mass)
    n = k.shape[0]
    if k.shape != (n,n) or m.shape != (n,n) or not 1 <= n <= settings.max_dofs:
        raise ValueError('iterative operators must be matching bounded square matrices')
    if any(operator.dtype is None or np.issubdtype(operator.dtype, np.complexfloating) for operator in (k,m)):
        raise ValueError('iterative elastic operators require real dtypes')
    probes, diagonal = np.asarray(probes), np.asarray(mass_diagonal)
    if (np.iscomplexobj(probes) or probes.ndim != 2 or probes.shape[1] != n or not 1 <= len(probes) <= 32
            or not np.isfinite(probes).all() or np.iscomplexobj(diagonal) or diagonal.shape != (n,)
            or not np.isfinite(diagonal).all() or np.any(diagonal <= 0)):
        raise ValueError('invalid probes or positive mass diagonal')
    if stiffness_diagonal is not None:
        stiffness_diagonal = np.asarray(stiffness_diagonal)
        if (np.iscomplexobj(stiffness_diagonal) or stiffness_diagonal.shape != (n,)
                or not np.isfinite(stiffness_diagonal).all() or np.any(stiffness_diagonal < 0)):
            raise ValueError('stiffness diagonal must be finite and nonnegative')
    samples_count = steps//output_stride+1
    workspace_mib = ((32+len(probes))*n*8 + samples_count*len(probes)*8 + (steps+1)*12)/1024**2
    if workspace_mib > settings.max_workspace_mib or samples_count*len(probes)*8 > 16*1024**2:
        raise MemoryError('iterative state/probe storage exceeds its declared screen')
    if guard:
        guard.check(force=True)
    probes, diagonal = probes.astype(float, copy=False), diagonal.astype(float, copy=False)
    def vector(value):
        if value is None:
            return np.zeros(n)
        result = np.asarray(value)
        if np.iscomplexobj(result) or result.shape != (n,) or not np.isfinite(result).all():
            raise ValueError('iterative states and forces must be finite real vectors')
        return result.astype(float, copy=True)
    u, v = vector(initial_displacement), vector(initial_velocity)
    effective_diagonal = diagonal if stiffness_diagonal is None else diagonal+.25*dt_s**2*stiffness_diagonal
    effective = m + (.25*dt_s**2)*k
    maximum_residual = 0.
    def solve(operator, rhs, guess, operator_diagonal):
        nonlocal maximum_residual
        result = checked_cg(operator,rhs,guess,operator_diagonal,settings,guard=guard)
        maximum_residual = max(maximum_residual,result.relative_residual)
        return result.solution,result.iterations
    load = vector(force(0.) if force else None)
    initial_rhs = load-k @ u
    acceleration, initial_iterations = solve(m, initial_rhs, np.zeros(n),diagonal) if np.any(initial_rhs) else (np.zeros(n), 0)
    samples, energy = np.empty((samples_count,len(probes))), np.empty(steps+1)
    iterations = np.empty(steps+1, dtype=np.int32)
    def physical_energy():
        return float((v @ (m @ v) + u @ (k @ u))/2)
    samples[0], energy[0], iterations[0] = probes @ u, physical_energy(), initial_iterations
    for step in range(1,steps+1):
        if guard:
            guard.check()
        predicted = u+dt_s*v+.25*dt_s**2*acceleration
        load = vector(force(step*dt_s) if force else None)
        following, iterations[step] = solve(effective, load-k @ predicted, acceleration,effective_diagonal)
        u = predicted+.25*dt_s**2*following
        v += .5*dt_s*(acceleration+following)
        acceleration = following
        energy[step] = physical_energy()
        if step % output_stride == 0:
            samples[step//output_stride] = probes @ u
    if not np.isfinite(samples).all() or not np.isfinite(energy).all():
        raise ArithmeticError('iterative integration produced nonfinite traces or energy')
    if guard:
        guard.check(force=True)
    return IterativeResult(np.arange(samples_count)*dt_s*output_stride, samples,
        np.arange(steps+1)*dt_s, energy, u, v, maximum_residual, iterations, workspace_mib)
