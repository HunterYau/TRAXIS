"""Average-acceleration Newmark for validated symmetric elastic operators.

The assembled mass must be positive definite and stiffness positive
semidefinite. Physical construction/independent operator checks establish
those properties; symmetry and positive mass diagonal are checked here.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from traxis.config import positive


@dataclass
class ImplicitResult:
    time_s: np.ndarray
    displacement_m: np.ndarray
    energy_time_s: np.ndarray
    energy_j: np.ndarray
    final_displacement_m: np.ndarray
    final_velocity_m_s: np.ndarray
    maximum_relative_residual: float
    factor_nonzeros: int


def implicit_simulate(stiffness, mass, steps: int, dt_s: float, probes, *, force=None,
                      initial_displacement=None, initial_velocity=None, output_stride=1, guard=None):
    positive('time step', dt_s)
    if (type(steps) is not int or not 1 <= steps <= 50000 or type(output_stride) is not int
            or not 1 <= output_stride <= 64 or steps % output_stride):
        raise ValueError('invalid internal steps or output stride')
    if any(np.iscomplexobj(value) for value in (stiffness, mass)):
        raise ValueError('implicit time integration requires real operators')
    stiffness, mass = [sparse.csr_matrix(value, dtype=float) for value in (stiffness, mass)]
    ndof = stiffness.shape[0]
    if not 1 <= ndof <= 60000 or mass.shape != (ndof, ndof) or stiffness.shape != mass.shape:
        raise ValueError('need matching bounded square operators')
    for matrix in (stiffness, mass):
        if not np.isfinite(matrix.data).all() or matrix.nnz > 4000000:
            raise ValueError('operator entries exceed bounds or contain invalid values')
        error = abs(matrix - matrix.T).max()
        scale = abs(matrix).max()
        if error > 1e-11 * max(float(scale), 1e-30):
            raise ValueError('implicit elastic operators must be symmetric')
    if (mass.diagonal() <= 0).any():
        raise ValueError('mass diagonal must be positive')
    probes = np.asarray(probes, dtype=float)
    if (probes.ndim != 2 or probes.shape[1] != ndof or not 1 <= len(probes) <= 32
            or not np.isfinite(probes).all() or (steps // output_stride + 1) * len(probes) * 8 > 16 * 1024**2):
        raise ValueError('probe shape/values/storage exceed bounds')
    def vector(value):
        result = np.zeros(ndof) if value is None else np.asarray(value, dtype=float).copy()
        if result.shape != (ndof,) or not np.isfinite(result).all():
            raise ValueError('state/force must be a finite node-major vector')
        return result
    displacement, velocity = vector(initial_displacement), vector(initial_velocity)
    if guard:
        guard.check(force=True)
    # The direct factorization is kept for every internal step. Callers run
    # larger systems in a monitored worker so native allocations can be stopped.
    matrix = (mass + .25 * dt_s**2 * stiffness).tocsc()
    factor = splu(matrix, permc_spec='MMD_AT_PLUS_A')
    if guard:
        guard.check(force=True)
    load = vector(force(0.) if force else None)
    residual = load - stiffness @ displacement
    if np.any(residual):
        acceleration = splu(mass.tocsc(), permc_spec='MMD_AT_PLUS_A').solve(residual)
    else:
        acceleration = np.zeros(ndof)
    samples = np.empty((steps // output_stride + 1, len(probes)))
    energy = np.empty(steps + 1)
    def physical_energy():
        return float((velocity @ (mass @ velocity) + displacement @ (stiffness @ displacement)) / 2)
    samples[0], energy[0] = probes @ displacement, physical_energy()
    maximum_residual = 0.
    for step in range(1, steps + 1):
        if guard:
            guard.check()
        predicted = displacement + dt_s * velocity + .25 * dt_s**2 * acceleration
        load = vector(force(step * dt_s) if force else None)
        rhs = load - stiffness @ predicted
        following = factor.solve(rhs)
        measured = float(np.linalg.norm(matrix @ following - rhs) / max(float(np.linalg.norm(rhs)), 1e-30))
        if not math.isfinite(measured) or measured > 1e-9:
            raise ArithmeticError('implicit linear solve residual exceeds tolerance')
        maximum_residual = max(maximum_residual, measured)
        displacement = predicted + .25 * dt_s**2 * following
        velocity += .5 * dt_s * (acceleration + following)
        acceleration = following
        energy[step] = physical_energy()
        if step % output_stride == 0:
            samples[step // output_stride] = probes @ displacement
    if not np.isfinite(samples).all() or not np.isfinite(energy).all():
        raise ArithmeticError('implicit simulation produced invalid states/probes')
    if guard:
        guard.check(force=True)
    return ImplicitResult(np.arange(steps // output_stride + 1) * dt_s * output_stride, samples,
                           np.arange(steps + 1) * dt_s, energy, displacement, velocity,
                           maximum_residual, factor.L.nnz + factor.U.nnz)
