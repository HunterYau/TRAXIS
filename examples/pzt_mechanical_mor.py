"""One-sided mechanical Krylov/Galerkin reduction: Mr=V'MV, Kr=V'KV.

Same trial/test basis preserves symmetric positive mass and nonnegative
stiffness. No ERA pole edits, fitted damping, or physical parameter changes.
Reference: Beattie/Gugercin, CDC-ECC 2005, Krylov-Based Model Reduction of
Second-Order Systems with Proportional Damping; arXiv:2010.06331 survey.
"""
import math
import time

import numpy as np
import psutil
from scipy.linalg import eigh


def memory_estimate(ndof, maximum_order):
    if type(ndof) is not int or type(maximum_order) is not int or not 1 <= maximum_order <= ndof:
        raise ValueError('Valid full dimension and requested projection order required')
    # GPU-resident V and MV, vectors, and allowance for BLAS/projection scratch.
    device = 8 * ndof * (2 * maximum_order + 20) + 64 * 1024**2
    return dict(full_dofs=ndof, maximum_order=maximum_order,
        extra_device_workspace_bytes=device, host_workspace_bytes=8 * ndof * 3 + 64 * 1024**2,
        estimate_is_not_an_allocation_guarantee=True,
        basis='full physical vectors; no mesh/time/precision reduction')


def project(ndof, load, observe, mass, inverse_mass, stiffness, maximum_order,
            *, xp=np, event=None):
    """Mass-orthonormal polynomial Krylov basis with two-pass reorthogonalization.

    All operators act on vectors; full mass/stiffness matrices are never built.
    observe(x) must return the actual displacement-to-voltage measurements.
    """
    started = time.perf_counter()
    estimate = memory_estimate(ndof, maximum_order)
    if xp is np:
        available = psutil.virtual_memory().available
    else:
        available = xp.cuda.runtime.memGetInfo()[0]
    if estimate['extra_device_workspace_bytes'] > available:
        raise MemoryError('Requested physical basis exceeds currently available memory')
    f = xp.asarray(load, dtype=xp.float64)
    if f.shape != (ndof,) or not bool(xp.isfinite(f).all()):
        raise ValueError('Compatible finite physical load required')
    v = xp.empty((ndof, maximum_order), dtype=xp.float64, order='F')
    mv = xp.empty_like(v)
    # Starting from M^-1 f produces the displacement/acceleration Krylov space.
    candidate = inverse_mass(f.copy())
    rank = 0
    for k in range(maximum_order):
        mc = mass(candidate)
        initial = float(xp.sqrt(xp.dot(candidate, mc)).item())
        if not math.isfinite(initial) or initial <= 0:
            raise ArithmeticError('Nonpositive Krylov mass norm')
        for _ in range(2):
            if k:
                coefficients = v[:, :k].T @ mc
                candidate -= v[:, :k] @ coefficients
                mc -= mv[:, :k] @ coefficients
        # Refresh the exact mass action after reorthogonalization.
        mc = mass(candidate)
        norm2 = float(xp.dot(candidate, mc).item())
        if not math.isfinite(norm2) or norm2 < 0:
            raise ArithmeticError('Invalid mass inner product after orthogonalization')
        if math.sqrt(norm2) <= 1e-12 * initial:
            if k == 0:
                raise ArithmeticError('Zero controllable basis')
            break  # Exact/numerical Krylov closure; never silently claim requested rank.
        norm = math.sqrt(norm2)
        v[:, k], mv[:, k] = candidate / norm, mc / norm
        rank = k + 1
        if event:
            event('physical_krylov_basis', steps_completed=rank, requested_steps=maximum_order,
                  elapsed_seconds=time.perf_counter() - started)
        if rank < maximum_order:
            candidate = inverse_mass(stiffness(v[:, k]))
    v, mv = v[:, :rank], mv[:, :rank]
    mr = v.T @ mv
    kr = xp.empty((rank, rank), dtype=xp.float64)
    cr = []
    for j in range(rank):
        kr[:, j] = v.T @ stiffness(v[:, j])
        cr.append(observe(v[:, j]))
        if event:
            event('physical_congruence_projection', steps_completed=j + 1, requested_steps=rank)
    br = v.T @ f
    cr = xp.stack(cr, axis=1)
    host = (lambda x: np.asarray(x)) if xp is np else xp.asnumpy
    model = {name: host(value) for name, value in [('mass', mr), ('stiffness', kr),
                                                 ('load', br), ('observe', cr)]}
    orth = float(np.linalg.norm(model['mass'] - np.eye(rank), ord=2))
    if orth > 1e-8:
        raise ArithmeticError(f'Mass-orthonormal basis verification failed: {orth:g}')
    return model, dict(method='Mass-inner-product Krylov, one-sided mechanical congruence',
        requested_order=maximum_order, available_order=rank, numerical_krylov_closure=rank < maximum_order,
        mass_orthogonality_error=orth, build_seconds=time.perf_counter() - started,
        full_fields_reconstructable=False, poles_clipped=False, damping_added=False,
        physical_parameter_changes=False, waveform_accuracy_requires_validation=True,
        memory_estimate=estimate)


def impulse(model, order, samples, dt_s, *, event=None):
    """Evaluate the SAME discrete velocity-Verlet impulse, via modal coordinates.

    Unit input at node 1; zero displacement at nodes 0/1, h[2]=dt^2 C M^-1 f.
    Eigenvalues are checked, never clipped. The selected subspace must have
    positive stiffness (free rigid modes are not claimed to have bounded motion).
    """
    if (type(order) is not int or type(samples) is not int or samples < 6
            or not 1 <= order <= len(model['load']) or not math.isfinite(dt_s) or dt_s <= 0):
        raise ValueError('Valid mechanical order, grid and timestep required')
    m = np.asarray(model['mass'])[:order, :order]
    k = np.asarray(model['stiffness'])[:order, :order]
    b = np.asarray(model['load'])[:order]
    c = np.asarray(model['observe'])[:, :order]
    if (m.shape != (order, order) or k.shape != m.shape or b.shape != (order,)
            or c.ndim != 2 or c.shape[1] != order
            or any(np.iscomplexobj(x) or not np.isfinite(x).all() for x in (m, k, b, c))):
        raise ValueError('Compatible finite real mechanical matrices required')
    for name, x in [('mass', m), ('stiffness', k)]:
        if np.linalg.norm(x - x.T) > 1e-9 * max(np.linalg.norm(x), np.finfo(float).tiny):
            raise ArithmeticError(f'Projected {name} is not symmetric')
    # Remove only skew roundoff after checking it, not eigenvalue signs.
    lam, modes = eigh((k + k.T) / 2, (m + m.T) / 2)
    if np.min(lam) <= 0:
        raise ArithmeticError('Projected stiffness is not positive; no eigenvalue clipping or rigid-mode hiding')
    if dt_s**2 * np.max(lam) >= 4:
        raise ArithmeticError('Original timestep violates the reduced Verlet stability condition')
    gamma = 2 - dt_s**2 * lam
    modal_load, modal_probe = modes.T @ b, c @ modes
    before, state = np.zeros(order), dt_s**2 * modal_load
    output = np.zeros((samples, c.shape[0]))
    started = time.perf_counter()
    for node in range(2, samples):
        output[node] = modal_probe @ state
        following = gamma * state - before
        before, state = state, following
        if event and (node % 8192 == 0 or node == samples - 1):
            event('mechanical_reduced_impulse', steps_completed=node, requested_steps=samples - 1)
    if not np.isfinite(output).all():
        raise ArithmeticError('Nonfinite mechanical reduced response')
    return output, dict(order=order, impulse_seconds=time.perf_counter() - started,
        minimum_eigenvalue_s2=float(lam[0]), maximum_eigenvalue_s2=float(lam[-1]),
        verlet_stability_margin=4 - float(dt_s**2 * lam[-1]),
        marginal_oscillatory_stability_verified=True, poles_clipped=False,
        damping_added=False, original_integration_grid=True, fitted_alignment=False)


def coupled_actions(storage, probes):
    """Exact adapter for the existing merged consistent-mass PZT storage."""
    cp = storage.cp
    def mass(x):
        result = storage.diagonal * x
        for ids, _, local_mass, _ in storage.patches:
            result[ids] = storage.matvec(local_mass, x[ids])
        return result
    def inverse(x):
        storage.inverse_mass_inplace(x)
        return x
    def stiffness(x):
        result = cp.empty_like(x)
        storage._launch(x, result, 1., 0.)
        return result
    ids = [cp.asarray(i) for i in probes.indices]
    weights = [cp.asarray(w) for w in probes.weights]
    def observe(x):
        return cp.stack([cp.dot(w, x[i]) for i, w in zip(ids, weights)])
    return mass, inverse, stiffness, observe


def build_plate(config, maximum_order, *, event=None):
    """Adapter to the actual plate and consistently coupled 3-D PZT operators."""
    from examples import resolved_pzt_queue as queue
    from examples.experimental_five import PeakGuard
    from traxis.resources import ResourceLimits
    from traxis.gpu_tensor import make_layout
    from traxis.config import Plate
    note = event or (lambda *args, **kwargs: None)
    if queue.read_setup(config) or config.get('damping_profile'):
        raise ValueError('Mechanical remedy currently supports the saved ideal open-port lossless configuration only')
    layout = make_layout(Plate(*config['plate_m']), tuple(config['mesh']['cells']), config['mesh']['order'],
                         max_dofs=32000000, max_device_mib=6144, max_host_mib=4096)
    # Body DOFs are added by queue.build; estimate conservatively using its
    # supported three bodies of at most 1024 nodes each.
    estimate = memory_estimate(int(layout.screen['vector_dofs']) + 3 * 1024 * 3, maximum_order)
    import cupy as cp
    free = cp.cuda.runtime.memGetInfo()[0]
    if estimate['extra_device_workspace_bytes'] + 6144 * 1024**2 > free:
        raise MemoryError('Full plate plus physical basis does not fit current GPU; mesh/order unchanged')
    if estimate['host_workspace_bytes'] + 4096 * 1024**2 > psutil.virtual_memory().available:
        raise MemoryError('Physical model host workspace unavailable')
    guard = PeakGuard(ResourceLimits(**{key: None for key in config['resource_limits']}),
                      telemetry_errors_fatal=False)
    system, load, probes, steps = queue.build(config, guard, note)
    storage = system.stiffness.storage
    mass, inverse, stiffness, observe = coupled_actions(storage, probes)
    with storage.lock, cp.cuda.Device(storage.device_id):
        model, info = project(len(load), load, observe, mass, inverse, stiffness,
                              maximum_order, xp=cp, event=note)
    info.update(full_model_stable_dt_s=system.stable_dt_s, required_full_steps=steps,
                consistent_body_mass_preserved=True, open_port_piezo_stiffness_preserved=True)
    return model, info
