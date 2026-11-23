"""Exact-SVD eigensystem realization (ERA) for the saved discrete PZT response.

Juang & Pappa, NASA 1985: H0=U S Vt; A=S^-1/2 U.T H1 V S^-1/2;
B=S^1/2 Vt[:, :1]; C=U[:outputs, :] S^1/2. No waveform gain/phase fitting.
This reduces the fixed input-output dynamics, not the full displacement field.
"""
import math
import time

import numpy as np
from scipy.linalg import eigvals
from scipy.linalg.blas import dgemv


def kernel_data(time_s, voltage):
    t, h = np.asarray(time_s), np.asarray(voltage)
    if (np.iscomplexobj(t) or np.iscomplexobj(h) or t.ndim != 1 or len(t) < 6
            or h.ndim != 2 or h.shape[0] != len(t) or not 1 <= h.shape[1] <= 32
            or not np.isfinite(t).all() or not np.isfinite(h).all() or t[0] != 0
            or np.any(h[:2] != 0.)):
        raise ValueError('Finite, real zero-start interior-impulse recording required')
    dt = float(t[1] - t[0])
    if dt <= 0 or not np.allclose(np.diff(t), dt, rtol=1e-9, atol=1e-15):
        raise ValueError('Uniform increasing internal recording grid required')
    # h[2] is CB, h[3] is CAB. The drive impulse was applied at solver node 1.
    return h[2:].astype(np.float64, copy=False), dt


def dimensions(samples, outputs=2):
    if type(samples) is not int or samples < 6 or type(outputs) is not int or outputs < 1:
        raise ValueError('Valid recording dimensions required')
    markov_count = samples - 2
    rows = markov_count // 2
    columns = markov_count - rows
    matrix_bytes = rows * outputs * columns * 8
    return dict(block_rows=rows, block_columns=columns, outputs=outputs,
        hankel_shape=[rows * outputs, columns], one_hankel_bytes=matrix_bytes,
        conservative_device_workspace_bytes=7 * matrix_bytes,
        conservative_host_workspace_bytes=3 * matrix_bytes,
        estimate_is_not_a_solver_allocation_guarantee=True)


def hankel(markov, rows, columns, shift=0):
    markov = np.asarray(markov)
    if (markov.ndim != 2 or np.iscomplexobj(markov) or not np.isfinite(markov).all()
            or type(rows) is not int or type(columns) is not int or min(rows, columns) < 1
            or shift not in (0, 1) or rows + columns + shift - 1 > len(markov)):
        raise ValueError('Insufficient real Markov parameters for requested Hankel matrix')
    outputs = markov.shape[1]
    result = np.empty((rows * outputs, columns), dtype=np.float64)
    for output in range(outputs):
        windows = np.lib.stride_tricks.sliding_window_view(markov[shift:, output], columns)
        result[output::outputs] = windows[:rows]
    return result


def fit_era(time_s, voltage, max_order, *, rows=None, columns=None, backend='cupy', event=None):
    """Fit nested balanced realizations. Production build uses exact FP64 GPU SVD.

    No decimation, randomized SVD, tail window, damping, or pole stabilization.
    The dense full-record factorization is expensive and its cost is reported.
    """
    started = time.perf_counter()
    markov, dt = kernel_data(time_s, voltage)
    screen = dimensions(len(time_s), markov.shape[1])
    rows = screen['block_rows'] if rows is None else rows
    columns = screen['block_columns'] if columns is None else columns
    if (type(rows) is not int or type(columns) is not int or rows < 1 or columns < 1
            or rows + columns > len(markov) or type(max_order) is not int
            or not 1 <= max_order <= min(rows * markov.shape[1], columns)):
        raise ValueError('Invalid ERA dimensions or maximum order')
    actual_bytes = rows * markov.shape[1] * columns * 8
    if backend not in ('numpy', 'cupy'):
        raise ValueError('Only exact numpy/cupy FP64 SVD is supported')
    if backend == 'cupy':
        import cupy as xp
        import psutil
        free, total = xp.cuda.runtime.memGetInfo()
        if 7 * actual_bytes > free or 3 * actual_bytes > psutil.virtual_memory().available:
            raise MemoryError('Declared exact-SVD workspace does not fit; no silent rank or sampling reduction')
    else:
        xp = np
        # CPU is the explicit small-system verification backend. A full GPU
        # study is never silently redirected to the local CPU.
        if actual_bytes > 128 * 1024**2:
            raise MemoryError('CPU verification backend is limited to small exact-SVD problems')
    def note(phase, **data):
        if event:
            event(phase, **data)
    note('assembling_hankel_0', shape=[rows * markov.shape[1], columns], bytes=actual_bytes)
    host = hankel(markov, rows, columns)
    matrix = xp.asarray(host)
    if backend == 'cupy':
        del host
    note('exact_svd_started', shape=list(matrix.shape), maximum_order=max_order)
    if backend == 'cupy':
        from examples.pzt_era_cuda import exact_svd
        u, singular, vt = exact_svd(matrix, event=event)
        xp.cuda.get_current_stream().synchronize()
        spectrum = xp.asnumpy(singular)
    else:
        u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
        spectrum = singular.copy()
    if not np.isfinite(spectrum).all() or spectrum[0] <= 0:
        raise ArithmeticError('Hankel SVD has no finite observable response')
    cutoff = np.finfo(float).eps * max(matrix.shape) * spectrum[0]
    numerical_rank = int(np.count_nonzero(spectrum > cutoff))
    available = min(max_order, numerical_rank)
    if available < 1:
        raise ArithmeticError('No numerically resolvable ERA states')
    left = u[:, :available].copy()
    right = vt[:available].T.copy()
    values = singular[:available].copy()
    del matrix, u, vt, singular
    if backend == 'cupy':
        xp.get_default_memory_pool().free_all_blocks()
    note('assembling_hankel_1', available_order=available, numerical_rank=numerical_rank)
    host = hankel(markov, rows, columns, shift=1)
    shifted = xp.asarray(host)
    if backend == 'cupy':
        del host
    root = xp.sqrt(values)
    reduced = (left.T @ shifted) @ right
    reduced /= root[:, None]
    reduced /= root[None, :]
    inputs = root[:, None] * right[:1].T
    outputs = left[:markov.shape[1]] * root[None, :]
    if backend == 'cupy':
        a, b, c = [xp.asnumpy(value) for value in (reduced, inputs, outputs)]
        del left, right, values, shifted, root, reduced, inputs, outputs
        xp.get_default_memory_pool().free_all_blocks()
    else:
        a, b, c = reduced, inputs, outputs
    if not all(np.isfinite(value).all() for value in (a, b, c)):
        raise ArithmeticError('Nonfinite reduced realization')
    diagnostic = dict(method='NASA ERA, exact full-record FP64 SVD', backend=backend,
        svd_driver='cusolverDnXgesvd, 64-bit dimensions and byte workspaces' if backend == 'cupy' else 'numpy.linalg.svd',
        requested_maximum_order=max_order, available_order=available, numerical_rank=numerical_rank,
        singular_value_cutoff=float(cutoff), block_rows=rows, block_columns=columns,
        internal_dt_s=dt, build_seconds=time.perf_counter() - started,
        identification_input='Unit discrete force at original solver node 1',
        reduction_scope='Fixed actuator to fixed PZT receiver voltages; no full-field reconstruction',
        source_or_response_gain_fit=False, phase_fit=False, artificial_damping=False,
        time_decimation=False, randomized_svd=False, poles_modified=False,
        preserves_second_order_mechanical_structure=False,
        infinite_time_stability_or_energy_bound_claimed=False)
    return dict(a=a, b=b, c=c, singular_values=spectrum, dt_s=dt), diagnostic


def reduced_impulse(model, order, samples, *, event=None):
    """Evaluate h at every original internal time node using real FP64 recurrence."""
    matrices = [np.asarray(model[key]) for key in ('a', 'b', 'c')]
    a0, b0, c0 = matrices
    if (any(np.iscomplexobj(value) or not np.isfinite(value).all() for value in matrices)
            or a0.ndim != 2 or a0.shape[0] != a0.shape[1] or b0.shape != (len(a0), 1)
            or c0.ndim != 2 or c0.shape[1] != len(a0) or c0.shape[0] < 1):
        raise ValueError('Finite real compatible state-space matrices required')
    if (type(order) is not int or type(samples) is not int or samples < 6
            or not 1 <= order <= len(model['a'])):
        raise ValueError('Valid reduced order and recording length required')
    a = np.array(model['a'][:order, :order], dtype=np.float64, order='F')
    b = np.array(model['b'][:order, 0], dtype=np.float64)
    c = np.array(model['c'][:, :order], dtype=np.float64)
    if not all(np.isfinite(value).all() for value in (a, b, c)):
        raise ValueError('Finite real state-space matrices required')
    poles = eigvals(a, check_finite=False)
    radius = float(np.max(abs(poles)))
    log_growth = max(0., math.log(radius)) * (samples - 2) if radius > 0 else 0.
    if not np.isfinite(radius) or log_growth > math.log(1.01):
        raise ArithmeticError(f'Reduced poles exceed 1% exponential growth over the recording: radius={radius:.12g}; not clipped')
    output = np.zeros((samples, len(c)), dtype=np.float64)
    state, following = b, np.empty_like(b)
    started, last = time.perf_counter(), -math.inf
    for node in range(2, samples):
        output[node] = c @ state
        following = dgemv(1., a, state, beta=0., y=following, overwrite_y=1)
        state, following = following, state
        if event and (node == samples - 1 or time.perf_counter() - last >= 15):
            event('reduced_propagation', steps_completed=node, requested_steps=samples - 1,
                  elapsed_seconds=time.perf_counter() - started)
            last = time.perf_counter()
    if not np.isfinite(output).all():
        raise ArithmeticError('Nonfinite reduced impulse response')
    spectrum = np.asarray(model['singular_values'])
    if (spectrum.ndim != 1 or np.iscomplexobj(spectrum) or not np.isfinite(spectrum).all()
            or np.any(spectrum < 0)):
        raise ValueError('Finite nonnegative Hankel singular spectrum required')
    denominator = float(np.linalg.norm(spectrum))
    if 'hankel_frobenius_norm' in model:
        denominator = float(model['hankel_frobenius_norm'])
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError('Positive finite full Hankel norm required')
        tail = math.sqrt(max(0., denominator**2 - float(np.dot(spectrum[:order], spectrum[:order])))) / denominator
    else:
        tail = float(np.linalg.norm(spectrum[order:]) / denominator) if denominator else None
    return output, dict(order=order, spectral_radius=radius, maximum_pole_growth_factor=math.exp(log_growth),
        hankel_relative_frobenius_tail=tail,
        tail_uses_partial_singular_spectrum=bool(model.get('singular_spectrum_is_partial', False)),
        impulse_seconds=time.perf_counter() - started, state_precision='float64',
        integration='Discrete ERA recurrence on the original internal time grid; no new integrator',
        poles_modified=False)
