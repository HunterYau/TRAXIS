"""ERA with FFT Hankel products and PROPACK partial SVD; no dense Hankel table.

FFT Hankel matvec: Minster et al., arXiv:2003.11872, section 2.2.
Partial SVD: scipy.sparse.linalg.svds(solver='propack'). Physics is unchanged.
"""
import math
import time

import numpy as np
import psutil
from scipy.fft import next_fast_len
from scipy.sparse.linalg import LinearOperator, svds

from examples.pzt_era import kernel_data, dimensions

SVD_TOLERANCE = 1e-12
RESIDUAL_TOLERANCE = 1e-9
BLOCK = 8
SEED = 1729


def memory_estimate(samples, outputs=2, max_order=2048):
    shape = dimensions(samples, outputs)
    rows, columns = shape['block_rows'], shape['block_columns']
    m, n = rows * outputs, columns
    if type(max_order) is not int or not 1 <= max_order <= min(m, n):
        raise ValueError('Valid maximum order required')
    kmax = min(m + 1, n + 1, 2 * max_order + 32)
    fft_length = next_fast_len(rows + columns + max(rows, columns) - 2)
    # PROPACK's explicitly bounded u/v bases and real work array, plus sorting,
    # retained singular vectors, block products, projection, and FFT allowance.
    solver = m * (kmax + 1) + n * kmax + 2 * kmax**2 + 9 * kmax + 2 * (m + n) + 8
    host = 8 * (solver + 2 * (m + n) * max_order + 6 * max_order**2)
    fft_workspace = 256 * 1024**2 + 64 * fft_length * BLOCK
    return dict(block_rows=rows, block_columns=columns, outputs=outputs,
        hankel_shape=[m, n], dense_hankel_allocated=False,
        equivalent_one_dense_hankel_bytes=shape['one_hankel_bytes'],
        maximum_order=max_order, krylov_dimension=kmax, fft_length=fft_length,
        estimated_host_workspace_bytes=host + fft_workspace,
        estimated_device_workspace_bytes=fft_workspace,
        host_solver='PROPACK Lanczos bidiagonalization', fft_batch_columns=BLOCK,
        estimate_is_not_an_allocation_guarantee=True)


class HankelOperator(LinearOperator):
    """Real block-Hankel H[i*p+o,j]=markov[i+j+shift,o], via linear FFT convolution."""
    def __init__(self, markov, rows, columns, *, shift=0, backend='numpy', event=None):
        data = np.asarray(markov)
        if (data.ndim != 2 or np.iscomplexobj(data) or not np.isfinite(data).all()
                or data.shape[1] < 1 or type(rows) is not int or type(columns) is not int
                or min(rows, columns) < 1 or shift not in (0, 1)
                or rows + columns + shift - 1 > len(data) or backend not in ('numpy', 'cupy')):
            raise ValueError('Finite real Markov data and compatible Hankel dimensions required')
        if backend == 'cupy':
            import cupy as xp
        else:
            xp = np
        self.xp, self.backend = xp, backend
        self.rows, self.columns, self.outputs = rows, columns, data.shape[1]
        self.fft_length = next_fast_len(rows + columns + max(rows, columns) - 2)
        self.kernel = xp.fft.rfft(xp.asarray(data[shift:shift + rows + columns - 1]),
                                  n=self.fft_length, axis=0)
        self.products, self.event, self.last = 0, event, -math.inf
        super().__init__(dtype=np.dtype('float64'), shape=(rows * self.outputs, columns))

    def _apply(self, value, transpose=False):
        x = np.asarray(value)
        expected = self.shape[0] if transpose else self.shape[1]
        if x.ndim == 1:
            x = x[:, None]
        if (x.ndim != 2 or x.shape[0] != expected or np.iscomplexobj(x)
                or not np.isfinite(x).all()):
            raise ValueError('Compatible finite real operator input required')
        xp, r, c, p = self.xp, self.rows, self.columns, self.outputs
        result = np.empty((self.shape[1] if transpose else self.shape[0], x.shape[1]), dtype=np.float64)
        for start in range(0, x.shape[1], BLOCK):
            stop = min(start + BLOCK, x.shape[1])
            batch = xp.asarray(x[:, start:stop], dtype=xp.float64)
            if transpose:
                product = None
                for output in range(p):
                    transformed = xp.fft.rfft(batch[output::p][::-1], n=self.fft_length, axis=0)
                    term = self.kernel[:, output, None] * transformed
                    product = term if product is None else product + term
                answer = xp.fft.irfft(product, n=self.fft_length, axis=0)[r - 1:r - 1 + c]
            else:
                transformed = xp.fft.rfft(batch[::-1], n=self.fft_length, axis=0)
                answer = xp.empty((r * p, stop - start), dtype=xp.float64)
                for output in range(p):
                    answer[output::p] = xp.fft.irfft(self.kernel[:, output, None] * transformed,
                        n=self.fft_length, axis=0)[c - 1:c - 1 + r]
            result[:, start:stop] = xp.asnumpy(answer) if self.backend == 'cupy' else answer
        self.products += x.shape[1]
        if self.event and time.perf_counter() - self.last >= 15:
            self.event('matrix_free_hankel_products', vectors_processed=self.products,
                       transpose=transpose, dense_hankel_allocated=False)
            self.last = time.perf_counter()
        if not np.isfinite(result).all():
            raise ArithmeticError('Nonfinite FFT Hankel product')
        return result

    def _matvec(self, value):
        return self._apply(value)[:, 0]

    def _rmatvec(self, value):
        return self._apply(value, True)[:, 0]

    def _matmat(self, value):
        return self._apply(value)

    def _rmatmat(self, value):
        return self._apply(value, True)


def frobenius_norm(markov, rows, columns):
    # Counts each anti-diagonal's repetitions without allocating H itself.
    k = np.arange(rows + columns - 1)
    count = np.minimum(np.minimum(k + 1, rows), np.minimum(columns, rows + columns - 1 - k))
    return float(np.sqrt(np.dot(count, np.sum(np.asarray(markov)[:len(k)]**2, axis=1))))


def fit_era(time_s, voltage, max_order, *, backend='cupy', event=None):
    """Leading ERA states from the full recording; fail rather than loosen accuracy."""
    started = time.perf_counter()
    markov, dt = kernel_data(time_s, voltage)
    if backend not in ('numpy', 'cupy'):
        raise ValueError('Explicit numpy or cupy FFT backend required')
    estimate = memory_estimate(len(time_s), markov.shape[1], max_order)
    if estimate['estimated_host_workspace_bytes'] > psutil.virtual_memory().available:
        raise MemoryError('Estimated matrix-free host workspace does not fit; no order reduction')
    if backend == 'cupy':
        import cupy as cp
        if estimate['estimated_device_workspace_bytes'] > cp.cuda.runtime.memGetInfo()[0]:
            raise MemoryError('Estimated matrix-free FFT workspace does not fit')
    rows, columns = estimate['block_rows'], estimate['block_columns']
    def note(phase, **values):
        if event:
            event(phase, **values)
    op = HankelOperator(markov, rows, columns, backend=backend, event=event)
    note('matrix_free_partial_svd_started', maximum_order=max_order,
         krylov_dimension=estimate['krylov_dimension'], dense_hankel_allocated=False)
    u, singular, vt = svds(op, k=max_order, solver='propack', which='LM',
        tol=SVD_TOLERANCE, maxiter=estimate['krylov_dimension'], rng=SEED,
        v0=np.random.default_rng(SEED).standard_normal(op.shape[0]))
    indices = np.argsort(singular)[::-1]
    singular, u, v = singular[indices], u[:, indices], vt[indices].T
    del vt
    if not np.isfinite(singular).all() or singular[0] <= 0:
        raise ArithmeticError('No finite observable partial singular spectrum')
    cutoff = np.finfo(float).eps * max(op.shape) * singular[0]
    order = int(np.count_nonzero(singular > cutoff))
    if order < 1:
        raise ArithmeticError('No numerically resolvable ERA states')
    u, v, singular = u[:, :order], v[:, :order], singular[:order]
    # Both singular equations and orthogonality are checked independently of
    # PROPACK's stopping decision. No unconverged results or dense fallback.
    residual = 0.
    for start in range(0, order, BLOCK):
        end = min(start + BLOCK, order)
        first = op @ v[:, start:end] - u[:, start:end] * singular[start:end]
        second = op.T @ u[:, start:end] - v[:, start:end] * singular[start:end]
        residual = max(residual, float(np.linalg.norm(first, axis=0).max() / singular[0]),
                       float(np.linalg.norm(second, axis=0).max() / singular[0]))
    orthogonality = max(float(np.max(abs(u.T @ u - np.eye(order)))),
                        float(np.max(abs(v.T @ v - np.eye(order)))))
    if residual > RESIDUAL_TOLERANCE or orthogonality > 1e-7:
        raise ArithmeticError(f'Partial SVD failed verification: residual={residual:g}, orthogonality={orthogonality:g}')
    shifted = HankelOperator(markov, rows, columns, shift=1, backend=backend)
    root = np.sqrt(singular)
    a = np.empty((order, order), dtype=np.float64)
    note('projecting_shifted_hankel_in_blocks', available_order=order, block_columns=BLOCK)
    for start in range(0, order, BLOCK):
        end = min(start + BLOCK, order)
        a[:, start:end] = (u.T @ (shifted @ v[:, start:end])) / root[:, None] / root[None, start:end]
    b, c = root[:, None] * v[:1].T, u[:markov.shape[1]] * root[None, :]
    if not all(np.isfinite(value).all() for value in (a, b, c)):
        raise ArithmeticError('Nonfinite matrix-free ERA realization')
    norm = frobenius_norm(markov, rows, columns)
    model = dict(a=a, b=b, c=c, singular_values=singular, dt_s=dt,
                 hankel_frobenius_norm=norm, singular_spectrum_is_partial=True)
    diagnostic = dict(method='ERA, matrix-free FFT Hankel products and PROPACK partial SVD',
        backend=backend, host_solver='scipy PROPACK', randomized_svd=False,
        requested_maximum_order=max_order, available_order=order,
        resolved_retained_singular_values=order, full_numerical_rank_known=False,
        singular_spectrum_is_partial=True, singular_value_cutoff=float(cutoff),
        internal_dt_s=dt, build_seconds=time.perf_counter() - started,
        svd_tolerance=SVD_TOLERANCE, verified_svd_residual_over_largest_singular_value=residual,
        verified_orthogonality_error=orthogonality, starting_vector_seed=SEED,
        memory_estimate=estimate, source_or_response_gain_fit=False, phase_fit=False,
        time_decimation=False, poles_modified=False, artificial_damping=False,
        dense_hankel_allocated=False, hankel_frobenius_norm=norm,
        preserves_second_order_mechanical_structure=False,
        infinite_time_stability_or_energy_bound_claimed=False,
        reduction_scope='Fixed actuator to fixed PZT receiver voltages; no full-field reconstruction')
    return model, diagnostic
