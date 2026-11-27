"""Precompute exact sparse PZT actions for the existing dense GPU executor.

No changes to element assembly, interface maps, consistent mass or LTS. The
sparse implementation remains the reference. Electrical condensation and mass
inversion happen once in bounded column batches, rather than during stepping.
"""
from types import SimpleNamespace
import math
import numpy as np
import psutil
from scipy import sparse


def estimate(layout, geometry_estimate, block_columns=64):
    e = dict(geometry_estimate)
    sizes = e['patch_dofs']
    n = e['total_mechanical_dofs']
    dense = 3 * sum(q*q for q in sizes) * 8
    # One patch is assembled on the CPU, uploaded, then released. Device
    # storage retains all three matrices per patch plus the original buffers.
    temporary = max(3*q*q*8 + 8*(q + 3*e['body_nodes_each'])*block_columns*8
                    + (q//3)**2*8 for q in sizes)
    host = temporary + 6*n*8 + 1536*2**20
    storage = math.ceil(layout.screen['device_storage_screen_mib']*2**20) + dense + 3*n*8
    workspace = 32*n*8 + sum(sizes)*24 + e['total_mechanical_dofs'] + 4*layout.screen['elements'] + 32768*16 + 32769*2*8
    return dict(e, estimated_host_peak_bytes=math.ceil(1.15*host),
        estimated_gpu_peak_bytes=math.ceil(1.15*(storage+workspace)),
        attachment_workspace_bytes=temporary, merge_workspace_bytes=0,
        dense_pzt_device_bytes=dense, block_columns=block_columns,
        estimated_case_total_seconds_range=[None, None],
        estimated_primary_kernel_seconds_range=[None, None],
        estimate_note='FP64 precomputed K/M/M-inverse for three disjoint supports; one CPU patch at a time. Includes 15% planning margin. Runtime checks available memory; forecast is not measured peak.',
        timing_note='No full spectral precomputed benchmark. Setup time and full-run speed remain unmeasured.')


def matrices(attachment, mass, *, block_columns=64, max_workspace_mib=16384,
             progress=None):
    """Materialize the SAME K and full coupled M^-1 as sparse factor actions."""
    a = attachment
    n = len(a.indices)
    if type(block_columns) is not int or block_columns < 1:
        raise ValueError('Positive integer batch size required')
    needed = 3*n*n*8 + (n//3)**2*8 + 8*(n+3*a.data.n)*block_columns*8
    if needed > max_workspace_mib*2**20 or needed > .8*psutil.virtual_memory().available:
        raise MemoryError('Precomputed patch exceeds available host memory or workspace; use sparse fallback or a larger cloud host')
    # Pull: u_c = T x[positions[:,c]]. This is exactly SparseAttachment.pull,
    # expressed as one sparse map, preserving Cartesian and patch index order.
    t = a.t.tocoo()
    transform = sparse.coo_matrix((np.tile(t.data, 3),
        (np.concatenate([t.row+c*a.data.n for c in range(3)]),
         np.concatenate([a.positions[t.col, c] for c in range(3)]))),
        shape=(3*a.data.n, n)).tocsr()
    mechanical = (transform.T @ a.data.stiffness @ transform).tocsr()
    coupling = (transform.T @ a.data.bi).tocsr()
    k = np.empty((n,n), dtype=np.float64, order='C')
    for first in range(0, n, block_columns):
        last = min(first+block_columns, n)
        rhs = coupling[first:last].T.toarray()
        block = mechanical[:,first:last].toarray()
        block += coupling @ a.data.electrical_lu.solve(rhs)
        if a.electrical == 'open':
            block += a.coupling[:,None]*a.coupling[None,first:last]/a.total_capacitance_f
        k[:,first:last] = block
        if progress: progress('precomputing_stiffness', columns_completed=last, columns_total=n)
    del transform, mechanical, coupling
    m = np.zeros((n,n), dtype=np.float64, order='C')
    inv = np.zeros_like(m)
    scalar = mass.matrix.toarray()
    for c in range(3):
        ids = a.positions[:,c]
        m[np.ix_(ids,ids)] = scalar
    del scalar
    q = mass.matrix.shape[0]
    for first in range(0, q, block_columns):
        last = min(first+block_columns, q)
        rhs = np.zeros((q,last-first))
        rhs[np.arange(first,last),np.arange(last-first)] = 1.
        block = mass.factor.solve(rhs)
        for c in range(3):
            inv[np.ix_(a.positions[:,c],a.positions[first:last,c])] = block
        if progress: progress('precomputing_mass_inverse', columns_completed=last, columns_total=q)
    # Bounded randomized probes against the retained exact factor actions.
    errors = dict(stiffness=0.,mass=0.,mass_inverse=0.,mass_residual=0.)
    rng = np.random.default_rng(2941)
    for _ in range(3):
        x = rng.normal(size=n)
        for name, actual, reference in (
            ('stiffness',k@x,a.stiffness_action(x)),
            ('mass',m@x,mass.action(x)),
            ('mass_inverse',inv@x,mass.solve(x)),
            ('mass_residual',mass.action(inv@x),x)):
            err = float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference),1e-300))
            if not math.isfinite(err): raise ArithmeticError('Nonfinite dense PZT verification')
            errors[name] = max(errors[name],err)
    if max(errors.values()) > 1e-10:
        raise ArithmeticError('Precomputed PZT action verification failed: '+str(errors))
    return SimpleNamespace(indices=a.indices, stiffness=k, mass=m, inverse=inv,
        inverse_error=errors['mass_residual'], verification=errors)
