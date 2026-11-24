"""Lossless LTS-LF2(p), Grote & Mitkova, arXiv:1205.0654, section 3.1.

Coordinates may have a block mass if the fine projector contains whole mass
blocks. ``action`` is M^-1 K and ``local_action`` is exactly M^-1 K P,
including all output/interface rows. Never freeze interface forces.
"""
import math
import numpy as np
from scipy.signal import fftconvolve


def lf2_step(previous, current, coarse_dt, substeps, action, local_action,
             fine, load, source, time_s):
    """Published LTS-LF2 recurrence with time-symmetric fine forcing.

    Source samples at t +/- m*h are required. This is not ordinary independent
    regional Verlet. Fine/coarse projection acts on INPUT columns, not outputs.
    ``load`` must be wholly fine; sources outside it are rejected by adapters.
    """
    if (not math.isfinite(coarse_dt) or coarse_dt <= 0
            or type(substeps) is not int or not 1 <= substeps <= 16):
        raise ValueError('Positive coarse dt and integer substeps in 1..16 required')
    h = coarse_dt / substeps
    coarse = current.copy()
    coarse[fine] = 0
    w = -action(coarse)
    q0 = 2 * current
    q1 = q0 + h*h * (w + load*source(time_s) - .5*local_action(q0))
    for m in range(1, substeps):
        q2 = 2*q1-q0+h*h*(2*w + load*(source(time_s+m*h)+source(time_s-m*h))
                           - local_action(q1))
        q0, q1 = q1, q2
    return q1-previous


def polyphase_reconstruct(kernels, fine_source):
    """Exact discrete multirate superposition; no waveform alignment or fit.

    kernels[r] is the zero-state response to a unit input at fine index r+1.
    Output is on coarse nodes; the input is on fine nodes and starts at zero.
    Each input phase needs its own kernel because LTS is periodic on that grid.
    """
    k, f = np.asarray(kernels), np.asarray(fine_source)
    if (k.ndim != 3 or f.ndim != 1 or np.iscomplexobj(k) or np.iscomplexobj(f)
            or not np.isfinite(k).all() or not np.isfinite(f).all()
            or not 1 <= k.shape[0] <= 16 or k.shape[1] < 3
            or len(f) != (k.shape[1]-1)*k.shape[0]+1 or f[0] != 0
            or np.any(k[:, 0] != 0)):
        raise ValueError('Finite phase kernels and zero-start fine-grid source required')
    result = np.zeros_like(k[0], dtype=float)
    for r in range(k.shape[0]):
        result += fftconvolve(k[r], f[r+1::k.shape[0], None], axes=0)[:len(result)]
    return result


def fine_partition(system, load, boxes=(), edge_width_m=0.):
    """Fine PZT blocks, drive support, optional physical boxes and edge strip.

    Mass closure is essential: component-wise mass lumping is not substituted.
    Each box is [xmin,xmax,ymin,ymax,zmin,zmax] in physical metres.
    """
    n = len(system.mass_diagonal)
    fine = np.asarray(load) != 0
    if fine.shape != (n,) or not math.isfinite(edge_width_m) or edge_width_m < 0:
        raise ValueError('Matching load and nonnegative edge width required')
    layout = system.layout
    ns = layout.screen['scalar_dofs']
    if boxes or edge_width_m:
        # Bounded 1D axis broadcasting, no full (N,3) coordinate table.
        x,y,z = layout.axes_m
        if not np.array_equal(layout.transform, np.eye(3)):
            raise ValueError('Physical box selection currently requires aligned axes')
        selected = np.zeros((len(x),len(y),len(z)), dtype=bool)
        for box in boxes:
            b = np.asarray(box, dtype=float)
            if b.shape != (6,) or not np.isfinite(b).all() or np.any(b[::2] > b[1::2]):
                raise ValueError('Each box needs finite ordered coordinate bounds')
            selected |= ((x[:,None,None]>=b[0]) & (x[:,None,None]<=b[1])
                & (y[None,:,None]>=b[2]) & (y[None,:,None]<=b[3])
                & (z[None,None,:]>=b[4]) & (z[None,None,:]<=b[5]))
        if edge_width_m:
            selected |= ((x[:,None,None]<=edge_width_m) | (x[:,None,None]>=x[-1]-edge_width_m)
                | (y[None,:,None]<=edge_width_m) | (y[None,:,None]>=y[-1]-edge_width_m))
        for c in range(3): fine[c*ns:(c+1)*ns] |= selected.ravel()
    for patch in getattr(system, 'body_patches', ()):
        fine[patch.indices] = True
    return fine
