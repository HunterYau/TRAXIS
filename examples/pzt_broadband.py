"""Causal discrete impulse reconstruction for the existing linear Verlet solver.

No division by a finite-pulse spectrum, frequency cutoff, gain fit, or time fit.
The kernel is the voltage response to F[1]=1, all other F[n]=0; F[0]=0.
For a zero-start target, y[n] = sum(j>=0) kernel[n-j] * F[j+1].
The one-step offset avoids Verlet's half-weight initial forcing at n=0.
"""
import math
import numpy as np
from scipy.signal import fftconvolve, hilbert

from examples.pzt_convergence import metrics
from traxis.signal_comparison import waveform_metrics


def interior_impulse(dt_s):
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError('Positive finite internal time step required')

    def force(time_s):
        if not math.isfinite(time_s):
            raise ValueError('Finite time required')
        # Only solver node t=dt is active. This is a discrete identification
        # input, not a claim about the physical generator or a continuous pulse.
        return 1. if abs(time_s - dt_s) < dt_s * .25 else 0.

    return force


def reconstruct(kernel_voltage, target_samples):
    kernel = np.asarray(kernel_voltage)
    source = np.asarray(target_samples)
    if (np.iscomplexobj(kernel) or np.iscomplexobj(source) or kernel.ndim != 2
            or source.ndim != 1 or len(source) < 4 or kernel.shape[0] != len(source)
            or not 1 <= kernel.shape[1] <= 32 or not np.isfinite(kernel).all()
            or not np.isfinite(source).all() or source[0] != 0.):
        raise ValueError('Matching finite real internal-grid traces and F[0]=0 required')
    if np.any(kernel[:2] != 0.):
        raise ValueError('Interior impulse kernel must have zero voltage at nodes 0 and 1')
    # fftconvolve uses enough zero-padding for LINEAR rather than cyclic
    # convolution. A persistent lossless tail therefore cannot wrap to t=0.
    result = fftconvolve(kernel.astype(np.float64), source[1:, None].astype(np.float64),
                         mode='full', axes=0)[:len(source)]
    if not np.isfinite(result).all():
        raise ArithmeticError('Nonfinite reconstructed voltage')
    return result


def paired_metrics(reference, predicted):
    reference = np.asarray(reference, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if (reference.ndim != 2 or reference.shape[1] != 2 or predicted.shape != reference.shape
            or len(reference) < 4 or not np.isfinite(reference).all() or not np.isfinite(predicted).all()):
        raise ValueError('Matching two-channel voltage traces required')
    answer = metrics(predicted, reference)
    for name, a, b in [('top', reference[:, 0], predicted[:, 0]),
                       ('bottom', reference[:, 1], predicted[:, 1]),
                       ('S0-related', reference[:, 0] + reference[:, 1], predicted[:, 0] + predicted[:, 1]),
                       ('A0-related', reference[:, 0] - reference[:, 1], predicted[:, 0] - predicted[:, 1])]:
        answer[name].update(waveform_metrics(a, b))
        ea, eb = abs(hilbert(a)), abs(hilbert(b))
        answer[name]['envelope_pearson'] = (float(np.clip(np.corrcoef(ea, eb)[0, 1], -1., 1.))
                                             if min(np.std(ea), np.std(eb)) > 0 else None)
    return answer


def within_target(channels, threshold_percent=1.):
    return all(row['raw_l2_change_percent'] is not None
               and row['raw_l2_change_percent'] < threshold_percent for row in channels.values())


def common_grid(first_time, first_voltage, second_time, second_voltage):
    """Compare nested internal grids at exact shared nodes, without interpolation."""
    a, b = np.asarray(first_time), np.asarray(second_time)
    va, vb = np.asarray(first_voltage), np.asarray(second_voltage)
    if len(a) < 3 or len(b) < 3 or a[0] != 0 or b[0] != 0:
        raise ValueError('Zero-start time grids required')
    if len(a) > len(b):
        return common_grid(b, vb, a, va)[::-1]
    if (len(b) - 1) % (len(a) - 1):
        raise ValueError('Time grids are not nested')
    stride = (len(b) - 1) // (len(a) - 1)
    if not np.allclose(a, b[::stride], rtol=0, atol=1e-15):
        raise ValueError('Time grids do not share exact recording nodes')
    return va, vb[::stride]
