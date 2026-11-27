"""Common-delay ERA with an exact early FIR branch, not fitted alignment.

Delay separation follows Pelling/Sarradj, arXiv:2506.08870. The finite early
prefix is retained exactly so a threshold never deletes numerical precursors.
The same integer shift is used for both receivers. No record tail is padded.
"""
import numpy as np

from examples.pzt_era import kernel_data, reduced_impulse
from examples.pzt_era_matrix_free import fit_era


def split_record(time_s, voltage, prefix_relative_norm=1e-8, *, delay_samples=None):
    markov, dt = kernel_data(time_s, voltage)
    if not np.isfinite(prefix_relative_norm) or not 0 <= prefix_relative_norm < 1:
        raise ValueError('Finite prefix norm fraction in [0,1) required')
    # Include sums/differences so nearly cancelling channels cannot hide an
    # important precursor. No theoretical arrival or ground-truth timing fit.
    channels = [markov[:, j] for j in range(markov.shape[1])]
    if markov.shape[1] == 2:
        channels += [markov[:, 0] + markov[:, 1], markov[:, 0] - markov[:, 1]]
    limit = len(time_s) - 6
    if delay_samples is None:
        delay = limit
        for row in channels:
            norm2 = float(row @ row)
            if norm2 > 0:
                energy = np.r_[0., np.cumsum(row * row)]
                delay = min(delay, int(np.searchsorted(energy,
                    prefix_relative_norm**2 * norm2, side='right') - 1))
    else:
        if type(delay_samples) is not int or not 0 <= delay_samples <= limit:
            raise ValueError('Delay must leave at least six recording nodes')
        delay = delay_samples
    if not np.any(markov[delay:]):
        raise ValueError('Delayed dynamic branch is identically zero')
    tail = np.vstack([np.zeros((2, markov.shape[1])), markov[delay:]])
    ratios = [float(np.linalg.norm(row[:delay]) / np.linalg.norm(row))
              if np.linalg.norm(row) else 0. for row in channels]
    info = dict(delay_samples=delay, delay_s=delay * dt,
        early_prefix_retained_exactly=True, precursor_discarded=False,
        prefix_relative_norm_by_channel_and_sum_difference=ratios,
        prefix_relative_norm_budget=prefix_relative_norm,
        relative_receiver_timing_preserved=True, fitted_alignment=False,
        original_samples=len(time_s), dynamic_samples=len(tail),
        future_samples_padded=False, stability_guaranteed=False)
    return np.arange(len(tail)) * dt, tail, markov[:delay].copy(), info


def fit_delayed(time_s, voltage, max_order, *, backend='cupy', event=None,
                prefix_relative_norm=1e-8, delay_samples=None):
    t, tail, prefix, info = split_record(time_s, voltage, prefix_relative_norm,
                                       delay_samples=delay_samples)
    model, diagnostic = fit_era(t, tail, max_order, backend=backend, event=event)
    model.update(exact_prefix=prefix, original_samples=len(time_s))
    diagnostic['delay_handling'] = info
    return model, diagnostic


def delayed_impulse(model, order, samples, *, event=None, diagnostic_only=False):
    prefix = np.asarray(model['exact_prefix'])
    if (prefix.ndim != 2 or prefix.shape[1] != len(model['c'])
            or not np.isfinite(prefix).all() or np.iscomplexobj(prefix)
            or samples != int(model['original_samples'])):
        raise ValueError('Exact finite prefix and original recording length required')
    tail, info = reduced_impulse(model, order, samples - len(prefix),
                                 event=event, diagnostic_only=diagnostic_only)
    result = np.zeros((samples, prefix.shape[1]))
    result[2:2 + len(prefix)] = prefix
    result[2 + len(prefix):] = tail[2:]
    info.update(delay_samples=len(prefix), early_prefix_retained_exactly=True,
                original_samples=samples, fitted_alignment=False)
    return result, info
