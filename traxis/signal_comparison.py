"""Pointwise voltage comparisons; no fitted time, polarity or gain alignment."""
import numpy as np


def waveform_metrics(measured, predicted):
    measured=np.asarray(measured,dtype=float);predicted=np.asarray(predicted,dtype=float)
    if (measured.ndim!=1 or predicted.shape!=measured.shape or len(measured)<2
            or not np.isfinite(measured).all() or not np.isfinite(predicted).all()):
        raise ValueError('waveforms must be matching finite one-dimensional arrays')
    real_norm=float(np.linalg.norm(measured));sim_norm=float(np.linalg.norm(predicted))
    difference=predicted-measured
    return dict(
        rmse_v=float(np.sqrt(np.mean(difference**2))),
        raw_voltage_difference_percent=(100*float(np.linalg.norm(difference))/real_norm
                                        if real_norm>0 else None),
        normalized_wiggle_difference_percent=(100*float(np.linalg.norm(
            predicted/sim_norm-measured/real_norm)) if min(real_norm,sim_norm)>0 else None),
        correlation=(float(np.clip(np.corrcoef(measured,predicted)[0,1],-1.,1.))
                     if np.std(measured)>0 and np.std(predicted)>0 else None),
        measured_peak_to_peak_v=float(np.ptp(measured)),
        simulated_peak_to_peak_v=float(np.ptp(predicted)))
