"""Known Hann toneburst, verified against recorded generator samples before use."""
import math
import numpy as np


def analytic_source(frequency_hz,num_cycles,time_s,generator_v):
    if (not math.isfinite(frequency_hz) or frequency_hz<=0 or type(num_cycles) is not int
            or num_cycles<=0):raise ValueError('Positive frequency and cycle count required')
    t=np.asarray(time_s,dtype=float);v=np.asarray(generator_v,dtype=float)
    if (t.ndim!=1 or len(t)<3 or v.shape!=t.shape or not np.isfinite(t).all()
            or not np.isfinite(v).all() or t[0]!=0 or np.any(np.diff(t)<=0)):
        raise ValueError('Matching finite drive samples starting at zero required')
    duration=num_cycles/frequency_hz
    def pulse(time):
        if np.ndim(time)==0:
            time=float(time)
            if not math.isfinite(time):raise ValueError('Finite drive evaluation time required')
            return (math.sin(2*math.pi*frequency_hz*time)*math.sin(math.pi*time/duration)**2
                    if 0<time<duration else 0.)
        time=np.asarray(time,dtype=float)
        if not np.isfinite(time).all():raise ValueError('Finite drive evaluation times required')
        return np.where((time>0)&(time<duration),
            np.sin(2*np.pi*frequency_hz*time)*np.sin(np.pi*time/duration)**2,0.)
    expected=pulse(t)
    error=float(np.max(abs(v-expected)))
    if not np.allclose(v,expected,rtol=1e-12,atol=1e-12):
        raise ValueError('Recorded generator differs from the stated analytic pulse; refusing to fit it')
    peak=float(np.max(abs(v)))
    if peak==0:raise ValueError('Nonzero recorded generator required')
    def normalized(time):return pulse(time)/peak
    return normalized,dict(method='exact_five_cycle_Hann_sine' if num_cycles==5 else 'exact_Hann_sine',
        frequency_hz=frequency_hz,num_cycles=num_cycles,pulse_end_s=duration,
        formula='sin(2*pi*f*t)*sin(pi*t/T)^2 for 0<t<T; zero elsewhere; T=cycles/f',
        normalization='Divide by the original recorded sample peak, identical to the linear-drive control',
        recorded_sample_peak_v=peak,max_recorded_sample_difference_v=error,
        received_signal_fitted=False,hardware_transfer_calibrated=False)
