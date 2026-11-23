"""Explicit candidate reconstruction of sampled drive; not calibrated hardware.

Default production interpolation is deliberately not changed by this module.
The FIR removes sampling images while retaining the recorded baseband waveform.
"""
import numpy as np
from scipy.signal import resample_poly


def reconstructed_drive(time_s, voltage_v, *, up=32):
    t=np.asarray(time_s,dtype=float);v=np.asarray(voltage_v,dtype=float)
    if (t.ndim!=1 or v.shape!=t.shape or len(t)<3 or not np.isfinite(t).all()
            or not np.isfinite(v).all() or not np.all(np.diff(t)>0)
            or not np.allclose(np.diff(t),t[1]-t[0],rtol=1e-9,atol=1e-15)
            or type(up) is not int or not 2<=up<=64):
        raise ValueError('Uniform finite drive samples and interpolation factor 2..64 required')
    peak=float(np.max(abs(v)))
    if peak==0:raise ValueError('Nonzero drive required')
    # Zero-pad both ends before the symmetric FIR, retaining its pre/post ringing.
    # For t<0 the simulation still starts at rest; no claim of exact causal DAC modeling.
    padding=32
    signal=resample_poly(np.pad(v/peak,(padding,padding)),up,1,window=('kaiser',10.))
    grid=t[0]+(np.arange(len(signal))/up-padding)*(t[1]-t[0])
    def value(time):return np.interp(time,grid,signal,left=0.,right=0.)
    return value,dict(method='candidate_polyphase_reconstruction',up=up,
        source_sample_rate_hz=1/(t[1]-t[0]),filter='SciPy polyphase FIR, Kaiser beta=10',
        original_peak_normalization_v=peak,no_received_signal_fitting=True,
        hardware_calibrated=False,limitations=[
            'Actual NI generator digital-filter setting and terminal waveform are unknown.',
            'Zero-phase reconstruction has boundary ringing; not a causal hardware transfer model.',
            'Input time axis/sample rate is taken from the public stored waveform, not assumed DAC clock.'])
