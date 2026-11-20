"""Explicit hardware assumptions; never infer gain from received waveforms."""
import math
import numpy as np


def applied_actuator_voltage(time_s, generator_time_s, generator_voltage_v, amplifier):
    gain=amplifier['gain_v_v'];delay=amplifier['delay_s']
    if not math.isfinite(gain) or gain<=0 or not math.isfinite(delay) or delay<0:
        raise ValueError('amplifier needs finite positive gain and nonnegative delay')
    return gain*np.interp(np.asarray(time_s)-delay,generator_time_s,generator_voltage_v,left=0.,right=0.)
