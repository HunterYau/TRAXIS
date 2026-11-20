"""Documented narrowband effective damping, not a fitted specimen property."""
import math


def damping_parameters(profile, frequency_hz):
    if not math.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError('damping reference frequency must be finite and positive')
    if profile['model'] != 'narrowband_mass_proportional':
        raise ValueError('unsupported damping model')
    factors=[]
    for mode in ('longitudinal','transverse'):
        row=profile['bulk_measurements'][mode]
        attenuation=row['attenuation_db_per_m_per_mhz'];speed=row['speed_m_s']
        if not all(math.isfinite(x) and x > 0 for x in (attenuation,speed)):
            raise ValueError('attenuation and speed must be finite and positive')
        # A(x)=A0 exp(-a*x), dB=20 log10(A0/A).
        # Weak-loss eta ~= 2*a*c/omega at 1 MHz.
        factors.append(2*(attenuation*math.log(10)/20)*speed/(2*math.pi*1e6))
    eta=sum(factors)/len(factors)
    return dict(model=profile['model'],reference_frequency_hz=frequency_hz,
                loss_factor=eta,mass_damping_per_s=eta*2*math.pi*frequency_hz,
                assumed_material=profile['material'],specimen_calibrated=False,
                scope='effective whole coupled assembly; not separate aluminum/PZT/glue loss tensors')
