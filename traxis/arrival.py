"""Independent, theory-guided adaptive envelope packet localization.

The 10%-envelope packet boundary follows Zeng et al., Sensors 2016,
doi:10.3390/s16101623. The bounded search and noise/ambiguity safeguards
are project choices, not a reproduction of that paper's compensation method.
"""
from functools import lru_cache
import numpy as np
from scipy.signal import find_peaks, hilbert
from traxis.config import Material
from traxis.dispersion import mode_velocity


@lru_cache(maxsize=128)
def _prediction(frequency,mode,distance,thickness,young,poisson,density,cycles):
    material=Material(young_pa=young,poisson=poisson,density_kg_m3=density)
    # Five-cycle Hann toneburst main-lobe approximation, not measured bandwidth.
    band=frequency*np.linspace(1-2/cycles,1+2/cycles,11)
    speeds=np.array([mode_velocity(float(f),thickness,mode,material).group_m_s for f in band])
    center=distance/mode_velocity(frequency,thickness,mode,material).group_m_s
    return center,float(np.min(distance/speeds)),float(np.max(distance/speeds)),float(np.max(speeds))


def adaptive_packet(t,signal,frequency,mode,*,distance_m=.25,thickness_m=.001,
                    material=Material(young_pa=71e9,poisson=.33,density_kg_m3=2700.),
                    cycles=5,reflection_distance_m=.75):
    t=np.asarray(t,dtype=float);signal=np.asarray(signal,dtype=float)
    if (t.ndim!=1 or signal.shape!=t.shape or len(t)<2 or not np.isfinite(t).all()
            or not np.isfinite(signal).all() or np.any(np.diff(t)<=0)
            or frequency<=0 or mode not in ('S0','A0') or cycles<3 or distance_m<=0
            or thickness_m<=0 or reflection_distance_m<=distance_m):
        raise ValueError('invalid arrival recording or geometry')
    grid=np.arange(1750)*2e-7
    if t[0]>grid[0] or t[-1]<grid[-1]:
        raise ValueError('incomplete shared interval')
    y=np.interp(grid,t,signal);y-=y[grid<20e-6].mean()
    envelope=abs(hilbert(y,8192)[:len(grid)])
    flight,earliest,latest,max_speed=_prediction(float(frequency),mode,float(distance_m),
        float(thickness_m),material.young_pa,material.poisson,material.density_kg_m3,cycles)
    duration=cycles/frequency
    # A declared 25% theoretical travel-time allowance; never fitted to agreement.
    lower=max(20e-6,.75*earliest)
    reflection=reflection_distance_m/max_speed
    upper=min(grid[-1],1.25*latest+duration,reflection)
    selected=np.flatnonzero((grid>=lower)&(grid<=upper))
    if len(selected)<3:
        raise ValueError('no direct-arrival search interval in saved record')
    noise=envelope[grid<min(20e-6,lower/2)]
    median=float(np.median(noise))
    noise_floor=median+6*1.4826*float(np.median(abs(noise-median)))
    local=envelope[selected];maximum=float(np.max(local))
    threshold=max(noise_floor,.1*maximum)
    ids,_=find_peaks(local,height=threshold,prominence=max(noise_floor,.1*maximum),
                     distance=max(1,int(.5/frequency/2e-7)))
    peak=int(selected[int(np.argmax(local))])
    strong=[int(selected[i]) for i in ids if local[i]>=.7*maximum]
    ambiguous=any(abs(grid[i]-grid[peak])>=duration/2 for i in strong)
    left=right=peak
    packet_threshold=max(noise_floor,.1*float(envelope[peak]))
    while left>selected[0] and envelope[left-1]>packet_threshold:left-=1
    while right<selected[-1] and envelope[right+1]>packet_threshold:right+=1
    edge=peak in (selected[0],selected[-1])
    clipped=left==selected[0] or right==selected[-1]
    status=('no_signal' if maximum<=max(noise_floor,np.finfo(float).tiny) else
            'ambiguous_packets' if ambiguous else 'unresolved_arrival' if edge or clipped else 'localized_packet')
    travel=float(grid[peak]-duration/2)
    if travel<=0 and status=='localized_packet':status='unresolved_arrival'
    return dict(method='theory_guided_adaptive_envelope_v1',status=status,
        packet_peak_us=float(grid[peak]*1e6),flight_us=travel*1e6,
        predicted_flight_us=flight*1e6,predicted_peak_us=(flight+duration/2)*1e6,
        window_us=[float(grid[left]*1e6),float(grid[right]*1e6)],
        search_gate_us=[float(grid[selected[0]]*1e6),float(grid[selected[-1]]*1e6)],
        same_mode_reflection_bound_us=reflection*1e6,window_edge=bool(edge),packet_clipped=bool(clipped),
        noise_threshold=noise_floor,packet_threshold=packet_threshold,
        candidate_peak_us=[float(grid[i]*1e6) for i in strong],
        limitations='Envelope-peak travel-time proxy, not onset. Mode identity and converted/reflected-mode contamination remain unverified.')
