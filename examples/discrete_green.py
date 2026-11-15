"""Replay a fixed linear Verlet system from its discrete unit-force response.

The response starts at zero displacement/velocity and uses force[1] = 1 N,
with all other force samples zero. For an input whose force[0] is zero,
u[n] = sum(g[n-j+1] * f[j], j=1..n-1). The saved response already contains
the integrator's time-step scaling, so the convolution uses force in Newtons.
"""
import math

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft


def synthesize(green_m_per_n, force_n, *, max_workspace_mib=512):
    green,force=np.asarray(green_m_per_n),np.asarray(force_n)
    if (green.dtype.kind not in 'fiu' or force.dtype.kind not in 'fiu'
            or green.ndim!=2 or not 3<=green.shape[0]<=500001 or not 1<=green.shape[1]<=32
            or force.shape!=(green.shape[0],) or not np.isfinite(green).all()
            or not np.isfinite(force).all() or force[0]!=0 or np.any(green[:2]!=0)
            or not math.isfinite(max_workspace_mib) or not 0<max_workspace_mib<=4096):
        raise ValueError('Green replay needs finite real traces, matching force, zero initial samples and a bounded workspace')
    n,receivers=green.shape
    fft_size=next_fast_len(2*n-2,real=True)
    # Include both spectra, their product, inverse output, copies and inputs.
    workspace_mib=(32*fft_size*receivers+16*n*receivers+16*fft_size)/1024**2
    if workspace_mib>max_workspace_mib:
        raise MemoryError('Green replay exceeds its declared FFT workspace')
    green,force=green.astype(np.float64,copy=False),force.astype(np.float64,copy=False)
    spectrum=rfft(green,n=fft_size,axis=0,workers=1)
    spectrum*=rfft(force[1:],n=fft_size,workers=1)[:,None]
    answer=irfft(spectrum,n=fft_size,axis=0,workers=1)[:n].copy()
    answer[:2]=0.  # Exact initial conditions of this discrete input convention.
    if not np.isfinite(answer).all():raise ArithmeticError('Green replay produced a nonfinite trace')
    return answer,dict(method='discrete_verlet_zero_padded_fft',impulse_force_index=1,
        force_samples=n,receivers=receivers,fft_size=fft_size,workspace_screen_mib=workspace_mib)
