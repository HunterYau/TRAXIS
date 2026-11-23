"""Explicit common-band sampling; not an identified instrument transfer model."""
from fractions import Fraction
import numpy as np
from scipy.signal import resample_poly


def antialiased_samples(time_s,values,target_time_s,rate_hz=5000000):
    t=np.asarray(time_s,dtype=float);v=np.asarray(values,dtype=float);target=np.asarray(target_time_s,dtype=float)
    if (t.ndim!=1 or len(t)<3 or v.ndim not in (1,2) or v.shape[0]!=len(t)
            or not np.isfinite(t).all() or not np.isfinite(v).all() or not np.isfinite(target).all()
            or target.ndim!=1 or not np.isfinite(rate_hz) or rate_hz<=0):
        raise ValueError('Finite matching uniform traces required')
    if len(target)>1 and np.any(np.diff(target)<=0):raise ValueError('Increasing target times required')
    dt=t[1]-t[0]
    if dt<=0 or not np.allclose(np.diff(t),dt,rtol=1e-9,atol=1e-15):
        raise ValueError('Uniform increasing input time required')
    if len(target) and (target[0]<t[0]-1e-15 or target[-1]>t[-1]+1e-15):
        raise ValueError('Requested time exceeds saved recording')
    ratio=Fraction(rate_hz*dt).limit_denominator(100000)
    if not np.isclose(float(ratio),rate_hz*dt,rtol=1e-10,atol=0):
        raise ValueError('Input/output rates cannot be represented accurately within filter bound')
    filtered=resample_poly(v,ratio.numerator,ratio.denominator,axis=0,window=('kaiser',10.))
    grid=t[0]+np.arange(len(filtered))/(float(ratio)/dt)
    if v.ndim==1:return np.interp(target,grid,filtered)
    return np.column_stack([np.interp(target,grid,filtered[:,j]) for j in range(v.shape[1])])
