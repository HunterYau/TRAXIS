"""Fundamental Rayleigh-Lamb modes of an infinite isotropic free plate.

The pole-free traction determinants follow the equations in Perelli's thesis,
section 1.3: https://amsdottorato.unibo.it/6321/1/Perelli_Alessandro_tesi.pdf
Thickness is the full thickness; h in the determinant is half that value.
This reference omits finite edges, losses, recesses, bonding and PZT loading.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq

from traxis.config import Material, positive


def _sin_over_z(z: complex) -> complex:
    return np.sin(z) / z if abs(z) > 1e-8 else 1 - z**2 / 6 + z**4 / 120


def _determinant(kh: float, wh_over_ct: float, ct_over_cl: float, mode: str) -> float:
    """Real traction determinant, scaled by its two term magnitudes.

    Divide the S determinant by qh and the A determinant by ph to remove
    spurious roots at q=0 or p=0. No tangent poles are searched as zeros.
    """
    p = np.sqrt(complex((wh_over_ct * ct_over_cl)**2 - kh**2))
    q = np.sqrt(complex(wh_over_ct**2 - kh**2))
    a = (q**2 - kh**2)**2
    if mode == "S0":
        first = a * np.cos(p) * _sin_over_z(q)
        second = 4 * kh**2 * p * np.sin(p) * np.cos(q)
    else:
        first = a * _sin_over_z(p) * np.cos(q)
        second = 4 * kh**2 * q * np.cos(p) * np.sin(q)
    denominator = abs(first) + abs(second)
    if denominator == 0 or not np.isfinite(denominator):
        raise ArithmeticError("ill-conditioned Rayleigh-Lamb determinant")
    result = (first + second) / denominator
    if abs(result.imag) > 1e-10:
        raise ArithmeticError("traction determinant unexpectedly complex")
    return float(result.real)


def wavenumber_rad_m(frequency_hz: float, thickness_m: float, mode: str,
                     material: Material = Material()) -> float:
    positive("frequency", frequency_hz)
    positive("thickness", thickness_m)
    if mode not in ("A0", "S0"):
        raise ValueError("supported modes are A0 and S0")
    cl, ct = material.bulk_speeds_m_s
    fd = frequency_hz * thickness_m / ct
    if not 1e-4 <= fd <= 0.25:
        raise ValueError("fundamental-mode search supports 1e-4 <= f*thickness/cT <= 0.25")
    h = thickness_m / 2
    w = 2 * math.pi * frequency_hz * h / ct
    if mode == "S0":
        lo, hi = w * ct / cl * (1 + 1e-9), w * (1 - 1e-9)
    else:
        lo, hi = w * (1 + 1e-9), max(10 * math.sqrt(w), 20 * w)
    grid = np.geomspace(lo, hi, 512)
    values = [_determinant(k, w, ct / cl, mode) for k in grid]
    roots = []
    for i in range(len(grid) - 1):
        if values[i] == 0:
            roots.append(grid[i])
        elif values[i] * values[i + 1] < 0:
            roots.append(brentq(_determinant, grid[i], grid[i + 1], args=(w, ct / cl, mode),
                                xtol=1e-13, rtol=1e-12))
    if len(roots) != 1:
        raise ArithmeticError(f"expected one {mode} root in low-frequency bracket, found {len(roots)}")
    return roots[0] / h


@dataclass(frozen=True)
class ModeVelocity:
    mode: str
    frequency_hz: float
    thickness_m: float
    wavenumber_rad_m: float
    phase_m_s: float
    group_m_s: float
    wavelength_m: float


def mode_velocity(frequency_hz: float, thickness_m: float, mode: str,
                  material: Material = Material(), derivative_fraction: float = 1e-3) -> ModeVelocity:
    if not math.isfinite(derivative_fraction) or not 1e-5 <= derivative_fraction <= 0.01:
        raise ValueError("frequency derivative fraction must lie in [1e-5, 0.01]")
    k = wavenumber_rad_m(frequency_hz, thickness_m, mode, material)
    df = frequency_hz * derivative_fraction
    ct = material.bulk_speeds_m_s[1]
    low_f, high_f = 1e-4 * ct / thickness_m, 0.25 * ct / thickness_m
    if frequency_hz - df < low_f:
        k1 = wavenumber_rad_m(frequency_hz + df, thickness_m, mode, material)
        k2 = wavenumber_rad_m(frequency_hz + 2 * df, thickness_m, mode, material)
        dk_df = (-3 * k + 4 * k1 - k2) / (2 * df)
    elif frequency_hz + df > high_f:
        k1 = wavenumber_rad_m(frequency_hz - df, thickness_m, mode, material)
        k2 = wavenumber_rad_m(frequency_hz - 2 * df, thickness_m, mode, material)
        dk_df = (3 * k - 4 * k1 + k2) / (2 * df)
    else:
        kp = wavenumber_rad_m(frequency_hz + df, thickness_m, mode, material)
        km = wavenumber_rad_m(frequency_hz - df, thickness_m, mode, material)
        dk_df = (kp - km) / (2 * df)
    group = 2 * math.pi / dk_df
    if not math.isfinite(group) or group <= 0:
        raise ArithmeticError("invalid fundamental-mode group velocity")
    return ModeVelocity(mode, frequency_hz, thickness_m, k, 2 * math.pi * frequency_hz / k,
                        group, 2 * math.pi / k)
