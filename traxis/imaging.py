"""Baseline-subtracted envelope delay-and-sum imaging.

The bistatic path delay follows Sharif Khodaei & Aliabadi (2014),
https://doi.org/10.1088/0964-1726/23/7/075007 . The image retains signal units;
it is not a quantitative material-loss estimate without separate calibration.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.signal import hilbert

from traxis.bank import Observation
from traxis.config import positive
from traxis.resources import ResourceGuard


@dataclass(frozen=True)
class DASSettings:
    group_velocity_m_s: dict[float, float]
    time_offset_s: float = 0.0
    shape: tuple[int, int] = (128, 128)

    def __post_init__(self):
        if not self.group_velocity_m_s:
            raise ValueError("provide a group velocity for every excitation frequency")
        for frequency, speed in self.group_velocity_m_s.items():
            positive("frequency", frequency)
            positive("group velocity", speed)
        if not math.isfinite(self.time_offset_s):
            raise ValueError("time offset must be finite")
        if len(self.shape) != 2 or any(type(n) is not int or not 1 <= n <= 512 for n in self.shape):
            raise ValueError("DAS shape must contain two integers in [1,512]")


@dataclass(frozen=True)
class DASImage:
    intensity: np.ndarray
    coverage: np.ndarray
    record_count: int


def delay_and_sum(observation: Observation, settings: DASSettings,
                  guard: ResourceGuard | None = None) -> DASImage:
    """Average residual-envelope evidence over acquired records only.

    `time_offset_s` includes drive-envelope and acquisition delays, estimated
    from training/control measurements. Out-of-record arrivals contribute zero.
    Coverage reports the fraction of acquired paths with in-record arrival time.
    No per-image/per-record maximum normalization discards amplitude information.
    """
    ny, nx = settings.shape
    p = observation.plate
    x = (np.arange(nx) + 0.5) * p.width_m / nx
    y = (np.arange(ny) + 0.5) * p.height_m / ny
    image, coverage = np.zeros(settings.shape), np.zeros(settings.shape)
    indices = np.flatnonzero(observation.observed)
    for index in indices:
        if guard:
            guard.check()
        a = observation.actions[index]
        if a.frequency_hz not in settings.group_velocity_m_s:
            raise ValueError(f"missing group velocity for {a.frequency_hz:g} Hz")
        residual = (observation.waveforms[index] - observation.baseline[index]).astype(float)
        # Zero padding reduces FFT wraparound; it cannot fix truncated packets.
        pad = len(residual) // 2
        envelope = abs(hilbert(np.pad(residual, (pad, pad))))[pad:pad + len(residual)]
        tx, rx = observation.sensors_xy_m[[a.transmitter, a.receiver]]
        distance = np.hypot(x[None, :] - tx[0], y[:, None] - tx[1])
        distance += np.hypot(x[None, :] - rx[0], y[:, None] - rx[1])
        arrival = settings.time_offset_s + distance / settings.group_velocity_m_s[a.frequency_hz]
        inside = (arrival >= observation.time_s[0]) & (arrival <= observation.time_s[-1])
        image += np.interp(arrival.ravel(), observation.time_s, envelope, left=0, right=0).reshape(settings.shape)
        coverage += inside
    if len(indices):
        image /= len(indices)
        coverage /= len(indices)
    return DASImage(image.astype(np.float32), coverage.astype(np.float32), len(indices))
