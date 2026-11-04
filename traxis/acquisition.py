"""Stable action identities and specimen-level split assignments."""

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from traxis.config import Plate, positive


def sensor_layout(plate: Plate, count: int = 8, inset_m: float = 0.02) -> np.ndarray:
    if not math.isfinite(inset_m) or not 0 < inset_m < min(plate.width_m, plate.height_m) / 2:
        raise ValueError("sensor inset must be inside the plate")
    w, h, a = plate.width_m, plate.height_m, inset_m
    if count == 8:
        # Counterclockwise corner/midpoint order, beginning at bottom left.
        return np.array([[a, a], [w / 2, a], [w - a, a], [w - a, h / 2],
                         [w - a, h - a], [w / 2, h - a], [a, h - a], [a, h / 2]])
    if count == 12:
        theta = np.arange(count) * 2 * np.pi / count
        return np.column_stack([w / 2 + (w / 2 - a) * np.cos(theta),
                                h / 2 + (h / 2 - a) * np.sin(theta)])
    raise ValueError("supported layouts are eight corner/edge or twelve ring sensors")


@dataclass(frozen=True)
class Action:
    transmitter: int
    receiver: int
    frequency_hz: float


def action_catalog(sensor_count: int, frequencies_hz: tuple[float, ...],
                   reciprocal: bool = True) -> tuple[Action, ...]:
    if type(sensor_count) is not int or sensor_count < 2:
        raise ValueError("need at least two sensors")
    if not frequencies_hz or len(set(frequencies_hz)) != len(frequencies_hz):
        raise ValueError("frequencies must be nonempty and unique")
    for f in frequencies_hz:
        positive("frequency", f)
    return tuple(Action(tx, rx, float(f)) for tx in range(sensor_count)
                 for rx in range(sensor_count) if tx != rx and (not reciprocal or tx < rx)
                 for f in frequencies_hz)


def specimen_split(specimen_id: str, seed: int = 2026,
                   fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> str:
    """Order-independent deterministic assignment; variants must share a parent ID.

    Hashing gives approximate proportions, not an exact stratified split. For
    small physical datasets use a separately reviewed explicit manifest.
    """
    if not specimen_id or not isinstance(specimen_id, str):
        raise ValueError("specimen ID must be a nonempty string")
    if len(fractions) != 3 or any(not math.isfinite(v) or v < 0 for v in fractions) or not math.isclose(sum(fractions), 1):
        raise ValueError("three nonnegative fractions must sum to one")
    digest = hashlib.sha256(f"{seed}:{specimen_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if value < fractions[0] else "validation" if value < sum(fractions[:2]) else "test"
